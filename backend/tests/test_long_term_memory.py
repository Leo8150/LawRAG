"""Offline tests for checkpointed short-term and dual-store long-term memory."""

import asyncio
from typing import TypedDict

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph


class FakeMemoryVectorStore:
    def __init__(self):
        self.documents: dict[str, Document] = {}
        self.add_batches = 0

    def delete(self, ids):
        for memory_id in ids:
            self.documents.pop(memory_id, None)

    def add_texts(self, texts, metadatas, ids):
        self.add_batches += 1
        for text, metadata, memory_id in zip(texts, metadatas, ids):
            self.documents[memory_id] = Document(page_content=text, metadata=metadata)
        return ids

    def similarity_search(self, query, k, filter):
        return [
            document
            for document in self.documents.values()
            if document.metadata.get("user_id") == filter.get("user_id")
        ][:k]


def test_long_term_memory_deduplicates_and_separates_confirmation(tmp_path):
    from app.memory.long_term import LongTermMemoryService
    from app.memory.repository import LongTermMemoryRepository

    repository = LongTermMemoryRepository(f"sqlite:///{tmp_path / 'memory.db'}")
    vectorstore = FakeMemoryVectorStore()
    service = LongTermMemoryService(repository, vectorstore)

    first = asyncio.run(service.remember_confirmed(
        "user-a",
        [{"category": "case_fact", "content": "劳动合同于2026年1月签订"}],
        "thread-a",
    ))
    second = asyncio.run(service.remember_confirmed(
        "user-a",
        [{"category": "case_fact", "content": "劳动合同于2026年1月签订"}],
        "thread-b",
    ))
    episode_id = asyncio.run(service.remember_episode(
        "user-a",
        "thread-a",
        "公司拖欠工资怎么办？",
        "可以申请劳动仲裁。",
        [{"node": "execute_tools", "tools": ["retrieve_legal_evidence"]}],
        ["劳动合同法"],
    ))

    assert first == second
    assert episode_id is not None
    rows = repository.list_user("user-a")
    assert len(rows) == 2
    semantic = next(row for row in rows if row.memory_type == "semantic")
    episodic = next(row for row in rows if row.memory_type == "episodic")
    assert semantic.is_confirmed is True
    assert semantic.source_kind == "user_confirmed"
    assert episodic.is_confirmed is False
    assert episodic.source_kind == "agent_episode"

    recalled = asyncio.run(service.recall("user-a", "劳动合同工资"))
    context = service.format_context(recalled)
    assert "用户已确认" in context
    assert "不得视为已确认事实" in context
    assert all(item.memory_id in vectorstore.documents for item in recalled)


def test_short_term_builder_compresses_history_and_carries_confirmed_facts(monkeypatch):
    from app.config import settings
    from app.memory.checkpoint import ShortTermContextBuilder, ShortTermSnapshot

    monkeypatch.setattr(settings, "SHORT_TERM_RECENT_TURNS", 2)
    monkeypatch.setattr(settings, "SHORT_TERM_SUMMARY_MAX_CHARS", 500)
    snapshot = ShortTermSnapshot(summary="", recent_turns=[], case_facts=[], turn_count=0)
    for index in range(4):
        snapshot = ShortTermContextBuilder.advance(
            snapshot,
            f"第{index}问",
            f"第{index}答",
            ["用户已确认已经签订书面合同"] if index == 0 else [],
        )

    assert len(snapshot.recent_turns) == 2
    assert "第0问" in snapshot.summary
    assert snapshot.turn_count == 4
    assert snapshot.case_facts == ["用户已确认已经签订书面合同"]
    context = ShortTermContextBuilder.build_context(snapshot)
    assert "历史消息压缩摘要" in context
    assert "用户已确认的本案事实" in context
    resolved = ShortTermContextBuilder.resolve_question("那如果已经付款呢？", snapshot)
    assert "上一轮问题" in resolved


def test_remember_turn_batches_semantic_and_episode_embeddings(tmp_path):
    from app.memory.long_term import LongTermMemoryService
    from app.memory.repository import LongTermMemoryRepository

    repository = LongTermMemoryRepository(f"sqlite:///{tmp_path / 'turn-memory.db'}")
    vectorstore = FakeMemoryVectorStore()
    service = LongTermMemoryService(repository, vectorstore)

    memory_ids = asyncio.run(service.remember_turn(
        user_id="user-a",
        thread_id="thread-a",
        confirmed_items=[
            {"category": "case_fact", "content": "用户确认工资已拖欠三个月"},
            {"category": "preference", "content": "用户希望回答先给结论"},
        ],
        question="公司拖欠工资怎么办？",
        answer="可以准备证据后申请劳动仲裁。",
        trace=[{"node": "execute_tools", "tools": ["retrieve_legal_evidence"]}],
        source_ids=["劳动合同法"],
    ))

    assert len(memory_ids) == 3
    assert vectorstore.add_batches == 1
    assert len(repository.list_user("user-a")) == 3


def test_langgraph_checkpointer_isolates_thread_ids():
    from langgraph.checkpoint.memory import InMemorySaver

    class CounterState(TypedDict, total=False):
        count: int

    async def increment(state: CounterState):
        return {"count": state.get("count", 0) + 1}

    builder = StateGraph(CounterState)
    builder.add_node("increment", increment)
    builder.add_edge(START, "increment")
    builder.add_edge("increment", END)
    graph = builder.compile(checkpointer=InMemorySaver())

    async def scenario():
        thread_a = {"configurable": {"thread_id": "thread-a"}}
        thread_b = {"configurable": {"thread_id": "thread-b"}}
        first_a = await graph.ainvoke({"count": 0}, config=thread_a)
        second_a = await graph.ainvoke({}, config=thread_a)
        first_b = await graph.ainvoke({"count": 10}, config=thread_b)
        return first_a, second_a, first_b

    first_a, second_a, first_b = asyncio.run(scenario())
    assert first_a["count"] == 1
    assert second_a["count"] == 2
    assert first_b["count"] == 11


def test_chat_request_memory_defaults_are_privacy_safe():
    from app.models.schemas import ChatRequest

    request = ChatRequest(question="测试记忆")
    assert request.memory_mode == "off"
    assert request.user_id is None
    assert request.confirmed_memories == []
