"""Offline tests for the single-path Agentic RAG graph."""

import asyncio
from types import SimpleNamespace

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


class FakeConfig:
    collection_names = ["laws", "cases"]
    top_k = 5
    generation_strategy = SimpleNamespace(value="standard")

    def to_dict(self):
        return {"collection_names": self.collection_names, "top_k": self.top_k}


class FakePipeline:
    def __init__(self):
        self.config = FakeConfig()
        self.repository = None
        self.metrics = {
            "llm_calls": 0,
            "memory_turns": 0,
            "retrieved_child_count": 0,
            "reranked_child_count": 0,
            "parent_candidate_count": 0,
        }

    async def _select_skill(self, question):
        from app.skills.loader import LegalSkill

        skill = LegalSkill(
            name="general_legal",
            description="通用法律问题",
            display_name="通用法律",
            collection_names=("laws", "cases"),
            use_kg=False,
            context_priority={"law": 0},
            content="使用权威法律证据回答。",
        )
        return skill, [HumanMessage(content=question)]

    async def _retrieve(self, queries, hyde):
        return [Document(
            page_content="民法典规定依法成立的合同受法律保护。",
            metadata={"child_chunk_id": "c1", "parent_chunk_id": "p1", "doc_type": "law"},
        )]

    async def _rerank(self, question, documents, top_k=None):
        documents[0].metadata["rerank_score"] = 0.9
        return documents

    def _aggregate_and_hydrate_parents(self, children):
        return [Document(
            page_content="《中华人民共和国民法典》依法成立的合同，受法律保护。",
            metadata={
                "parent_chunk_id": "p1",
                "doc_type": "law",
                "law_name": "中华人民共和国民法典",
                "rerank_score": 0.9,
            },
        )]

    async def _kg_lookup(self, question, allow_llm_fallback=False):
        return [], []

    async def _generate(self, question, context, memory_context, skill_messages, skill_catalog):
        return "根据[来源1]，依法成立的合同受法律保护。", False


class FakeAgentLLM:
    def bind_tools(self, tools, tool_choice="auto"):
        assert tool_choice == "auto"
        return self

    async def ainvoke(self, messages):
        system = str(messages[0].content)
        if "检索协调器" in system:
            return AIMessage(content="", tool_calls=[{
                "name": "retrieve_legal_evidence",
                "args": {"query": "合同法律效力", "collection": "laws", "top_k": 5},
                "id": "retrieve-1",
            }])
        if "证据充分性" in system:
            return AIMessage(content='{"sufficient": true, "missing_information": [], "follow_up_query": ""}')
        if "引用审查器" in system:
            return AIMessage(content='{"grounded": true, "citation_complete": true, "feedback": ""}')
        raise AssertionError(f"未预期的 Agent 调用: {system}")


def test_agentic_graph_runs_tool_grade_generate_and_grounding(monkeypatch):
    from app.agentic import runtime as runtime_module

    monkeypatch.setattr(runtime_module, "get_llm", lambda **kwargs: FakeAgentLLM())
    runner = runtime_module.AgenticRAGRunner(FakePipeline())
    result = asyncio.run(runner.run("合同成立后是否受法律保护？", "agentic-test-session"))

    nodes = [item["node"] for item in result.agent_trace]
    assert nodes == [
        "prepare",
        "load_skill",
        "decide_tools",
        "execute_tools",
        "grade_evidence",
        "compact_context",
        "generate",
        "verify_grounding",
    ]
    assert result.metrics.grounding_passed is True
    assert result.metrics.retrieval_rounds == 1
    assert result.pipeline_config["architecture"] == "agentic_rag"
    assert result.sources[0].metadata["parent_chunk_id"] == "p1"


def test_agent_policy_injects_mandatory_retrieval_when_model_skips_it(monkeypatch):
    from app.agentic import runtime as runtime_module

    class SkipToolLLM(FakeAgentLLM):
        async def ainvoke(self, messages):
            if "检索协调器" in str(messages[0].content):
                return AIMessage(content="不需要工具", tool_calls=[])
            return await super().ainvoke(messages)

    monkeypatch.setattr(runtime_module, "get_llm", lambda **kwargs: SkipToolLLM())
    runner = runtime_module.AgenticRAGRunner(FakePipeline())
    state = {
        "resolved_question": "合同是否有效？",
        "pending_query": "合同是否有效？",
        "messages": [],
        "evidence": [],
        "trace": [],
        "tool_rounds": 0,
    }
    update = asyncio.run(runner._decide_tools(state))

    assert update["tool_calls"][0]["name"] == "retrieve_legal_evidence"
    assert isinstance(update["messages"][-1], AIMessage)
