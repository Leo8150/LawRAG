"""可恢复上下文压缩、法律证据打包、Memory 与 Skill 的离线测试。"""

import asyncio

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage


def test_legal_evidence_packer_respects_budget_and_deduplicates():
    from app.context.compactor import LegalEvidencePacker

    law = Document(
        page_content="中华人民共和国刑法第二百六十四条。" * 100,
        metadata={"doc_type": "law", "law_name": "中华人民共和国刑法", "article_number": "264"},
    )
    duplicate = Document(page_content=law.page_content, metadata=dict(law.metadata))
    case = Document(
        page_content="基本案情。法院认为入户盗窃应当依法认定。裁判结果为有罪。" * 80,
        metadata={"doc_type": "case", "source_file": "case.json"},
    )

    result = LegalEvidencePacker(max_context_tokens=700).compact([case, law, duplicate])

    assert result.stages == [
        "evidence_budget",
        "per_document_snip",
        "deduplicate",
        "extractive_pack",
    ]
    assert result.tokens_after <= result.token_budget + 30  # 来源标签估算允许小幅误差
    assert sum("第二百六十四条" in d.page_content for d in result.documents) == 1
    assert "来源1" in result.context


def test_tool_result_budget_persists_large_latest_result(tmp_path):
    from app.context import MessageContextCompactor

    compactor = MessageContextCompactor(
        tmp_path,
        tool_result_budget_chars=200,
        large_result_char_limit=100,
        preview_chars=40,
    )
    messages = [
        AIMessage(content="", tool_calls=[{
            "name": "retrieve_legal_evidence",
            "args": {},
            "id": "large-result",
        }]),
        ToolMessage(content="法条证据" * 100, tool_call_id="large-result"),
    ]

    async def unused_summary(history, active_request):
        raise AssertionError("前三阶段能够处理时不应调用摘要模型")

    result = asyncio.run(compactor.prepare(messages, "当前法律问题", unused_summary))

    assert result.stages == ["tool_result_budget", "snip_compact"]
    assert len(result.persisted_outputs) == 1
    assert "Large tool result saved at" in result.messages[-1].content
    assert "法条证据" * 100 == open(result.persisted_outputs[0], encoding="utf-8").read()


def test_snip_compact_archives_history_without_orphaning_tool_result(tmp_path):
    from app.context import MessageContextCompactor, MessageCompactionResult

    compactor = MessageContextCompactor(tmp_path, max_messages=8)
    messages = [HumanMessage(content=f"历史消息 {index}") for index in range(8)]
    messages.extend([
        AIMessage(content="", tool_calls=[{
            "name": "search_statutes",
            "args": {},
            "id": "paired-call",
        }]),
        ToolMessage(content="法条结果", tool_call_id="paired-call"),
        HumanMessage(content="继续处理"),
    ])
    stats = MessageCompactionResult(messages=messages, chars_before=0, chars_after=0)

    compacted = compactor.snip_compact(messages, stats)

    tool_index = next(index for index, item in enumerate(compacted) if isinstance(item, ToolMessage))
    assert isinstance(compacted[tool_index - 1], AIMessage)
    assert compacted[tool_index].tool_call_id == "paired-call"
    assert stats.transcripts
    assert "messages archived" in next(
        item.content for item in compacted
        if isinstance(item, HumanMessage) and "messages archived" in str(item.content)
    )


def test_micro_compact_keeps_latest_unseen_results_and_persists_old_ones(tmp_path):
    from app.context import MessageContextCompactor, MessageCompactionResult

    compactor = MessageContextCompactor(tmp_path, keep_recent_results=1)
    messages = []
    for index in range(4):
        call_id = f"call-{index}"
        messages.extend([
            AIMessage(content="", tool_calls=[{"name": "tool", "args": {}, "id": call_id}]),
            ToolMessage(content=f"旧结果{index}" * 100, tool_call_id=call_id),
        ])
    stats = MessageCompactionResult(messages=messages, chars_before=0, chars_after=0)
    latest_content = messages[-1].content

    compacted = compactor.micro_compact(messages, 1200, stats)

    assert compacted[-1].content == latest_content
    assert any("Earlier tool result saved at" in str(item.content) for item in compacted)
    assert stats.persisted_outputs


def test_compact_history_runs_only_after_deterministic_stages(tmp_path):
    from app.context import MessageContextCompactor

    compactor = MessageContextCompactor(
        tmp_path,
        context_char_limit=500,
        target_ratio=0.8,
        max_messages=50,
    )
    messages = [HumanMessage(content="历史事实" * 100) for _ in range(5)]
    summary_calls = []

    async def summarize(history, active_request):
        summary_calls.append((len(history), active_request))
        return "已确认目标与剩余任务。"

    result = asyncio.run(compactor.prepare(messages, "当前用户请求", summarize))

    assert result.stages == [
        "tool_result_budget",
        "snip_compact",
        "micro_compact",
        "compact_history",
    ]
    assert result.summary_used is True
    assert summary_calls == [(5, "当前用户请求")]
    assert "Current user request:\n当前用户请求" in result.messages[0].content
    assert "Conversation summary:\n已确认目标与剩余任务。" in result.messages[0].content
    assert result.transcripts


def test_reactive_compact_keeps_recent_tool_pair_and_retries_once(tmp_path):
    from app.context import MessageContextCompactor

    compactor = MessageContextCompactor(tmp_path)
    messages = [HumanMessage(content=f"旧消息 {index}") for index in range(6)]
    messages.extend([
        AIMessage(content="", tool_calls=[{
            "name": "search_court_cases",
            "args": {},
            "id": "recent-pair",
        }]),
        ToolMessage(content="最新案例结果", tool_call_id="recent-pair"),
    ])

    async def summarize(history, active_request):
        assert len(history) == 6
        return "较早历史摘要"

    result = asyncio.run(compactor.reactive_compact(
        messages,
        "当前请求",
        summarize,
        keep_recent_messages=1,
    ))

    assert result.reactive is True
    assert result.summary_used is True
    assert isinstance(result.messages[-2], AIMessage)
    assert isinstance(result.messages[-1], ToolMessage)
    assert result.messages[-1].tool_call_id == "recent-pair"
    assert compactor.is_context_limit_error(RuntimeError("prompt_too_long")) is True


def test_short_term_memory_resolves_follow_up_and_expires_by_turn_limit():
    from app.memory.service import ShortTermMemoryService

    service = ShortTermMemoryService(ttl_seconds=60, max_turns=2)
    conversation_id = "test-conversation"
    service.append(conversation_id, "盗窃三千元如何处理？", "需要结合地区标准判断。", ["刑法264条"], "criminal_law")

    memory = service.load(conversation_id)
    resolved, context = service.resolve_question("如果已经退赃呢？", memory)
    assert "上一轮问题" in resolved
    assert "盗窃三千元" in context
    independent, _ = service.resolve_question("劳动仲裁时效多久？", memory)
    assert independent == "劳动仲裁时效多久？"

    service.append(conversation_id, "第二问", "第二答", [], "criminal_law")
    service.append(conversation_id, "第三问", "第三答", [], "criminal_law")
    assert len(service.load(conversation_id).turns) == 2
    assert service.clear(conversation_id) is True


def test_skill_loader_scans_frontmatter_then_loads_on_demand():
    from app.skills.loader import SkillLoader

    loader = SkillLoader()
    catalog = loader.scan()

    assert "criminal_law" in [item.name for item in catalog]
    assert "罪名认定" in loader.catalog_prompt()
    assert loader._cache == {}  # 扫描阶段没有加载任何完整 Skill

    criminal = loader.load("criminal_law")
    assert criminal.name == "criminal_law"
    assert criminal.use_kg is True
    assert "犯罪构成" in criminal.instructions
    assert list(loader._cache) == ["criminal_law"]


def test_load_skill_returns_complete_markdown_as_tool_result():
    from app.skills.tool import load_skill

    content = load_skill.invoke({"name": "contract_law"})
    assert content.startswith("---")
    assert "name: contract_law" in content
    assert "# 合同法律问答" in content


def test_pipeline_auto_skill_selection_appends_tool_result(monkeypatch):
    import asyncio
    from langchain_core.messages import AIMessage, ToolMessage
    from app.services import pipeline as pipeline_module

    class FakeSelector:
        def bind_tools(self, tools, tool_choice):
            assert tool_choice == "load_skill"
            return self

        async def ainvoke(self, messages):
            return AIMessage(
                content="",
                tool_calls=[{
                    "name": "load_skill",
                    "args": {"name": "labor_law"},
                    "id": "call-test",
                }],
            )

    monkeypatch.setattr(pipeline_module, "get_llm", lambda **kwargs: FakeSelector())
    rag_pipeline = pipeline_module.RAGPipeline(pipeline_module.PipelineConfig())
    skill, messages = asyncio.run(rag_pipeline._select_skill("公司拖欠工资怎么办？"))

    assert skill.name == "labor_law"
    assert len(messages) == 3
    assert isinstance(messages[2], ToolMessage)
    assert messages[2].tool_call_id == "call-test"
    assert "# 劳动法律问答" in messages[2].content


def test_chat_request_has_conversation_and_skill_defaults():
    from app.models.schemas import ChatRequest

    request = ChatRequest(question="测试问题")
    assert len(request.conversation_id) >= 8
    assert request.skill_name == "auto"
