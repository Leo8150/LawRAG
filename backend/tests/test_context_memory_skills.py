"""四阶段上下文压缩、短期记忆与 Skill Loading 的离线测试。"""

from langchain_core.documents import Document


def test_four_stage_compactor_respects_budget_and_deduplicates():
    from app.context.compactor import FourStageContextCompactor

    law = Document(
        page_content="中华人民共和国刑法第二百六十四条。" * 100,
        metadata={"doc_type": "law", "law_name": "中华人民共和国刑法", "article_number": "264"},
    )
    duplicate = Document(page_content=law.page_content, metadata=dict(law.metadata))
    case = Document(
        page_content="基本案情。法院认为入户盗窃应当依法认定。裁判结果为有罪。" * 80,
        metadata={"doc_type": "case", "source_file": "case.json"},
    )

    result = FourStageContextCompactor(max_context_tokens=700).compact([case, law, duplicate])

    assert result.stages == ["budget", "snip", "micro", "summary"]
    assert result.tokens_after <= result.token_budget + 30  # 来源标签估算允许小幅误差
    assert sum("第二百六十四条" in d.page_content for d in result.documents) == 1
    assert "来源1" in result.context


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
