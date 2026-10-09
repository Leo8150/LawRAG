"""RAG 可插拔管线 — 编排查询变换、检索、重排序、生成等阶段"""

import time
from uuid import uuid4
from dataclasses import dataclass, field, asdict
from enum import Enum

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage, ToolMessage

from app.core.llm import get_llm
from app.core.retriever import get_hybrid_retriever
from app.services.reranker import simple_rerank
from app.services.query_rewriter import multi_query_rewrite
from app.services.prompts import (
    LEGAL_QA_PROMPT,
    LEGAL_STRUCTURED_PROMPT,
)
from app.utils.metadata import format_source_display
from app.models.schemas import ChatResponse, SourceDocument, StageMetrics
from app.config import settings
from app.context import FourStageContextCompactor
from app.memory import memory_service
from app.skills import LegalSkill, load_skill, skill_loader


# ===================== 策略枚举 =====================

class QueryTransformStrategy(str, Enum):
    NONE = "none"
    MULTI_QUERY = "multi_query"
    HYDE = "hyde"
    DECOMPOSE = "decompose"
    MULTI_QUERY_HYDE = "multi_query_hyde"


class RerankStrategy(str, Enum):
    NONE = "none"
    SIMPLE = "simple"
    CLOUD = "cloud"


class GenerationStrategy(str, Enum):
    STANDARD = "standard"
    SELF_REFLECT = "self_reflect"
    STRUCTURED = "structured_legal"


# ===================== 管线配置 =====================

@dataclass
class PipelineConfig:
    query_transform: QueryTransformStrategy = QueryTransformStrategy.NONE
    rerank_strategy: RerankStrategy = RerankStrategy.SIMPLE
    generation_strategy: GenerationStrategy = GenerationStrategy.STANDARD
    use_kg: bool = False
    top_k: int = 5
    collection_names: list[str] = field(default_factory=lambda: ["laws", "cases"])
    skill_name: str = "auto"

    def to_dict(self) -> dict:
        return {
            "query_transform": self.query_transform.value,
            "rerank_strategy": self.rerank_strategy.value,
            "generation_strategy": self.generation_strategy.value,
            "use_kg": self.use_kg,
            "top_k": self.top_k,
            "collection_names": self.collection_names,
            "skill_name": self.skill_name,
        }


# ===================== RAG 管线 =====================

class RAGPipeline:
    """可插拔 RAG 管线，按配置调度各阶段策略"""

    def __init__(self, config: PipelineConfig):
        self.config = config
        self.metrics: dict = {
            "query_rewrite_ms": None,
            "retrieval_ms": 0,
            "rerank_ms": None,
            "generation_ms": 0,
            "total_ms": 0,
            "kg_lookup_ms": None,
            "self_reflect_ms": None,
            "was_corrected": False,
            "llm_calls": 0,
            "llm_calls_saved": 0,
            "rerank_api_calls": 0,
            "reranker_model": None,
            "rerank_candidates": 0,
            "rerank_tokens": 0,
            "rerank_fallback": False,
            "context_compact_ms": 0,
            "context_tokens_before": 0,
            "context_tokens_after": 0,
            "memory_turns": 0,
        }

    async def execute(self, question: str, conversation_id: str | None = None) -> ChatResponse:
        total_start = time.time()

        # Stage 0: 恢复短期记忆；目录进入 system prompt，完整 Skill 通过 tool_result 注入。
        conversation_id = conversation_id or uuid4().hex
        memory = memory_service.load(conversation_id)
        resolved_question, memory_context = memory_service.resolve_question(question, memory)
        self.metrics["memory_turns"] = len(memory.turns)
        skill, skill_messages = await self._select_skill(resolved_question)

        # Skill 可以收窄数据源；调用方显式选择单库时保持调用方选择。
        if set(self.config.collection_names) == {settings.LAWS_COLLECTION, settings.CASES_COLLECTION}:
            self.config.collection_names = list(skill.collection_names)

        # Stage 1: 查询变换
        search_queries, hyde_doc, rewritten_queries = await self._query_transform(resolved_question)
        self.metrics["llm_calls"] += self._count_transform_calls()

        # Stage 2: 检索（纯检索，零 LLM 调用）
        all_docs = await self._retrieve(search_queries, hyde_doc)

        # Stage 2.5: KG 查找（并入检索结果）
        kg_entities: list[str] = []
        effective_use_kg = self.config.use_kg or skill.use_kg
        if effective_use_kg:
            kg_entities, kg_docs = await self._kg_lookup(
                resolved_question,
                allow_llm_fallback=self.config.use_kg,
            )
            # KG 精确匹配命中时节省 1 次 LLM 调用
            if kg_docs:
                self.metrics["llm_calls_saved"] += 1
                existing_keys = {hash(d.page_content[:200]) for d in all_docs}
                for kd in kg_docs:
                    key = hash(kd.page_content[:200])
                    if key not in existing_keys:
                        all_docs.insert(0, kd)
                        existing_keys.add(key)

        # Stage 3: 重排序
        reranked_docs = await self._rerank(resolved_question, all_docs)

        # Stage 4: 四阶段上下文压缩（Budget -> Snip -> Micro -> Summary）
        compact_start = time.time()
        compact_result = FourStageContextCompactor(settings.CONTEXT_MAX_TOKENS).compact(
            reranked_docs,
            question=resolved_question,
            memory_context=memory_context,
            skill_instructions=skill.instructions,
            priority=skill.context_priority,
        )
        self.metrics["context_compact_ms"] = round((time.time() - compact_start) * 1000, 1)
        self.metrics["context_tokens_before"] = compact_result.tokens_before
        self.metrics["context_tokens_after"] = compact_result.tokens_after

        # Stage 5: 生成
        answer, was_corrected = await self._generate(
            resolved_question,
            compact_result.context,
            memory_context,
            skill_messages,
            skill_loader.catalog_prompt(),
        )
        self.metrics["was_corrected"] = was_corrected
        self.metrics["llm_calls"] += 1  # 生成至少 1 次
        if was_corrected:
            self.metrics["llm_calls"] += 2  # 反思验证 + 修正

        # 总耗时
        self.metrics["total_ms"] = round((time.time() - total_start) * 1000, 1)

        # 构建来源
        sources = []
        for doc in compact_result.documents:
            sources.append(SourceDocument(
                content=doc.page_content[:500],
                metadata=doc.metadata,
                score=doc.metadata.get("rerank_score"),
            ))

        # Stage 6: 只保存有限轮次和来源 ID，不复制整篇证据。
        source_ids = [format_source_display(doc.metadata) for doc in compact_result.documents]
        memory_service.append(conversation_id, question, answer, source_ids, skill.name)

        config_dict = self.config.to_dict()
        config_dict["use_kg"] = effective_use_kg
        return ChatResponse(
            answer=answer,
            sources=sources,
            metrics=StageMetrics(**self.metrics),
            rewritten_queries=rewritten_queries,
            kg_entities=kg_entities or None,
            generation_strategy=self.config.generation_strategy.value,
            pipeline_config=config_dict,
            conversation_id=conversation_id,
            resolved_question=resolved_question if resolved_question != question else None,
            active_skill=skill.name,
            context_compaction=compact_result.to_dict(),
        )

    async def _select_skill(self, question: str) -> tuple[LegalSkill, list[BaseMessage]]:
        """Ask the model to call load_skill, then append the result as a ToolMessage."""
        requested = self.config.skill_name
        if requested == "auto":
            selector = get_llm(temperature=0).bind_tools([load_skill], tool_choice="load_skill")
            selection = await selector.ainvoke([
                SystemMessage(content=(
                    "你是 LawRAG 的技能路由器。根据用户问题，从目录中选择且只选择一个技能，"
                    "并调用 load_skill(name)。不要直接回答法律问题。\n\n"
                    f"{skill_loader.catalog_prompt()}"
                )),
                HumanMessage(content=question),
            ])
            self.metrics["llm_calls"] += 1
            call = next(
                (item for item in selection.tool_calls if item.get("name") == "load_skill"),
                None,
            )
            requested = str((call or {}).get("args", {}).get("name", "general_legal"))
            if requested not in skill_loader.available():
                requested = "general_legal"
            call_id = str((call or {}).get("id") or f"load-skill-{uuid4().hex}")
        else:
            if requested not in skill_loader.available():
                requested = "general_legal"
            call_id = f"load-skill-{uuid4().hex}"

        # Normalize to exactly one tool call so every call has one matching result.
        selection = AIMessage(
            content="",
            tool_calls=[{"name": "load_skill", "args": {"name": requested}, "id": call_id}],
        )

        # This is the only point that reads the complete SKILL.md body.
        skill_content = load_skill.invoke({"name": requested})
        skill = skill_loader.load(requested)
        tool_result = ToolMessage(
            content=skill_content,
            tool_call_id=call_id,
            name="load_skill",
        )
        # Preserve the original user message, assistant tool call and tool result
        # as one valid message history for the next LLM invocation.
        return skill, [HumanMessage(content=question), selection, tool_result]

    # ---- Stage 1: 查询变换 ----

    def _count_transform_calls(self) -> int:
        """统计查询变换阶段的 LLM 调用次数"""
        s = self.config.query_transform
        if s == QueryTransformStrategy.NONE:
            return 0
        if s == QueryTransformStrategy.MULTI_QUERY_HYDE:
            return 2  # 并行但各调一次
        return 1  # MULTI_QUERY / HYDE / DECOMPOSE 各一次

    async def _query_transform(self, question: str) -> tuple[list[str], str | None, list[str] | None]:
        """返回 (search_queries, hyde_doc_or_none, rewritten_queries_or_none)"""
        strategy = self.config.query_transform
        if strategy == QueryTransformStrategy.NONE:
            return [question], None, None

        t0 = time.time()
        search_queries = [question]
        hyde_doc = None
        rewritten_queries = None

        if strategy == QueryTransformStrategy.MULTI_QUERY:
            search_queries = await multi_query_rewrite(question)
            rewritten_queries = search_queries

        elif strategy == QueryTransformStrategy.HYDE:
            from app.services.hyde import hyde_transform
            original, hypo = await hyde_transform(question)
            search_queries = [original]
            hyde_doc = hypo

        elif strategy == QueryTransformStrategy.DECOMPOSE:
            from app.services.query_rewriter import decompose_query
            sub_qs = await decompose_query(question)
            search_queries = sub_qs
            rewritten_queries = sub_qs

        elif strategy == QueryTransformStrategy.MULTI_QUERY_HYDE:
            import asyncio
            from app.services.hyde import hyde_transform
            # 并行执行多查询重写和 HyDE 生成
            mq_task = asyncio.create_task(multi_query_rewrite(question))
            hyde_task = asyncio.create_task(hyde_transform(question))
            mq, (_, hypo) = await asyncio.gather(mq_task, hyde_task)
            search_queries = mq
            hyde_doc = hypo
            rewritten_queries = mq

        self.metrics["query_rewrite_ms"] = round((time.time() - t0) * 1000, 1)
        return search_queries, hyde_doc, rewritten_queries

    # ---- Stage 2: 检索 ----

    async def _retrieve(self, search_queries: list[str], hyde_doc: str | None) -> list[Document]:
        t0 = time.time()
        # 重排序需要更大的候选池；不重排时只召回最终所需数量。
        candidate_k = self.config.top_k
        if self.config.rerank_strategy != RerankStrategy.NONE:
            candidate_k = max(candidate_k, settings.RERANKER_CANDIDATE_K)
        retriever = get_hybrid_retriever(self.config.collection_names, k=candidate_k)

        all_docs: list[Document] = []
        seen_contents: set[int] = set()

        if hyde_doc:
            # 使用 split query: BM25 用原始查询, 向量用假设文档
            docs = retriever.search_with_split_queries(
                bm25_query=search_queries[0],
                vector_query=hyde_doc,
            )
            for doc in docs:
                key = hash(doc.page_content[:200])
                if key not in seen_contents:
                    seen_contents.add(key)
                    all_docs.append(doc)

        # 常规多查询检索
        for q in search_queries:
            docs = retriever.invoke(q)
            for doc in docs:
                key = hash(doc.page_content[:200])
                if key not in seen_contents:
                    seen_contents.add(key)
                    all_docs.append(doc)

        self.metrics["retrieval_ms"] = round((time.time() - t0) * 1000, 1)
        return all_docs

    # ---- Stage 2.5: KG 查找 ----

    async def _kg_lookup(
        self,
        question: str,
        allow_llm_fallback: bool = True,
    ) -> tuple[list[str], list[Document]]:
        t0 = time.time()
        try:
            from app.services.kg_service import extract_crime_entities, kg_lookup
            entities = await extract_crime_entities(question, allow_llm_fallback=allow_llm_fallback)
            docs = kg_lookup(entities) if entities else []
            self.metrics["kg_lookup_ms"] = round((time.time() - t0) * 1000, 1)
            return entities, docs
        except Exception:
            self.metrics["kg_lookup_ms"] = round((time.time() - t0) * 1000, 1)
            return [], []

    # ---- Stage 3: 重排序 ----

    async def _rerank(self, question: str, all_docs: list[Document]) -> list[Document]:
        strategy = self.config.rerank_strategy
        top_k = self.config.top_k

        if strategy == RerankStrategy.NONE or not all_docs:
            return all_docs[:top_k]

        t0 = time.time()
        self.metrics["rerank_candidates"] = len(all_docs)
        try:
            if strategy == RerankStrategy.SIMPLE:
                scored = simple_rerank(question, all_docs, top_k=top_k)
                self.metrics["reranker_model"] = "jaccard+metadata"
            elif strategy == RerankStrategy.CLOUD:
                from app.services.cloud_reranker import cloud_rerank

                self.metrics["rerank_api_calls"] = 1
                self.metrics["rerank_candidates"] = min(
                    len(all_docs), settings.RERANKER_CANDIDATE_K
                )
                result = await cloud_rerank(question, all_docs, top_k=top_k)
                scored = result.ranked_documents
                self.metrics["reranker_model"] = result.model
                self.metrics["rerank_tokens"] = result.total_tokens
            else:
                scored = [(doc, 0.0) for doc in all_docs[:top_k]]

            reranked = []
            for rank, (doc, score) in enumerate(scored, 1):
                doc.metadata["rerank_score"] = round(float(score), 6)
                doc.metadata["rerank_rank"] = rank
                doc.metadata["rerank_strategy"] = strategy.value
                reranked.append(doc)
            self.metrics["rerank_ms"] = round((time.time() - t0) * 1000, 1)
            return reranked
        except Exception:
            # 云端重排不可用时回退到零云调用的轻量重排。
            self.metrics["rerank_fallback"] = True
            self.metrics["reranker_model"] = "jaccard+metadata (fallback)"
            scored = simple_rerank(question, all_docs, top_k=top_k)
            reranked = []
            for rank, (doc, score) in enumerate(scored, 1):
                doc.metadata["rerank_score"] = round(float(score), 6)
                doc.metadata["rerank_rank"] = rank
                doc.metadata["rerank_strategy"] = "simple_fallback"
                reranked.append(doc)
            self.metrics["rerank_ms"] = round((time.time() - t0) * 1000, 1)
            return reranked

    # ---- Stage 4: 生成 ----

    async def _generate(
        self,
        question: str,
        context: str,
        memory_context: str,
        skill_messages: list[BaseMessage],
        skill_catalog: str,
    ) -> tuple[str, bool]:
        t0 = time.time()
        strategy = self.config.generation_strategy

        # 选择 prompt
        if strategy == GenerationStrategy.STRUCTURED:
            prompt = LEGAL_STRUCTURED_PROMPT
        else:
            prompt = LEGAL_QA_PROMPT

        llm = get_llm()
        chain = prompt | llm
        response = await chain.ainvoke({
            "context": context,
            "question": question,
            "memory_context": memory_context or "无历史会话，本轮作为独立问题处理。",
            "skill_catalog": skill_catalog,
            "skill_messages": skill_messages,
        })
        answer = response.content
        was_corrected = False

        # 自我反思
        if strategy == GenerationStrategy.SELF_REFLECT:
            gen_ms = round((time.time() - t0) * 1000, 1)
            self.metrics["generation_ms"] = gen_ms

            t_reflect = time.time()
            from app.services.self_reflect import self_reflect_and_correct
            answer, was_corrected = await self_reflect_and_correct(
                question=question,
                initial_answer=answer,
                context=context,
                max_iterations=settings.SELF_REFLECT_MAX_ITER,
            )
            self.metrics["self_reflect_ms"] = round((time.time() - t_reflect) * 1000, 1)
            return answer, was_corrected

        self.metrics["generation_ms"] = round((time.time() - t0) * 1000, 1)
        return answer, was_corrected


# ===================== 公共辅助函数 =====================

def resolve_collections(collection: str) -> list[str]:
    """解析要检索的 collection"""
    if collection == "laws":
        return [settings.LAWS_COLLECTION]
    elif collection == "cases":
        return [settings.CASES_COLLECTION]
    else:
        return [settings.LAWS_COLLECTION, settings.CASES_COLLECTION]


def build_context(docs: list[Document], max_length: int = 4000) -> str:
    """兼容旧调用方式，内部统一使用四阶段上下文压缩器。"""
    return FourStageContextCompactor(max_context_tokens=max_length).compact(docs).context
