"""RAG 可插拔管线 — 编排查询变换、检索、重排序、生成等阶段"""

import time
from uuid import uuid4
from dataclasses import dataclass, field
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
from app.models.schemas import ChatResponse
from app.config import settings
from app.context import FourStageContextCompactor
from app.skills import LegalSkill, load_skill, skill_loader
from app.db.repository import ParentChildRepository, get_parent_child_repository


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

    def __init__(
        self,
        config: PipelineConfig,
        repository: ParentChildRepository | None = None,
    ):
        self.config = config
        self.repository = repository
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
            "retrieved_child_count": 0,
            "reranked_child_count": 0,
            "parent_candidate_count": 0,
            "evidence_grade_ms": 0,
            "grounding_check_ms": 0,
            "agent_tool_calls": 0,
            "agent_rounds": 0,
            "retrieval_rounds": 0,
            "grounding_passed": False,
        }

    async def execute(self, question: str, conversation_id: str | None = None) -> ChatResponse:
        """Execute the single Agentic RAG graph; no linear workflow route exists."""
        from app.agentic import AgenticRAGRunner

        return await AgenticRAGRunner(self).run(question, conversation_id)

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
        candidate_k = max(self.config.top_k, settings.CHILD_RERANK_TOP_K)
        if self.config.rerank_strategy != RerankStrategy.NONE:
            candidate_k = max(
                candidate_k,
                settings.RERANKER_CANDIDATE_K,
                settings.CHILD_RERANK_TOP_K * 2,
            )
        repository = self.repository or get_parent_child_repository()
        self.repository = repository
        retriever = get_hybrid_retriever(
            self.config.collection_names,
            k=candidate_k,
            repository=repository,
        )

        all_docs: list[Document] = []
        seen_child_ids: set[str] = set()

        if hyde_doc:
            # 使用 split query: BM25 用原始查询, 向量用假设文档
            docs = retriever.search_with_split_queries(
                bm25_query=search_queries[0],
                vector_query=hyde_doc,
            )
            for doc in docs:
                key = str(doc.metadata.get("child_chunk_id", ""))
                if key and key not in seen_child_ids:
                    seen_child_ids.add(key)
                    all_docs.append(doc)

        # 常规多查询检索
        for q in search_queries:
            docs = retriever.invoke(q)
            for doc in docs:
                key = str(doc.metadata.get("child_chunk_id", ""))
                if key and key not in seen_child_ids:
                    seen_child_ids.add(key)
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

    async def _rerank(
        self,
        question: str,
        all_docs: list[Document],
        top_k: int | None = None,
    ) -> list[Document]:
        strategy = self.config.rerank_strategy
        top_k = top_k or self.config.top_k

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

    def _aggregate_and_hydrate_parents(self, children: list[Document]) -> list[Document]:
        """Rank parents by their best child score plus a capped coverage bonus."""
        if not children:
            return []
        groups: dict[str, list[Document]] = {}
        legacy: list[Document] = []
        for child in children:
            parent_id = str(child.metadata.get("parent_chunk_id", ""))
            if not parent_id:
                legacy.append(child)
                continue
            groups.setdefault(parent_id, []).append(child)

        parent_scores: dict[str, float] = {}
        for parent_id, hits in groups.items():
            scores = [
                float(hit.metadata.get("rerank_score", hit.metadata.get("rrf_score", 0.0)) or 0.0)
                for hit in hits
            ]
            coverage = min(max(len(hits) - 1, 0), 3)
            parent_scores[parent_id] = max(scores, default=0.0) + settings.PARENT_HIT_BONUS * coverage

        parent_limit = max(self.config.top_k, settings.PARENT_TOP_K)
        parent_ids = [
            parent_id for parent_id, _ in
            sorted(parent_scores.items(), key=lambda item: item[1], reverse=True)[:parent_limit]
        ]
        if not parent_ids:
            return legacy[:parent_limit]

        repository = self.repository or get_parent_child_repository()
        self.repository = repository
        parents = repository.fetch_parents(parent_ids)
        for parent in parents:
            parent_id = str(parent.metadata.get("parent_chunk_id", ""))
            hits = groups.get(parent_id, [])
            parent.metadata["parent_score"] = round(parent_scores.get(parent_id, 0.0), 6)
            parent.metadata["rerank_score"] = round(parent_scores.get(parent_id, 0.0), 6)
            parent.metadata["matched_child_ids"] = [
                hit.metadata.get("child_chunk_id") for hit in hits
            ]
            parent.metadata["matched_child_count"] = len(hits)
            parent.metadata["matched_child_preview"] = [
                hit.page_content[:180] for hit in hits[:3]
            ]
        return parents + legacy[: max(0, parent_limit - len(parents))]

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
