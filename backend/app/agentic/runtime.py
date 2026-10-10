"""Bounded LangGraph runtime for the only online LawRAG execution path."""

from __future__ import annotations

import json
import time
from typing import Any
from uuid import uuid4

from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.graph import END, START, StateGraph

from app.agentic.prompts import (
    EVIDENCE_GRADE_SYSTEM,
    GROUNDING_CHECK_SYSTEM,
    QUERY_REWRITE_SYSTEM,
    TOOL_ROUTER_SYSTEM,
)
from app.agentic.state import AgenticRAGState
from app.agentic.tools import AgentToolRegistry, ToolExecution
from app.config import settings
from app.context import FourStageContextCompactor
from app.core.llm import get_llm
from app.db.repository import get_parent_child_repository
from app.memory import memory_service
from app.mcp import ExternalLegalMCPService
from app.models.schemas import ChatResponse, SourceDocument, StageMetrics
from app.skills import skill_loader
from app.utils.metadata import format_source_display


def _json_object(text: str) -> dict[str, Any]:
    cleaned = text.strip().removeprefix("```json").removeprefix("```").removesuffix("```").strip()
    start, end = cleaned.find("{"), cleaned.rfind("}")
    if start < 0 or end < start:
        return {}
    try:
        value = json.loads(cleaned[start : end + 1])
        return value if isinstance(value, dict) else {}
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}


def _trace(state: AgenticRAGState, node: str, **details: Any) -> list[dict[str, Any]]:
    return [*(state.get("trace") or []), {"node": node, **details}]


class AgenticRAGRunner:
    """Execute planning, tools, retrieval correction and grounding as one graph."""

    def __init__(self, pipeline: Any):
        self.pipeline = pipeline
        self.external_mcp = ExternalLegalMCPService()
        self.registry = AgentToolRegistry(
            retrieve=self._retrieve_tool,
            crime=self._crime_tool,
            source=self._source_tool,
            external=self._external_mcp_tool,
            external_tool_names=self.external_mcp.enabled_tools,
        )
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(AgenticRAGState)
        graph.add_node("prepare", self._prepare)
        graph.add_node("load_skill", self._load_skill)
        graph.add_node("decide_tools", self._decide_tools)
        graph.add_node("execute_tools", self._execute_tools)
        graph.add_node("grade_evidence", self._grade_evidence)
        graph.add_node("rewrite_query", self._rewrite_query)
        graph.add_node("compact_context", self._compact_context)
        graph.add_node("generate", self._generate)
        graph.add_node("verify_grounding", self._verify_grounding)

        graph.add_edge(START, "prepare")
        graph.add_edge("prepare", "load_skill")
        graph.add_edge("load_skill", "decide_tools")
        graph.add_conditional_edges(
            "decide_tools",
            lambda state: "tools" if state.get("tool_calls") else "grade",
            {"tools": "execute_tools", "grade": "grade_evidence"},
        )
        graph.add_edge("execute_tools", "grade_evidence")
        graph.add_conditional_edges(
            "grade_evidence",
            self._after_grade,
            {"rewrite": "rewrite_query", "compact": "compact_context"},
        )
        graph.add_edge("rewrite_query", "decide_tools")
        graph.add_edge("compact_context", "generate")
        graph.add_edge("generate", "verify_grounding")
        graph.add_conditional_edges(
            "verify_grounding",
            self._after_grounding,
            {"regenerate": "generate", "finish": END},
        )
        return graph.compile()

    async def run(self, question: str, conversation_id: str | None = None) -> ChatResponse:
        total_start = time.time()
        initial: AgenticRAGState = {
            "question": question,
            "conversation_id": conversation_id or uuid4().hex,
            "evidence": [],
            "kg_entities": [],
            "rewritten_queries": [],
            "missing_information": [],
            "retrieval_rounds": 0,
            "tool_rounds": 0,
            "generation_rounds": 0,
            "grounding_passed": False,
            "trace": [],
            "messages": [],
        }
        result: AgenticRAGState = await self.graph.ainvoke(initial)
        self.pipeline.metrics["total_ms"] = round((time.time() - total_start) * 1000, 1)
        self.pipeline.metrics["agent_tool_calls"] = sum(
            int(item.get("tool_count", 0)) for item in result.get("trace", [])
        )
        self.pipeline.metrics["agent_rounds"] = result.get("tool_rounds", 0)
        self.pipeline.metrics["retrieval_rounds"] = result.get("retrieval_rounds", 0)
        self.pipeline.metrics["grounding_passed"] = result.get("grounding_passed", False)

        documents = result.get("compacted_documents", [])
        sources = [
            SourceDocument(
                content=doc.page_content[:500],
                metadata=doc.metadata,
                score=doc.metadata.get("rerank_score"),
            )
            for doc in documents
        ]
        skill = result["skill"]
        answer = result.get("answer", "现有证据不足以形成可靠回答。")
        source_ids = [format_source_display(doc.metadata) for doc in documents]
        memory_service.append(result["conversation_id"], question, answer, source_ids, skill.name)

        config = self.pipeline.config.to_dict()
        config.update({
            "architecture": "agentic_rag",
            "max_tool_rounds": settings.AGENT_MAX_TOOL_ROUNDS,
            "max_retrieval_rounds": settings.AGENT_MAX_RETRIEVAL_ROUNDS,
        })
        return ChatResponse(
            answer=answer,
            sources=sources,
            metrics=StageMetrics(**self.pipeline.metrics),
            rewritten_queries=result.get("rewritten_queries") or None,
            kg_entities=result.get("kg_entities") or None,
            generation_strategy=self.pipeline.config.generation_strategy.value,
            pipeline_config=config,
            conversation_id=result["conversation_id"],
            resolved_question=(
                result.get("resolved_question")
                if result.get("resolved_question") != question else None
            ),
            active_skill=skill.name,
            context_compaction=result.get("compaction"),
            agent_trace=result.get("trace", []),
        )

    async def _prepare(self, state: AgenticRAGState) -> dict:
        memory = memory_service.load(state["conversation_id"])
        resolved, context = memory_service.resolve_question(state["question"], memory)
        self.pipeline.metrics["memory_turns"] = len(memory.turns)
        return {
            "resolved_question": resolved,
            "memory_context": context,
            "memory_turns": len(memory.turns),
            "pending_query": resolved,
            "trace": _trace(state, "prepare", memory_turns=len(memory.turns)),
        }

    async def _load_skill(self, state: AgenticRAGState) -> dict:
        skill, messages = await self.pipeline._select_skill(state["resolved_question"])
        if set(self.pipeline.config.collection_names) == {
            settings.LAWS_COLLECTION,
            settings.CASES_COLLECTION,
        }:
            self.pipeline.config.collection_names = list(skill.collection_names)
        return {
            "skill": skill,
            "skill_messages": messages,
            "messages": list(messages),
            "trace": _trace(state, "load_skill", skill=skill.name),
        }

    async def _decide_tools(self, state: AgenticRAGState) -> dict:
        if state.get("tool_rounds", 0) >= settings.AGENT_MAX_TOOL_ROUNDS:
            return {"tool_calls": [], "trace": _trace(state, "decide_tools", stopped="round_limit")}

        missing = "；".join(state.get("missing_information", [])) or "无"
        prompt = (
            f"用户问题：{state['resolved_question']}\n"
            f"本轮检索查询：{state.get('pending_query') or state['resolved_question']}\n"
            f"当前已有证据数：{len(state.get('evidence', []))}\n"
            f"尚缺信息：{missing}"
        )
        llm = get_llm(temperature=0).bind_tools(self.registry.tools, tool_choice="auto")
        decision = await llm.ainvoke([
            SystemMessage(content=TOOL_ROUTER_SYSTEM),
            *(state.get("messages") or []),
            HumanMessage(content=prompt),
        ])
        self.pipeline.metrics["llm_calls"] += 1
        calls = [call for call in getattr(decision, "tool_calls", []) if self.registry.available(call.get("name", ""))]
        calls = calls[:3]
        for call in calls:
            call.setdefault("id", f"agent-tool-{uuid4().hex}")
            arguments = dict(call.get("args") or {})
            if call.get("name") == "retrieve_legal_evidence":
                arguments["query"] = str(
                    arguments.get("query") or state.get("pending_query") or state["resolved_question"]
                )
                if arguments.get("collection") not in {"all", "laws", "cases"}:
                    arguments["collection"] = "all"
                try:
                    arguments["top_k"] = min(20, max(1, int(arguments.get("top_k", self.pipeline.config.top_k))))
                except (TypeError, ValueError):
                    arguments["top_k"] = self.pipeline.config.top_k
            call["args"] = arguments

        # 法律回答必须至少执行一次权威知识检索；LLM 未选择时由策略层补齐。
        has_retrieval = any(call.get("name") == "retrieve_legal_evidence" for call in calls)
        if not has_retrieval and (
            not state.get("evidence") or not state.get("evidence_sufficient", False)
        ):
            calls.append({
                "name": "retrieve_legal_evidence",
                "args": {
                    "query": state.get("pending_query") or state["resolved_question"],
                    "collection": "all",
                    "top_k": self.pipeline.config.top_k,
                },
                "id": f"policy-retrieve-{uuid4().hex}",
            })

        normalized = AIMessage(content="", tool_calls=calls)
        return {
            "tool_calls": calls,
            "messages": [*(state.get("messages") or []), normalized],
            "trace": _trace(state, "decide_tools", tool_count=len(calls), tools=[c["name"] for c in calls]),
        }

    async def _execute_tools(self, state: AgenticRAGState) -> dict:
        evidence = list(state.get("evidence", []))
        entities = list(state.get("kg_entities", []))
        messages = list(state.get("messages", []))
        retrieval_rounds = state.get("retrieval_rounds", 0)
        executed: list[str] = []

        for call in state.get("tool_calls", []):
            name = str(call.get("name", ""))
            try:
                execution = await self.registry.execute(name, dict(call.get("args") or {}))
            except Exception as exc:
                execution = ToolExecution(content=f"工具执行失败：{type(exc).__name__}")
            if name == "retrieve_legal_evidence":
                retrieval_rounds += 1
            evidence = self._merge_documents(evidence, execution.documents)
            entities = list(dict.fromkeys([*entities, *execution.entities]))
            messages.append(ToolMessage(
                content=execution.content,
                tool_call_id=str(call.get("id") or f"tool-{uuid4().hex}"),
                name=name,
            ))
            executed.append(name)

        return {
            "evidence": evidence,
            "kg_entities": entities,
            "messages": messages,
            "tool_calls": [],
            "retrieval_rounds": retrieval_rounds,
            "tool_rounds": state.get("tool_rounds", 0) + 1,
            "trace": _trace(state, "execute_tools", tools=executed, evidence_count=len(evidence)),
        }

    async def _grade_evidence(self, state: AgenticRAGState) -> dict:
        started = time.time()
        evidence = state.get("evidence", [])
        if not evidence:
            return {
                "evidence_sufficient": False,
                "missing_information": ["未检索到可用于回答的法律证据"],
                "pending_query": state["resolved_question"],
                "trace": _trace(state, "grade_evidence", sufficient=False, evidence_count=0),
            }

        preview = FourStageContextCompactor(2200).compact(
            evidence,
            question=state["resolved_question"],
        ).context
        response = await get_llm(temperature=0).ainvoke([
            SystemMessage(content=EVIDENCE_GRADE_SYSTEM),
            HumanMessage(content=f"问题：{state['resolved_question']}\n\n证据：\n{preview}"),
        ])
        self.pipeline.metrics["llm_calls"] += 1
        data = _json_object(str(response.content))
        sufficient = data.get("sufficient") is True
        missing = data.get("missing_information") or []
        if isinstance(missing, str):
            missing = [missing]
        follow_up = str(data.get("follow_up_query") or "").strip()
        self.pipeline.metrics["evidence_grade_ms"] = round((time.time() - started) * 1000, 1)
        return {
            "evidence_sufficient": sufficient,
            "missing_information": [str(item) for item in missing],
            "pending_query": follow_up or state.get("pending_query") or state["resolved_question"],
            "trace": _trace(state, "grade_evidence", sufficient=sufficient, evidence_count=len(evidence)),
        }

    def _after_grade(self, state: AgenticRAGState) -> str:
        if state.get("evidence_sufficient"):
            return "compact"
        if state.get("retrieval_rounds", 0) >= settings.AGENT_MAX_RETRIEVAL_ROUNDS:
            return "compact"
        if state.get("tool_rounds", 0) >= settings.AGENT_MAX_TOOL_ROUNDS:
            return "compact"
        return "rewrite"

    async def _rewrite_query(self, state: AgenticRAGState) -> dict:
        started = time.time()
        suggested = (state.get("pending_query") or "").strip()
        if not suggested or suggested == state["resolved_question"]:
            response = await get_llm(temperature=0).ainvoke([
                SystemMessage(content=QUERY_REWRITE_SYSTEM),
                HumanMessage(content=(
                    f"原问题：{state['resolved_question']}\n"
                    f"缺失信息：{'；'.join(state.get('missing_information', []))}"
                )),
            ])
            self.pipeline.metrics["llm_calls"] += 1
            suggested = str(response.content).strip()
        rewritten = [*state.get("rewritten_queries", []), suggested]
        self.pipeline.metrics["query_rewrite_ms"] = round((time.time() - started) * 1000, 1)
        return {
            "pending_query": suggested,
            "rewritten_queries": rewritten,
            "trace": _trace(state, "rewrite_query", query=suggested),
        }

    async def _compact_context(self, state: AgenticRAGState) -> dict:
        started = time.time()
        result = FourStageContextCompactor(settings.CONTEXT_MAX_TOKENS).compact(
            state.get("evidence", []),
            question=state["resolved_question"],
            memory_context=state.get("memory_context", ""),
            skill_instructions=state["skill"].instructions,
            priority=state["skill"].context_priority,
        )
        self.pipeline.metrics["context_compact_ms"] = round((time.time() - started) * 1000, 1)
        self.pipeline.metrics["context_tokens_before"] = result.tokens_before
        self.pipeline.metrics["context_tokens_after"] = result.tokens_after
        return {
            "compacted_context": result.context,
            "compacted_documents": result.documents,
            "compaction": result.to_dict(),
            "trace": _trace(state, "compact_context", document_count=len(result.documents)),
        }

    async def _generate(self, state: AgenticRAGState) -> dict:
        question = state["resolved_question"]
        if state.get("grounding_feedback"):
            question += f"\n\n上一版回答校验未通过，请按以下要求修正：{state['grounding_feedback']}"
        answer, corrected = await self.pipeline._generate(
            question,
            state.get("compacted_context", "未找到相关参考资料。"),
            state.get("memory_context", ""),
            state.get("skill_messages", []),
            skill_loader.catalog_prompt(),
        )
        self.pipeline.metrics["llm_calls"] += 1
        self.pipeline.metrics["was_corrected"] = corrected or state.get("generation_rounds", 0) > 0
        return {
            "answer": answer,
            "generation_rounds": state.get("generation_rounds", 0) + 1,
            "trace": _trace(state, "generate", round=state.get("generation_rounds", 0) + 1),
        }

    async def _verify_grounding(self, state: AgenticRAGState) -> dict:
        started = time.time()
        response = await get_llm(temperature=0).ainvoke([
            SystemMessage(content=GROUNDING_CHECK_SYSTEM),
            HumanMessage(content=(
                f"问题：{state['resolved_question']}\n\n"
                f"证据：\n{state.get('compacted_context', '')}\n\n"
                f"回答：\n{state.get('answer', '')}"
            )),
        ])
        self.pipeline.metrics["llm_calls"] += 1
        data = _json_object(str(response.content))
        grounded = data.get("grounded") is True
        citations = data.get("citation_complete") is True
        passed = grounded and citations
        feedback = str(data.get("feedback") or "补充明确来源，并删除证据无法支持的结论。")
        self.pipeline.metrics["grounding_check_ms"] = round((time.time() - started) * 1000, 1)
        return {
            "grounding_passed": passed,
            "grounding_feedback": "" if passed else feedback,
            "trace": _trace(state, "verify_grounding", passed=passed),
        }

    def _after_grounding(self, state: AgenticRAGState) -> str:
        if state.get("grounding_passed"):
            return "finish"
        if state.get("generation_rounds", 0) >= settings.AGENT_MAX_GENERATION_ROUNDS:
            return "finish"
        return "regenerate"

    async def _retrieve_tool(self, query: str, collection: str, top_k: int) -> ToolExecution:
        original_collections = list(self.pipeline.config.collection_names)
        original_top_k = self.pipeline.config.top_k
        mapping = {
            "laws": [settings.LAWS_COLLECTION],
            "cases": [settings.CASES_COLLECTION],
            "all": [settings.LAWS_COLLECTION, settings.CASES_COLLECTION],
        }
        requested = mapping.get(collection, mapping["all"])
        allowed = set(original_collections)
        self.pipeline.config.collection_names = [name for name in requested if name in allowed] or original_collections
        self.pipeline.config.top_k = min(20, max(1, top_k))
        try:
            children = await self.pipeline._retrieve([query], None)
            self.pipeline.metrics["retrieved_child_count"] += len(children)
            reranked = await self.pipeline._rerank(
                query,
                children,
                top_k=max(self.pipeline.config.top_k, settings.CHILD_RERANK_TOP_K),
            )
            self.pipeline.metrics["reranked_child_count"] += len(reranked)
            parents = self.pipeline._aggregate_and_hydrate_parents(reranked)
            self.pipeline.metrics["parent_candidate_count"] += len(parents)
            compacted = FourStageContextCompactor(settings.AGENT_TOOL_RESULT_MAX_TOKENS).compact(
                parents,
                question=query,
            )
            return ToolExecution(content=compacted.context, documents=parents)
        finally:
            self.pipeline.config.collection_names = original_collections
            self.pipeline.config.top_k = original_top_k

    async def _crime_tool(self, crime_name: str) -> ToolExecution:
        entities, documents = await self.pipeline._kg_lookup(crime_name, allow_llm_fallback=False)
        compacted = FourStageContextCompactor(settings.AGENT_TOOL_RESULT_MAX_TOKENS).compact(
            documents,
            question=crime_name,
        )
        return ToolExecution(content=compacted.context, documents=documents, entities=entities)

    async def _source_tool(self, chunk_id: str) -> ToolExecution:
        repository = self.pipeline.repository or get_parent_child_repository()
        self.pipeline.repository = repository
        source = repository.get_chunk_source(chunk_id)
        if source is None:
            return ToolExecution(content="未找到对应来源。")
        text = str(source.get("parent_text") or source.get("chunk_text") or "")
        document = Document(page_content=text, metadata={
            "doc_type": "source",
            "parent_chunk_id": source.get("parent_chunk_id", ""),
            "child_chunk_id": source.get("child_chunk_id", ""),
            "doc_id": source.get("doc_id", ""),
            "source_file": source.get("source_file", ""),
        })
        compacted = FourStageContextCompactor(settings.AGENT_TOOL_RESULT_MAX_TOKENS).compact([document])
        return ToolExecution(content=compacted.context, documents=[document])

    async def _external_mcp_tool(self, name: str, arguments: dict) -> ToolExecution:
        result = await self.external_mcp.execute(name, arguments)
        document = Document(page_content=result.content, metadata=result.metadata)
        compacted = FourStageContextCompactor(settings.AGENT_TOOL_RESULT_MAX_TOKENS).compact(
            [document],
            question=str(arguments.get("query") or arguments.get("keyword") or ""),
        )
        return ToolExecution(content=compacted.context, documents=[document])

    @staticmethod
    def _merge_documents(current: list[Document], incoming: list[Document]) -> list[Document]:
        merged = list(current)
        seen = {
            str(doc.metadata.get("parent_chunk_id") or doc.metadata.get("child_chunk_id") or doc.page_content[:160])
            for doc in current
        }
        for doc in incoming:
            key = str(doc.metadata.get("parent_chunk_id") or doc.metadata.get("child_chunk_id") or doc.page_content[:160])
            if key not in seen:
                seen.add(key)
                merged.append(doc)
        return merged
