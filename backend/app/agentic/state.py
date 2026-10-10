"""State contract shared by all Agentic RAG graph nodes."""

from __future__ import annotations

from typing import Any, TypedDict

from langchain_core.documents import Document
from langchain_core.messages import BaseMessage

from app.skills.loader import LegalSkill


class AgenticRAGState(TypedDict, total=False):
    question: str
    conversation_id: str
    thread_id: str
    user_id: str
    memory_mode: str
    confirmed_memories: list[dict[str, Any]]
    resolved_question: str
    memory_context: str
    memory_turns: int
    conversation_summary: str
    recent_turns: list[dict[str, str]]
    case_facts: list[str]
    recalled_memory_ids: list[str]
    skill: LegalSkill
    skill_messages: list[BaseMessage]
    messages: list[BaseMessage]
    pending_query: str
    tool_calls: list[dict[str, Any]]
    evidence: list[Document]
    kg_entities: list[str]
    rewritten_queries: list[str]
    missing_information: list[str]
    evidence_sufficient: bool
    retrieval_rounds: int
    tool_rounds: int
    generation_rounds: int
    compacted_context: str
    compacted_documents: list[Document]
    compaction: dict[str, Any]
    answer: str
    grounding_feedback: str
    grounding_passed: bool
    trace: list[dict[str, Any]]
