"""Semantic and episodic long-term memory backed by MySQL and ChromaDB."""

from __future__ import annotations

import asyncio
import hashlib
import re
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from app.config import settings
from app.core.vectorstore import get_vectorstore
from app.db.models import MemoryEntry
from app.memory.repository import LongTermMemoryRepository, get_long_term_memory_repository


@dataclass
class RecalledMemory:
    memory_id: str
    memory_type: str
    category: str
    content: str
    is_user_stated: bool
    source_kind: str


class LongTermMemoryService:
    def __init__(
        self,
        repository: LongTermMemoryRepository | None = None,
        vectorstore: Any | None = None,
    ):
        self.repository = repository
        self._vectorstore = vectorstore

    @property
    def repo(self) -> LongTermMemoryRepository:
        if self.repository is None:
            self.repository = get_long_term_memory_repository()
        return self.repository

    @property
    def vectorstore(self):
        if self._vectorstore is None:
            self._vectorstore = get_vectorstore(settings.MEMORY_COLLECTION)
        return self._vectorstore

    async def recall(self, user_id: str, query: str, top_k: int | None = None) -> list[RecalledMemory]:
        if not settings.LONG_TERM_MEMORY_ENABLED or not user_id.strip():
            return []
        limit = top_k or settings.MEMORY_RECALL_TOP_K
        try:
            documents = await asyncio.to_thread(
                self.vectorstore.similarity_search,
                query,
                limit,
                {"user_id": user_id},
            )
            memory_ids = [
                str(doc.metadata.get("memory_id", ""))
                for doc in documents
                if doc.metadata.get("memory_id")
            ]
            rows = await asyncio.to_thread(self.repo.fetch, user_id, memory_ids)
            await asyncio.to_thread(self.repo.record_access, [row.memory_id for row in rows])
            return [RecalledMemory(
                memory_id=row.memory_id,
                memory_type=row.memory_type,
                category=row.category,
                content=row.content,
                is_user_stated=row.source_kind in {"user_statement", "user_preference", "user_confirmed"},
                source_kind=row.source_kind,
            ) for row in rows]
        except Exception:
            # Long-term memory is enrichment; retrieval failure must not block legal QA.
            return []

    async def remember_user_statements(
        self,
        user_id: str,
        items: list[dict[str, Any]],
        thread_id: str,
    ) -> list[str]:
        if not settings.LONG_TERM_MEMORY_ENABLED or not user_id.strip():
            return []
        entries = self._semantic_entries(user_id, items, thread_id)
        return [entry.memory_id for entry in await self._store_many(entries)]

    async def remember_turn(
        self,
        *,
        user_id: str,
        thread_id: str,
        user_items: list[dict[str, Any]],
        question: str,
        answer: str,
        trace: list[dict[str, Any]],
        source_ids: list[str],
    ) -> list[str]:
        """Persist all memories from one turn with one embedding batch.

        Semantic entries only come from the constrained user-message extractor.
        The generated task episode is intentionally marked as agent-generated,
        so it can help route future work without becoming a user-stated fact.
        """
        if not settings.LONG_TERM_MEMORY_ENABLED or not user_id.strip():
            return []
        entries = self._semantic_entries(user_id, user_items, thread_id)
        episode = self._episode_entry(
            user_id=user_id,
            thread_id=thread_id,
            question=question,
            answer=answer,
            trace=trace,
            source_ids=source_ids,
        )
        if episode is not None:
            entries.append(episode)
        return [entry.memory_id for entry in await self._store_many(entries)]

    async def remember_episode(
        self,
        user_id: str,
        thread_id: str,
        question: str,
        answer: str,
        trace: list[dict[str, Any]],
        source_ids: list[str],
    ) -> str | None:
        if not settings.LONG_TERM_MEMORY_ENABLED or not user_id.strip():
            return None
        entry = self._episode_entry(
            user_id=user_id,
            thread_id=thread_id,
            question=question,
            answer=answer,
            trace=trace,
            source_ids=source_ids,
        )
        if entry is None:
            return None
        saved = await self._store_many([entry])
        return saved[0].memory_id if saved else None

    def _semantic_entries(
        self,
        user_id: str,
        items: list[dict[str, Any]],
        thread_id: str,
    ) -> list[MemoryEntry]:
        entries: list[MemoryEntry] = []
        for item in items[: settings.MEMORY_MAX_WRITES_PER_TURN]:
            content = _normalize(str(item.get("content", "")))
            category = str(item.get("category", "case_fact"))
            if not content or category not in {"case_fact", "preference"}:
                continue
            entries.append(self._entry(
                user_id=user_id,
                memory_type="semantic",
                category=category,
                content=content,
                source_kind="user_preference" if category == "preference" else "user_statement",
                user_stated=True,
                metadata={
                    "thread_id": thread_id,
                    "source_excerpt": str(item.get("source_excerpt", ""))[:1000],
                },
            ))
        return entries

    def _episode_entry(
        self,
        *,
        user_id: str,
        thread_id: str,
        question: str,
        answer: str,
        trace: list[dict[str, Any]],
        source_ids: list[str],
    ) -> MemoryEntry | None:
        tools = [
            tool
            for event in trace
            if event.get("node") == "execute_tools"
            for tool in event.get("tools", [])
        ]
        content = _normalize(
            f"历史任务：{question[:400]}\n处理结论：{answer[:800]}\n"
            f"使用工具：{', '.join(dict.fromkeys(tools)) or '无'}"
        )
        if not content:
            return None
        return self._entry(
            user_id=user_id,
            memory_type="episodic",
            category="task_summary",
            content=content,
            source_kind="agent_episode",
            user_stated=False,
            metadata={
                "thread_id": thread_id,
                "tools": list(dict.fromkeys(tools)),
                "source_ids": source_ids[:10],
            },
        )

    async def _store_many(self, entries: list[MemoryEntry]) -> list[MemoryEntry]:
        if not entries:
            return []
        saved_entries = [
            (await asyncio.to_thread(self.repo.upsert, entry))[0]
            for entry in entries
        ]
        ids = [entry.memory_id for entry in saved_entries]
        metadatas = [{
            "memory_id": entry.memory_id,
            "user_id": entry.user_id,
            "memory_type": entry.memory_type,
            "category": entry.category,
            "is_user_stated": entry.source_kind in {"user_statement", "user_preference", "user_confirmed"},
        } for entry in saved_entries]
        try:
            await asyncio.to_thread(self.vectorstore.delete, ids)
            await asyncio.to_thread(
                self.vectorstore.add_texts,
                [entry.content for entry in saved_entries],
                metadatas,
                ids,
            )
            for memory_id in ids:
                await asyncio.to_thread(self.repo.mark_indexed, memory_id, "indexed")
        except Exception:
            for memory_id in ids:
                await asyncio.to_thread(self.repo.mark_indexed, memory_id, "pending")
        return saved_entries

    @staticmethod
    def _entry(
        *,
        user_id: str,
        memory_type: str,
        category: str,
        content: str,
        source_kind: str,
        user_stated: bool,
        metadata: dict[str, Any],
    ) -> MemoryEntry:
        return MemoryEntry(
            memory_id=str(uuid4()),
            user_id=user_id.strip(),
            memory_type=memory_type,
            category=category,
            content=content,
            content_hash=hashlib.sha256(content.casefold().encode("utf-8")).hexdigest(),
            source_kind=source_kind,
            # Keep the existing database column for schema compatibility. It now
            # means the semantic item is grounded in an explicit user utterance,
            # not that the UI asked for a separate confirmation.
            is_confirmed=user_stated,
            confidence=1.0 if user_stated else 0.5,
            memory_metadata=metadata,
        )

    @staticmethod
    def format_context(memories: list[RecalledMemory]) -> str:
        if not memories:
            return ""
        semantic = [item for item in memories if item.memory_type == "semantic" and item.is_user_stated]
        episodic = [item for item in memories if item.memory_type == "episodic"]
        parts: list[str] = []
        if semantic:
            parts.append("长期语义记忆（来自用户明确陈述，可用于理解偏好和案件事实）：")
            parts.extend(f"- [{item.category}] {item.content}" for item in semantic)
        if episodic:
            parts.append("长期情景记忆（历史任务摘要，仅供流程参考，不得视为用户陈述的事实）：")
            parts.extend(f"- {item.content}" for item in episodic)
        return "\n".join(parts)


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


long_term_memory = LongTermMemoryService()
