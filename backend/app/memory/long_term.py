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
    is_confirmed: bool
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
                is_confirmed=row.is_confirmed,
                source_kind=row.source_kind,
            ) for row in rows]
        except Exception:
            # Long-term memory is enrichment; retrieval failure must not block legal QA.
            return []

    async def remember_confirmed(
        self,
        user_id: str,
        items: list[dict[str, Any]],
        thread_id: str,
    ) -> list[str]:
        written: list[str] = []
        for item in items[: settings.MEMORY_MAX_WRITES_PER_TURN]:
            content = _normalize(str(item.get("content", "")))
            category = str(item.get("category", "case_fact"))
            if not content or category not in {"case_fact", "preference"}:
                continue
            entry = self._entry(
                user_id=user_id,
                memory_type="semantic",
                category=category,
                content=content,
                source_kind="user_confirmed",
                confirmed=True,
                metadata={"thread_id": thread_id},
            )
            saved = await self._store(entry)
            written.append(saved.memory_id)
        return written

    async def remember_episode(
        self,
        user_id: str,
        thread_id: str,
        question: str,
        answer: str,
        trace: list[dict[str, Any]],
        source_ids: list[str],
    ) -> str | None:
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
        entry = self._entry(
            user_id=user_id,
            memory_type="episodic",
            category="task_summary",
            content=content,
            source_kind="agent_episode",
            confirmed=False,
            metadata={
                "thread_id": thread_id,
                "tools": list(dict.fromkeys(tools)),
                "source_ids": source_ids[:10],
            },
        )
        return (await self._store(entry)).memory_id

    async def _store(self, entry: MemoryEntry) -> MemoryEntry:
        saved, _ = await asyncio.to_thread(self.repo.upsert, entry)
        metadata = {
            "memory_id": saved.memory_id,
            "user_id": saved.user_id,
            "memory_type": saved.memory_type,
            "category": saved.category,
            "is_confirmed": saved.is_confirmed,
        }
        try:
            await asyncio.to_thread(self.vectorstore.delete, [saved.memory_id])
            await asyncio.to_thread(
                self.vectorstore.add_texts,
                [saved.content],
                [metadata],
                [saved.memory_id],
            )
            await asyncio.to_thread(self.repo.mark_indexed, saved.memory_id, "indexed")
        except Exception:
            await asyncio.to_thread(self.repo.mark_indexed, saved.memory_id, "pending")
        return saved

    @staticmethod
    def _entry(
        *,
        user_id: str,
        memory_type: str,
        category: str,
        content: str,
        source_kind: str,
        confirmed: bool,
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
            is_confirmed=confirmed,
            confidence=1.0 if confirmed else 0.5,
            memory_metadata=metadata,
        )

    @staticmethod
    def format_context(memories: list[RecalledMemory]) -> str:
        if not memories:
            return ""
        semantic = [item for item in memories if item.memory_type == "semantic" and item.is_confirmed]
        episodic = [item for item in memories if item.memory_type == "episodic"]
        parts: list[str] = []
        if semantic:
            parts.append("长期语义记忆（用户已确认，可用于理解偏好和案件事实）：")
            parts.extend(f"- [{item.category}] {item.content}" for item in semantic)
        if episodic:
            parts.append("长期情景记忆（历史任务摘要，仅供流程参考，不得视为已确认事实）：")
            parts.extend(f"- {item.content}" for item in episodic)
        return "\n".join(parts)


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


long_term_memory = LongTermMemoryService()
