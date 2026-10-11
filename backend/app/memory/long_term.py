"""Semantic and episodic long-term memory backed by MySQL and ChromaDB."""

from __future__ import annotations

import asyncio
import hashlib
import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
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
    similarity: float = 0.0
    strength: float = 0.0
    score: float = 0.0


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
            candidates = await asyncio.to_thread(
                self.vectorstore.similarity_search_with_relevance_scores,
                query,
                max(limit, settings.MEMORY_CANDIDATE_TOP_K),
                filter={"user_id": user_id},
            )
            memory_ids = [
                str(doc.metadata.get("memory_id", ""))
                for doc, _ in candidates
                if doc.metadata.get("memory_id")
            ]
            similarities = {
                str(doc.metadata.get("memory_id")): _clamp(float(score))
                for doc, score in candidates
                if doc.metadata.get("memory_id")
            }
            rows = await asyncio.to_thread(self.repo.fetch, user_id, memory_ids)
            now = datetime.utcnow()
            ranked: list[RecalledMemory] = []
            for row in rows:
                similarity = similarities.get(row.memory_id, 0.0)
                strength = self.memory_strength(row, now)
                ranked.append(RecalledMemory(
                    memory_id=row.memory_id,
                    memory_type=row.memory_type,
                    category=row.category,
                    content=row.content,
                    is_user_stated=row.source_kind in {
                        "user_statement", "user_preference", "user_confirmed"
                    },
                    source_kind=row.source_kind,
                    similarity=similarity,
                    strength=strength,
                    score=self.memory_score(similarity, strength),
                ))
            ranked.sort(key=lambda item: item.score, reverse=True)
            selected = self._apply_category_limits(ranked, limit)
            await asyncio.to_thread(
                self.repo.record_access,
                user_id,
                [item.memory_id for item in selected],
            )
            return selected
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

    @staticmethod
    def memory_strength(entry: MemoryEntry, now: datetime | None = None) -> float:
        """Return importance after exponential half-life decay."""
        current = now or datetime.utcnow()
        reference = entry.last_accessed_at or entry.updated_at or entry.created_at
        idle_days = max(0.0, (current - reference).total_seconds() / 86400)
        half_life = max(1, int(entry.half_life_days or 1))
        decay = math.exp(-math.log(2) * idle_days / half_life)
        return _clamp(float(entry.importance) * decay)

    @staticmethod
    def memory_score(similarity: float, strength: float) -> float:
        return (
            settings.MEMORY_SIMILARITY_WEIGHT * _clamp(similarity)
            + settings.MEMORY_STRENGTH_WEIGHT * _clamp(strength)
        )

    @staticmethod
    def _apply_category_limits(
        memories: list[RecalledMemory],
        limit: int,
    ) -> list[RecalledMemory]:
        category_limits = {"case_fact": 3, "preference": 1, "task_summary": 2}
        counts: dict[str, int] = {}
        selected: list[RecalledMemory] = []
        for memory in memories:
            category_limit = category_limits.get(memory.category, limit)
            if counts.get(memory.category, 0) >= category_limit:
                continue
            selected.append(memory)
            counts[memory.category] = counts.get(memory.category, 0) + 1
            if len(selected) >= limit:
                break
        return selected

    async def maintain(self, batch_size: int | None = None) -> dict[str, int]:
        """Soft-forget weak memories, remove vectors, and purge old tombstones."""
        limit = batch_size or settings.MEMORY_FORGET_BATCH_SIZE
        rows = await asyncio.to_thread(self.repo.list_active_for_maintenance, limit)
        now = datetime.utcnow()
        forgotten_ids = [
            row.memory_id
            for row in rows
            if self.memory_strength(row, now) < settings.MEMORY_FORGET_THRESHOLD
        ]
        forgotten = await asyncio.to_thread(self.repo.soft_forget, forgotten_ids, now)
        if forgotten_ids:
            await asyncio.to_thread(self.vectorstore.delete, forgotten_ids)
        purge_before = now - timedelta(days=settings.MEMORY_FORGET_RETENTION_DAYS)
        purged = await asyncio.to_thread(self.repo.purge_forgotten, purge_before, limit)
        return {"scanned": len(rows), "forgotten": forgotten, "purged": purged}

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
        importance, half_life_days = _memory_policy(category)
        return MemoryEntry(
            memory_id=str(uuid4()),
            user_id=user_id.strip(),
            memory_type=memory_type,
            category=category,
            importance=importance,
            half_life_days=half_life_days,
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


def _clamp(value: float) -> float:
    return max(0.0, min(1.0, value))


def _memory_policy(category: str) -> tuple[float, int]:
    if category == "case_fact":
        return (
            settings.MEMORY_CASE_FACT_IMPORTANCE,
            settings.MEMORY_CASE_FACT_HALF_LIFE_DAYS,
        )
    if category == "preference":
        return (
            settings.MEMORY_PREFERENCE_IMPORTANCE,
            settings.MEMORY_PREFERENCE_HALF_LIFE_DAYS,
        )
    return settings.MEMORY_EPISODIC_IMPORTANCE, settings.MEMORY_EPISODIC_HALF_LIFE_DAYS


long_term_memory = LongTermMemoryService()
