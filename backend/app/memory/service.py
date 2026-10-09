"""TTL-based short-term memory for multi-turn legal questions."""

from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field

from app.config import settings


@dataclass
class MemoryTurn:
    question: str
    answer: str
    sources: list[str] = field(default_factory=list)
    skill: str = "general_legal"
    created_at: float = field(default_factory=time.time)


@dataclass
class ConversationMemory:
    conversation_id: str
    turns: list[MemoryTurn] = field(default_factory=list)
    expires_at: float = 0.0


class ShortTermMemoryService:
    FOLLOW_UP_MARKERS = (
        "那么", "那如果", "如果", "这种情况", "上述", "前面", "还有", "以及", "呢", "他", "她", "该行为",
    )
    NEW_TOPIC_MARKERS = ("换个问题", "另一个问题", "重新问", "新案件", "另外一件事")

    def __init__(self, ttl_seconds: int = 86400, max_turns: int = 6, max_sessions: int = 1000):
        self.ttl_seconds = ttl_seconds
        self.max_turns = max_turns
        self.max_sessions = max_sessions
        self._items: dict[str, ConversationMemory] = {}
        self._lock = threading.RLock()

    def _purge(self) -> None:
        now = time.time()
        expired = [key for key, value in self._items.items() if value.expires_at <= now]
        for key in expired:
            self._items.pop(key, None)
        if len(self._items) > self.max_sessions:
            oldest = sorted(self._items.values(), key=lambda item: item.expires_at)
            for item in oldest[: len(self._items) - self.max_sessions]:
                self._items.pop(item.conversation_id, None)

    def load(self, conversation_id: str) -> ConversationMemory:
        with self._lock:
            self._purge()
            item = self._items.get(conversation_id)
            if item is None:
                item = ConversationMemory(conversation_id=conversation_id)
                self._items[conversation_id] = item
            item.expires_at = time.time() + self.ttl_seconds
            return item

    def is_follow_up(self, question: str, memory: ConversationMemory) -> bool:
        if not memory.turns:
            return False
        normalized = re.sub(r"\s+", "", question)
        if any(marker in normalized for marker in self.NEW_TOPIC_MARKERS):
            return False
        return any(marker in normalized for marker in self.FOLLOW_UP_MARKERS)

    def build_context(self, memory: ConversationMemory) -> str:
        if not memory.turns:
            return ""
        parts = ["以下是同一会话中最近的法律问答，仅用于理解当前追问："]
        for index, turn in enumerate(memory.turns[-3:], 1):
            parts.append(f"第{index}轮问题：{turn.question[:240]}")
            parts.append(f"第{index}轮结论：{turn.answer[:360]}")
        return "\n".join(parts)

    def resolve_question(self, question: str, memory: ConversationMemory) -> tuple[str, str]:
        context = self.build_context(memory)
        if not context or not self.is_follow_up(question, memory):
            return question, context
        latest = memory.turns[-1]
        resolved = f"上一轮问题：{latest.question}\n当前追问：{question}"
        return resolved, context

    def append(self, conversation_id: str, question: str, answer: str, sources: list[str], skill: str) -> None:
        with self._lock:
            item = self.load(conversation_id)
            item.turns.append(MemoryTurn(
                question=question,
                answer=answer,
                sources=sources[:10],
                skill=skill,
            ))
            item.turns = item.turns[-self.max_turns :]
            item.expires_at = time.time() + self.ttl_seconds

    def clear(self, conversation_id: str) -> bool:
        with self._lock:
            return self._items.pop(conversation_id, None) is not None


memory_service = ShortTermMemoryService(
    ttl_seconds=settings.MEMORY_TTL_SECONDS,
    max_turns=settings.MEMORY_MAX_TURNS,
)
