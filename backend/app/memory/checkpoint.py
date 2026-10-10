"""LangGraph checkpointer and bounded short-term conversation state."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass

from langgraph.checkpoint.memory import InMemorySaver

from app.config import settings


@dataclass
class ShortTermSnapshot:
    summary: str
    recent_turns: list[dict]
    case_facts: list[str]
    turn_count: int


class ShortTermCheckpointManager:
    """Own the process-local checkpointer and enforce TTL per thread_id."""

    def __init__(self):
        self.saver = InMemorySaver()
        self._touched: dict[str, float] = {}

    async def config(self, thread_id: str) -> dict:
        await self._purge_expired()
        self._touched[thread_id] = time.time()
        return {"configurable": {"thread_id": thread_id}}

    async def clear(self, thread_id: str) -> bool:
        existed = thread_id in self._touched
        await self.saver.adelete_thread(thread_id)
        self._touched.pop(thread_id, None)
        return existed

    async def _purge_expired(self) -> None:
        threshold = time.time() - settings.MEMORY_TTL_SECONDS
        expired = [key for key, touched in self._touched.items() if touched <= threshold]
        for thread_id in expired:
            await self.saver.adelete_thread(thread_id)
            self._touched.pop(thread_id, None)


class ShortTermContextBuilder:
    FOLLOW_UP_MARKERS = (
        "那么", "那如果", "如果", "这种情况", "上述", "前面", "还有", "以及", "呢",
        "他", "她", "该行为", "该合同", "该案件",
    )
    NEW_TOPIC_MARKERS = ("换个问题", "另一个问题", "重新问", "新案件", "另外一件事")

    @classmethod
    def from_values(cls, values: dict) -> ShortTermSnapshot:
        return ShortTermSnapshot(
            summary=str(values.get("conversation_summary") or ""),
            recent_turns=list(values.get("recent_turns") or []),
            case_facts=list(values.get("case_facts") or []),
            turn_count=int(values.get("memory_turns") or 0),
        )

    @classmethod
    def build_context(cls, snapshot: ShortTermSnapshot) -> str:
        parts: list[str] = []
        if snapshot.summary:
            parts.extend(["历史消息压缩摘要：", snapshot.summary])
        if snapshot.case_facts:
            parts.append("用户已确认的本案事实：")
            parts.extend(f"- {fact}" for fact in snapshot.case_facts[-12:])
        if snapshot.recent_turns:
            parts.append("最近对话：")
            for turn in snapshot.recent_turns[-settings.SHORT_TERM_RECENT_TURNS :]:
                parts.append(f"- 用户：{str(turn.get('question', ''))[:240]}")
                parts.append(f"  助手：{str(turn.get('answer', ''))[:360]}")
        return "\n".join(parts)

    @classmethod
    def resolve_question(cls, question: str, snapshot: ShortTermSnapshot) -> str:
        if not (snapshot.summary or snapshot.recent_turns):
            return question
        normalized = re.sub(r"\s+", "", question)
        if any(marker in normalized for marker in cls.NEW_TOPIC_MARKERS):
            return question
        if not any(marker in normalized for marker in cls.FOLLOW_UP_MARKERS):
            return question
        if snapshot.recent_turns:
            latest = snapshot.recent_turns[-1]
            return f"上一轮问题：{latest.get('question', '')}\n当前追问：{question}"
        return f"历史案件摘要：{snapshot.summary}\n当前追问：{question}"

    @classmethod
    def advance(
        cls,
        snapshot: ShortTermSnapshot,
        question: str,
        answer: str,
        confirmed_case_facts: list[str],
    ) -> ShortTermSnapshot:
        turns = [*snapshot.recent_turns, {"question": question, "answer": answer}]
        overflow = turns[:-settings.SHORT_TERM_RECENT_TURNS]
        recent = turns[-settings.SHORT_TERM_RECENT_TURNS :]
        summary_parts = [snapshot.summary] if snapshot.summary else []
        for turn in overflow:
            summary_parts.append(
                f"问：{str(turn.get('question', ''))[:180]} 答：{str(turn.get('answer', ''))[:260]}"
            )
        summary = "\n".join(part for part in summary_parts if part)
        summary = summary[-settings.SHORT_TERM_SUMMARY_MAX_CHARS :]
        facts = list(dict.fromkeys([
            *snapshot.case_facts,
            *(fact.strip() for fact in confirmed_case_facts if fact.strip()),
        ]))[-settings.SHORT_TERM_MAX_CASE_FACTS :]
        return ShortTermSnapshot(
            summary=summary,
            recent_turns=recent,
            case_facts=facts,
            turn_count=snapshot.turn_count + 1,
        )


checkpoint_manager = ShortTermCheckpointManager()
short_term_context = ShortTermContextBuilder()
