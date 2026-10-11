"""Recoverable four-stage compaction for LangChain agent messages.

The pipeline follows the design used by learn-claude-code s08:
tool_result_budget -> snip_compact -> micro_compact -> compact_history.
The first three stages are deterministic.  Only the final history summary may
invoke an LLM, and every removed payload is written to a local recoverable path.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage, message_to_dict


HistorySummarizer = Callable[[list[BaseMessage], str], Awaitable[str]]


@dataclass
class MessageCompactionResult:
    messages: list[BaseMessage]
    chars_before: int
    chars_after: int
    stages: list[str] = field(default_factory=list)
    persisted_outputs: list[str] = field(default_factory=list)
    transcripts: list[str] = field(default_factory=list)
    summary_used: bool = False
    reactive: bool = False

    def to_dict(self) -> dict:
        return {
            "chars_before": self.chars_before,
            "chars_after": self.chars_after,
            "message_count": len(self.messages),
            "stages": self.stages,
            "persisted_output_count": len(self.persisted_outputs),
            "transcript_count": len(self.transcripts),
            "summary_used": self.summary_used,
            "reactive": self.reactive,
        }


class MessageContextCompactor:
    """Compact message history in a fixed, least-loss-first order."""

    def __init__(
        self,
        artifact_root: str | Path,
        *,
        context_char_limit: int = 50000,
        target_ratio: float = 0.8,
        tool_result_budget_chars: int = 200000,
        large_result_char_limit: int = 30000,
        preview_chars: int = 2000,
        max_messages: int = 50,
        keep_recent_results: int = 3,
    ):
        self.context_char_limit = context_char_limit
        self.target_ratio = target_ratio
        self.tool_result_budget_chars = tool_result_budget_chars
        self.large_result_char_limit = large_result_char_limit
        self.preview_chars = preview_chars
        self.max_messages = max_messages
        self.keep_recent_results = keep_recent_results
        root = Path(artifact_root)
        self.output_dir = root / ".task_outputs" / "tool-results"
        self.transcript_dir = root / ".transcripts"

    async def prepare(
        self,
        messages: list[BaseMessage],
        active_request: str,
        summarizer: HistorySummarizer,
    ) -> MessageCompactionResult:
        """Run deterministic stages every turn and summarize only if required."""
        working = [message.model_copy(deep=True) for message in messages]
        result = MessageCompactionResult(
            messages=working,
            chars_before=self.estimate_chars(working),
            chars_after=0,
        )

        result.messages = self.tool_result_budget(result.messages, result)
        result.stages.append("tool_result_budget")
        result.messages = self.snip_compact(result.messages, result)
        result.stages.append("snip_compact")

        if self.estimate_chars(result.messages) > self.context_char_limit:
            target = int(self.context_char_limit * self.target_ratio)
            result.messages = self.micro_compact(result.messages, target, result)
            result.stages.append("micro_compact")
            if self.estimate_chars(result.messages) > self.context_char_limit:
                result.messages = self.fit_tool_results(result.messages, target, result)
            if self.estimate_chars(result.messages) > self.context_char_limit:
                result.messages = await self.compact_history(
                    result.messages,
                    active_request,
                    summarizer,
                    result,
                )
                result.stages.append("compact_history")

        result.chars_after = self.estimate_chars(result.messages)
        return result

    async def reactive_compact(
        self,
        messages: list[BaseMessage],
        active_request: str,
        summarizer: HistorySummarizer,
        keep_recent_messages: int = 5,
    ) -> MessageCompactionResult:
        """One-shot recovery path after an API prompt-too-long rejection."""
        working = [message.model_copy(deep=True) for message in messages]
        result = MessageCompactionResult(
            messages=working,
            chars_before=self.estimate_chars(working),
            chars_after=0,
            stages=["reactive_compact"],
            reactive=True,
        )
        transcript = self.write_transcript(working)
        result.transcripts.append(str(transcript))
        tail_start = max(0, len(working) - keep_recent_messages)
        tail_start = self._protect_tail_boundary(working, tail_start)
        old_history = working[:tail_start] if tail_start else working
        tail = working[tail_start:] if tail_start else []
        summary = await summarizer(old_history, active_request)
        result.messages = [self._summary_message(
            "Reactive compact",
            active_request,
            summary,
            transcript,
        ), *tail]
        result.summary_used = True
        result.chars_after = self.estimate_chars(result.messages)
        return result

    def tool_result_budget(
        self,
        messages: list[BaseMessage],
        result: MessageCompactionResult,
    ) -> list[BaseMessage]:
        """Spill oversized results from the newest tool batch when over budget."""
        positions = self._latest_tool_result_positions(messages)
        total = sum(len(self._content_text(messages[index])) for index in positions)
        ranked = sorted(
            positions,
            key=lambda index: len(self._content_text(messages[index])),
            reverse=True,
        )
        for index in ranked:
            if total <= self.tool_result_budget_chars:
                break
            content = self._content_text(messages[index])
            if len(content) <= self.large_result_char_limit:
                continue
            path = self.persist_tool_result(messages[index], content)
            result.persisted_outputs.append(str(path))
            preview = content[: self.preview_chars]
            replacement = (
                f"[Large tool result saved at {path}]\n"
                f"Preview ({len(preview)} of {len(content)} chars):\n{preview}"
            )
            messages[index] = messages[index].model_copy(update={"content": replacement})
            total = sum(len(self._content_text(messages[item])) for item in positions)
        return messages

    def snip_compact(
        self,
        messages: list[BaseMessage],
        result: MessageCompactionResult,
    ) -> list[BaseMessage]:
        """Archive long histories while preserving tool-call/result boundaries."""
        if len(messages) <= self.max_messages:
            return messages
        transcript = self.write_transcript(messages)
        result.transcripts.append(str(transcript))
        head_end = min(3, len(messages))
        tail_start = max(head_end, len(messages) - (self.max_messages - head_end - 1))

        if head_end and self._has_tool_calls(messages[head_end - 1]):
            while head_end < tail_start and isinstance(messages[head_end], ToolMessage):
                head_end += 1
        tail_start = self._protect_tail_boundary(messages, tail_start)
        if tail_start <= head_end:
            return messages

        removed = tail_start - head_end
        marker = HumanMessage(content=f"[{removed} messages archived at {transcript}]")
        return [*messages[:head_end], marker, *messages[tail_start:]]

    def micro_compact(
        self,
        messages: list[BaseMessage],
        target_chars: int,
        result: MessageCompactionResult,
    ) -> list[BaseMessage]:
        """Replace older consumed tool results with recoverable file references."""
        unseen = set(self._latest_tool_result_positions(messages))
        consumed = [
            index for index, message in enumerate(messages)
            if isinstance(message, ToolMessage) and index not in unseen
        ]
        candidates = consumed[:-self.keep_recent_results] if self.keep_recent_results else consumed
        for index in candidates:
            if self.estimate_chars(messages) <= target_chars:
                break
            content = self._content_text(messages[index])
            if len(content) <= 120:
                continue
            path = self._path_from_marker(content)
            if path is None:
                path = self.persist_tool_result(messages[index], content)
                result.persisted_outputs.append(str(path))
            messages[index] = messages[index].model_copy(
                update={"content": f"[Earlier tool result saved at {path}]"}
            )
        return messages

    def fit_tool_results(
        self,
        messages: list[BaseMessage],
        target_chars: int,
        result: MessageCompactionResult,
    ) -> list[BaseMessage]:
        """Fit an unseen latest batch before falling back to history summary."""
        positions = sorted(
            self._latest_tool_result_positions(messages),
            key=lambda index: len(self._content_text(messages[index])),
            reverse=True,
        )
        for index in positions:
            if self.estimate_chars(messages) <= target_chars:
                break
            content = self._content_text(messages[index])
            path = self._path_from_marker(content)
            if path is None:
                path = self.persist_tool_result(messages[index], content)
                result.persisted_outputs.append(str(path))
            preview = content[: min(1000, self.preview_chars)]
            messages[index] = messages[index].model_copy(update={
                "content": (
                    f"[Unseen tool result saved at {path}]\n"
                    f"Preview ({len(preview)} of {len(content)} chars):\n{preview}"
                )
            })
        return messages

    async def compact_history(
        self,
        messages: list[BaseMessage],
        active_request: str,
        summarizer: HistorySummarizer,
        result: MessageCompactionResult | None = None,
    ) -> list[BaseMessage]:
        """Archive and replace history with a factual model-generated state summary."""
        transcript = self.write_transcript(messages)
        summary = await summarizer(messages, active_request)
        if result is not None:
            result.transcripts.append(str(transcript))
            result.summary_used = True
        return [self._summary_message("Compacted", active_request, summary, transcript)]

    def persist_tool_result(self, message: BaseMessage, content: str) -> Path:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        raw_id = str(getattr(message, "tool_call_id", "unknown") or "unknown")
        safe_id = re.sub(r"[^0-9A-Za-z_.-]+", "_", raw_id)[:80] or "unknown"
        path = self.output_dir / f"{safe_id}-{time.time_ns()}.txt"
        path.write_text(content, encoding="utf-8")
        return path.resolve()

    def write_transcript(self, messages: list[BaseMessage]) -> Path:
        self.transcript_dir.mkdir(parents=True, exist_ok=True)
        path = self.transcript_dir / f"transcript-{time.time_ns()}.json"
        payload = [message_to_dict(message) for message in messages]
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
        return path.resolve()

    @staticmethod
    def estimate_chars(messages: list[BaseMessage]) -> int:
        payload = [message_to_dict(message) for message in messages]
        return len(json.dumps(payload, ensure_ascii=False, default=str))

    @staticmethod
    def is_context_limit_error(error: Exception) -> bool:
        message = str(error).casefold()
        return "prompt_too_long" in message or "too many tokens" in message or "context length" in message

    @staticmethod
    def _content_text(message: BaseMessage) -> str:
        content = message.content
        if isinstance(content, str):
            return content
        return json.dumps(content, ensure_ascii=False, default=str)

    @staticmethod
    def _has_tool_calls(message: BaseMessage) -> bool:
        return isinstance(message, AIMessage) and bool(message.tool_calls)

    @staticmethod
    def _latest_tool_result_positions(messages: list[BaseMessage]) -> list[int]:
        positions: list[int] = []
        index = len(messages) - 1
        while index >= 0 and isinstance(messages[index], ToolMessage):
            positions.append(index)
            index -= 1
        return list(reversed(positions))

    @classmethod
    def _protect_tail_boundary(cls, messages: list[BaseMessage], tail_start: int) -> int:
        if not messages or tail_start <= 0 or tail_start >= len(messages):
            return tail_start
        if not isinstance(messages[tail_start], ToolMessage):
            return tail_start
        first_tool = tail_start
        while first_tool > 0 and isinstance(messages[first_tool - 1], ToolMessage):
            first_tool -= 1
        if first_tool > 0 and cls._has_tool_calls(messages[first_tool - 1]):
            return first_tool - 1
        return first_tool

    @staticmethod
    def _path_from_marker(content: str) -> Path | None:
        match = re.search(r"(?:saved at|archived at) ([^\]\n]+)", content)
        return Path(match.group(1)).resolve() if match else None

    @staticmethod
    def _summary_message(label: str, active_request: str, summary: str, transcript: Path) -> HumanMessage:
        return HumanMessage(content=(
            f"[{label}]\n"
            f"Current user request:\n{active_request}\n\n"
            f"Conversation summary:\n{summary}\n\n"
            f"Full transcript: {transcript}"
        ))
