"""In-process Trace/Span collection based on contextvars.

The collector deliberately stores only operational metadata. Prompt text,
answers, tool results, user identifiers and credentials are never captured.
"""

from __future__ import annotations

import inspect
import re
import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar, Token
from datetime import datetime, timezone
from functools import wraps
from typing import Any, Iterator
from uuid import uuid4


_collector_var: ContextVar[TraceCollector | None] = ContextVar("rag_trace_collector", default=None)
_span_var: ContextVar[str | None] = ContextVar("rag_parent_span", default=None)
_SECRET_KEYS = {"api_key", "apikey", "authorization", "token", "password", "secret", "query", "question", "prompt", "content", "answer"}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(tzinfo=None)


def _safe_error(exc: BaseException) -> str:
    value = re.sub(r"(?i)(bearer\s+|api[_-]?key[=: ]+)[^\s,;]+", r"\1[REDACTED]", str(exc))
    value = re.sub(r"\bsk-[A-Za-z0-9_-]+", "[REDACTED]", value)
    return f"{type(exc).__name__}: {value}"[:500]


def _safe_metadata(value: Any, *, key: str = "") -> Any:
    """Keep metadata JSON-safe and reject likely sensitive/free-text fields."""
    if key.casefold() in _SECRET_KEYS:
        return "[REDACTED]"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return value[:256]
    if isinstance(value, (list, tuple, set)):
        return [_safe_metadata(item) for item in list(value)[:100]]
    if isinstance(value, dict):
        return {
            str(k)[:64]: _safe_metadata(v, key=str(k))
            for k, v in list(value.items())[:100]
        }
    return str(value)[:256]


def _usage(response: Any) -> tuple[int, int]:
    usage = getattr(response, "usage_metadata", None) or {}
    if usage:
        return int(usage.get("input_tokens", 0) or 0), int(usage.get("output_tokens", 0) or 0)
    meta = getattr(response, "response_metadata", None) or {}
    usage = meta.get("token_usage") or meta.get("usage") or {}
    return (
        int(usage.get("prompt_tokens", usage.get("input_tokens", 0)) or 0),
        int(usage.get("completion_tokens", usage.get("output_tokens", 0)) or 0),
    )


class SpanHandle:
    def __init__(self, collector: TraceCollector | None, span_id: str | None):
        self.collector = collector
        self.span_id = span_id

    def metadata(self, **values: Any) -> None:
        if self.collector and self.span_id:
            self.collector.update_span(self.span_id, metadata=values)

    def tokens(self, *, input_tokens: int = 0, output_tokens: int = 0) -> None:
        if self.collector and self.span_id:
            self.collector.update_span(
                self.span_id,
                input_tokens=input_tokens,
                output_tokens=output_tokens,
            )

    def response_tokens(self, response: Any) -> None:
        input_tokens, output_tokens = _usage(response)
        self.tokens(input_tokens=input_tokens, output_tokens=output_tokens)


class TraceCollector:
    """One request-local trace with ordered child spans."""

    def __init__(self, *, trace_id: str | None = None, metadata: dict | None = None):
        self.trace_id = trace_id or uuid4().hex
        self.started_at = _utc_now()
        self._started_perf = time.perf_counter()
        self.ended_at: datetime | None = None
        self.duration_ms = 0.0
        self.status = "running"
        self.error_summary: str | None = None
        self.metadata = _safe_metadata(metadata or {})
        self._spans: list[dict[str, Any]] = []
        self._by_id: dict[str, dict[str, Any]] = {}
        self._lock = threading.RLock()

    def start_span(self, name: str, kind: str, metadata: dict | None = None) -> str:
        span_id = uuid4().hex
        with self._lock:
            span = {
                "span_id": span_id,
                "trace_id": self.trace_id,
                "parent_span_id": _span_var.get(),
                "name": str(name)[:128],
                "kind": str(kind)[:32],
                "sequence": len(self._spans) + 1,
                "status": "running",
                "started_at": _utc_now(),
                "ended_at": None,
                "duration_ms": 0.0,
                "input_tokens": 0,
                "output_tokens": 0,
                "total_tokens": 0,
                "error_summary": None,
                "metadata": _safe_metadata(metadata or {}),
                "_started_perf": time.perf_counter(),
            }
            self._spans.append(span)
            self._by_id[span_id] = span
        return span_id

    def update_span(
        self,
        span_id: str,
        *,
        metadata: dict | None = None,
        input_tokens: int | None = None,
        output_tokens: int | None = None,
    ) -> None:
        with self._lock:
            span = self._by_id.get(span_id)
            if not span:
                return
            if metadata:
                span["metadata"].update(_safe_metadata(metadata))
            if input_tokens is not None:
                span["input_tokens"] = max(0, int(input_tokens))
            if output_tokens is not None:
                span["output_tokens"] = max(0, int(output_tokens))
            span["total_tokens"] = span["input_tokens"] + span["output_tokens"]

    def end_span(self, span_id: str, exc: BaseException | None = None) -> None:
        with self._lock:
            span = self._by_id.get(span_id)
            if not span or span["status"] != "running":
                return
            span["ended_at"] = _utc_now()
            span["duration_ms"] = round((time.perf_counter() - span.pop("_started_perf")) * 1000, 2)
            span["status"] = "error" if exc else "success"
            span["error_summary"] = _safe_error(exc) if exc else None

    def finish(self, exc: BaseException | None = None) -> dict[str, Any]:
        with self._lock:
            for span in self._spans:
                if span["status"] == "running":
                    self.end_span(span["span_id"], exc)
            self.ended_at = _utc_now()
            self.duration_ms = round((time.perf_counter() - self._started_perf) * 1000, 2)
            self.status = "error" if exc else "success"
            self.error_summary = _safe_error(exc) if exc else None
            return self.snapshot()

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            # Token-bearing leaf operation spans are the single source of truth;
            # parent node spans intentionally carry zero tokens.
            input_tokens = sum(s["input_tokens"] for s in self._spans)
            output_tokens = sum(s["output_tokens"] for s in self._spans)
            duration_ms = self.duration_ms if self.ended_at is not None else round(
                (time.perf_counter() - self._started_perf) * 1000, 2
            )
            spans = [{k: v for k, v in span.items() if not k.startswith("_")} for span in self._spans]
            return {
                "trace_id": self.trace_id,
                "status": self.status,
                "started_at": self.started_at,
                "ended_at": self.ended_at,
                "duration_ms": duration_ms,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "error_summary": self.error_summary,
                "metadata": self.metadata,
                "spans": spans,
            }


@contextmanager
def activate_trace(collector: TraceCollector) -> Iterator[TraceCollector]:
    token = _collector_var.set(collector)
    parent_token = _span_var.set(None)
    try:
        yield collector
    finally:
        _span_var.reset(parent_token)
        _collector_var.reset(token)


@contextmanager
def trace_span(name: str, kind: str = "operation", metadata: dict | None = None) -> Iterator[SpanHandle]:
    collector = _collector_var.get()
    if collector is None:
        yield SpanHandle(None, None)
        return
    span_id: str | None = None
    token: Token | None = None
    try:
        span_id = collector.start_span(name, kind, metadata)
        token = _span_var.set(span_id)
        handle = SpanHandle(collector, span_id)
    except Exception:
        yield SpanHandle(None, None)
        return
    try:
        yield handle
    except BaseException as exc:
        try:
            collector.end_span(span_id, exc)
        except Exception:
            pass
        raise
    else:
        try:
            collector.end_span(span_id)
        except Exception:
            pass
    finally:
        if token is not None:
            try:
                _span_var.reset(token)
            except Exception:
                pass


def _node_metadata(args: tuple[Any, ...], kwargs: dict[str, Any]) -> dict[str, Any]:
    state = next((arg for arg in args if isinstance(arg, dict) and "question" in arg), None)
    if state is None:
        state = kwargs.get("state") if isinstance(kwargs.get("state"), dict) else {}
    return {
        key: int(state.get(key, 0) or 0)
        for key in ("tool_rounds", "retrieval_rounds", "generation_rounds")
    }


def trace_node(name: str | None = None):
    """Trace a sync or async LangGraph node without changing its signature."""
    def decorate(func):
        span_name = name or func.__name__.lstrip("_")
        if inspect.iscoroutinefunction(func):
            @wraps(func)
            async def async_wrapper(*args, **kwargs):
                with trace_span(span_name, "node", _node_metadata(args, kwargs)):
                    return await func(*args, **kwargs)
            return async_wrapper

        @wraps(func)
        def sync_wrapper(*args, **kwargs):
            with trace_span(span_name, "node", _node_metadata(args, kwargs)):
                return func(*args, **kwargs)
        return sync_wrapper
    return decorate


async def traced_ainvoke(name: str, runnable: Any, value: Any, *, metadata: dict | None = None) -> Any:
    """Invoke a LangChain runnable and collect provider-reported token usage."""
    with trace_span(name, "llm", metadata) as span:
        response = await runnable.ainvoke(value)
        span.response_tokens(response)
        return response


def evidence_metadata(documents: list[Any]) -> dict[str, Any]:
    child_ids: list[str] = []
    parent_ids: list[str] = []
    law_ids: list[str] = []
    for document in documents:
        metadata = getattr(document, "metadata", {}) or {}
        child_ids.extend(str(v) for v in [metadata.get("child_chunk_id")] if v)
        parent_ids.extend(str(v) for v in [metadata.get("parent_chunk_id")] if v)
        law_value = metadata.get("law_id") or metadata.get("article_number")
        if law_value:
            law_ids.append(str(law_value))
    return {
        "result_count": len(documents),
        "chunk_ids": list(dict.fromkeys(child_ids))[:100],
        "parent_ids": list(dict.fromkeys(parent_ids))[:100],
        "law_ids": list(dict.fromkeys(law_ids))[:100],
    }
