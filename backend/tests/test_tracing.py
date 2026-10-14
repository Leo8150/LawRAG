"""Offline tests for lightweight Trace/Span observability."""

import asyncio
import time

import pytest
from langchain_core.messages import AIMessage

from app.tracing.core import TraceCollector, activate_trace, trace_node, trace_span, traced_ainvoke
from app.tracing.repository import TraceRepository
from app.tracing.persistence import TracePersistenceService


class UsageLLM:
    async def ainvoke(self, value):
        return AIMessage(
            content="ok",
            usage_metadata={"input_tokens": 7, "output_tokens": 3, "total_tokens": 10},
        )


def test_nested_spans_link_and_count_leaf_tokens_once():
    collector = TraceCollector(metadata={"architecture": "test"})
    with activate_trace(collector):
        with trace_span("generate", "node"):
            asyncio.run(traced_ainvoke("llm.generate", UsageLLM(), "not persisted"))

    snapshot = collector.finish()
    node, llm = snapshot["spans"]
    assert llm["parent_span_id"] == node["span_id"]
    assert node["total_tokens"] == 0
    assert snapshot["total_tokens"] == 10
    assert snapshot["status"] == "success"


def test_trace_node_supports_sync_async_and_records_error():
    @trace_node("sync_node")
    def sync_node(state):
        return state["value"]

    @trace_node("async_node")
    async def async_node(state):
        raise ValueError("safe failure")

    collector = TraceCollector()
    with activate_trace(collector):
        assert sync_node({"question": "redacted", "value": 1}) == 1
        with pytest.raises(ValueError):
            asyncio.run(async_node({"question": "redacted"}))
    snapshot = collector.finish()
    assert [span["status"] for span in snapshot["spans"]] == ["success", "error"]
    assert "ValueError" in snapshot["spans"][1]["error_summary"]
    assert "question" not in snapshot["spans"][0]["metadata"]


def test_trace_repository_round_trip_and_non_nested_stats(tmp_path):
    repository = TraceRepository(f"sqlite:///{tmp_path / 'traces.db'}")
    collector = TraceCollector(metadata={"architecture": "agentic_rag"})
    with activate_trace(collector):
        with trace_span("prepare", "node"):
            time.sleep(0.001)
        with trace_span("llm.route", "llm") as span:
            span.tokens(input_tokens=11, output_tokens=4)
    snapshot = collector.finish()
    repository.save(snapshot)

    restored = repository.get(collector.trace_id)
    assert restored is not None
    assert restored["metadata"] == {"architecture": "agentic_rag"}
    assert restored["spans"][0]["name"] == "prepare"
    stats = repository.stats()
    assert stats["trace_count"] == 1
    assert stats["total_tokens"] == 15
    assert stats["p95_latency_ms"] == snapshot["duration_ms"]
    assert stats["nodes"][0]["name"] == "prepare"


def test_sensitive_metadata_is_redacted():
    collector = TraceCollector(metadata={"api_key": "secret", "question": "private"})
    with activate_trace(collector):
        with trace_span("mcp.call", "mcp", {"authorization": "Bearer abc", "server": "legal"}):
            pass
    snapshot = collector.finish()
    assert snapshot["metadata"]["api_key"] == "[REDACTED]"
    assert snapshot["metadata"]["question"] == "[REDACTED]"
    assert snapshot["spans"][0]["metadata"]["authorization"] == "[REDACTED]"


def test_non_blocking_persistence_batches_and_is_failure_isolated():
    batches = []

    class FakeRepository:
        def save_many(self, snapshots):
            batches.append(snapshots)

    async def scenario():
        service = TracePersistenceService(FakeRepository)
        service.enqueue({"trace_id": "one"})
        service.enqueue({"trace_id": "two"})
        await service.flush()

        failing = TracePersistenceService(lambda: (_ for _ in ()).throw(RuntimeError("db down")))
        failing.enqueue({"trace_id": "ignored"})
        await failing.flush()  # tracing failures must not escape

    asyncio.run(scenario())
    assert [[item["trace_id"] for item in batch] for batch in batches] == [["one", "two"]]
