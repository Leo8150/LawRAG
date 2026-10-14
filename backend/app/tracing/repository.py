"""SQLAlchemy persistence and aggregate queries for RAG traces."""

from __future__ import annotations

import math
from collections import defaultdict

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import RAGSpan, RAGTrace


def _percentile(values: list[float], percentile: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(0, math.ceil(percentile * len(ordered)) - 1)
    return round(float(ordered[index]), 2)


class TraceRepository:
    def __init__(self, database_url: str | None = None, *, create_schema: bool = True):
        url = database_url or settings.MYSQL_URL
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        self.engine = create_engine(url, pool_pre_ping=True, connect_args=connect_args)
        if create_schema:
            RAGTrace.__table__.create(self.engine, checkfirst=True)
            RAGSpan.__table__.create(self.engine, checkfirst=True)

    def save(self, snapshot: dict) -> None:
        self.save_many([snapshot])

    def save_many(self, snapshots: list[dict]) -> None:
        """Persist several completed traces in one database transaction."""
        rows: list[tuple[RAGTrace, list[RAGSpan]]] = []
        for snapshot in snapshots:
            rows.append(self._models(snapshot))
        with Session(self.engine) as session, session.begin():
            for trace, spans in rows:
                session.merge(trace)
                for span in spans:
                    session.merge(span)

    @staticmethod
    def _models(snapshot: dict) -> tuple[RAGTrace, list[RAGSpan]]:
        trace_values = {k: snapshot.get(k) for k in (
            "trace_id", "status", "started_at", "ended_at", "duration_ms",
            "input_tokens", "output_tokens", "total_tokens", "error_summary",
        )}
        trace_values["trace_metadata"] = snapshot.get("metadata") or {}
        trace = RAGTrace(**trace_values)
        spans = []
        for item in snapshot.get("spans", []):
            values = {
            k: item.get(k) for k in (
                "span_id", "trace_id", "parent_span_id", "name", "kind", "sequence",
                "status", "started_at", "ended_at", "duration_ms", "input_tokens",
                "output_tokens", "total_tokens", "error_summary",
            )
            }
            values["span_metadata"] = item.get("metadata") or {}
            spans.append(RAGSpan(**values))
        return trace, spans

    def get(self, trace_id: str) -> dict | None:
        with Session(self.engine) as session:
            trace = session.get(RAGTrace, trace_id)
            if not trace:
                return None
            spans = session.scalars(
                select(RAGSpan).where(RAGSpan.trace_id == trace_id).order_by(RAGSpan.sequence)
            ).all()
            return self._trace_dict(trace, spans)

    def list_recent(self, limit: int = 50) -> list[dict]:
        with Session(self.engine) as session:
            rows = session.scalars(
                select(RAGTrace).order_by(RAGTrace.started_at.desc()).limit(min(max(limit, 1), 500))
            ).all()
            return [self._trace_dict(row, []) for row in rows]

    def stats(self, limit: int = 500) -> dict:
        with Session(self.engine) as session:
            traces = session.scalars(
                select(RAGTrace).order_by(RAGTrace.started_at.desc()).limit(min(max(limit, 1), 5000))
            ).all()
            trace_ids = [row.trace_id for row in traces]
            spans = session.scalars(
                select(RAGSpan).where(RAGSpan.trace_id.in_(trace_ids), RAGSpan.kind == "node")
            ).all() if trace_ids else []
        grouped: dict[str, list[RAGSpan]] = defaultdict(list)
        for span in spans:
            grouped[span.name].append(span)
        return {
            "trace_count": len(traces),
            "error_count": sum(row.status == "error" for row in traces),
            # End-to-end latency and total token values come from traces, never
            # from sums of inclusive parent/child span durations.
            "p95_latency_ms": _percentile([row.duration_ms for row in traces], .95),
            "total_tokens": sum(row.total_tokens for row in traces),
            "nodes": [{
                "name": name,
                "count": len(items),
                "avg_duration_ms": round(sum(x.duration_ms for x in items) / len(items), 2),
                "p95_duration_ms": _percentile([x.duration_ms for x in items], .95),
                "error_count": sum(x.status == "error" for x in items),
            } for name, items in sorted(grouped.items())],
        }

    @staticmethod
    def _trace_dict(trace: RAGTrace, spans: list[RAGSpan]) -> dict:
        data = {
            "trace_id": trace.trace_id,
            "status": trace.status,
            "started_at": trace.started_at,
            "ended_at": trace.ended_at,
            "duration_ms": trace.duration_ms,
            "input_tokens": trace.input_tokens,
            "output_tokens": trace.output_tokens,
            "total_tokens": trace.total_tokens,
            "error_summary": trace.error_summary,
            "metadata": trace.trace_metadata,
        }
        data["spans"] = []
        for span in spans:
            item = {
                "span_id": span.span_id,
                "trace_id": span.trace_id,
                "parent_span_id": span.parent_span_id,
                "name": span.name,
                "kind": span.kind,
                "sequence": span.sequence,
                "status": span.status,
                "started_at": span.started_at,
                "ended_at": span.ended_at,
                "duration_ms": span.duration_ms,
                "input_tokens": span.input_tokens,
                "output_tokens": span.output_tokens,
                "total_tokens": span.total_tokens,
                "error_summary": span.error_summary,
                "metadata": span.span_metadata,
            }
            data["spans"].append(item)
        return data
