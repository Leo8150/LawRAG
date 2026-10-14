"""Read-only endpoints for lightweight RAG observability."""

from fastapi import APIRouter, HTTPException, Query

from app.models.schemas import APIResponse
from app.tracing.repository import TraceRepository

router = APIRouter(prefix="/traces", tags=["traces"])


@router.get("", response_model=APIResponse)
def list_traces(limit: int = Query(50, ge=1, le=500)):
    return APIResponse(data=TraceRepository().list_recent(limit))


@router.get("/stats/summary", response_model=APIResponse)
def trace_stats(limit: int = Query(500, ge=1, le=5000)):
    return APIResponse(data=TraceRepository().stats(limit))


@router.get("/{trace_id}", response_model=APIResponse)
def get_trace(trace_id: str):
    item = TraceRepository().get(trace_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Trace not found")
    return APIResponse(data=item)
