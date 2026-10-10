"""Source traceability endpoints backed by MySQL."""

from fastapi import APIRouter, HTTPException

from app.db.repository import get_parent_child_repository
from app.models.schemas import APIResponse

router = APIRouter(prefix="/sources", tags=["来源追溯"])


@router.get("/chunks/{chunk_id}", response_model=APIResponse)
async def get_chunk_source(chunk_id: str):
    result = get_parent_child_repository().get_chunk_source(chunk_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Chunk 不存在")
    return APIResponse(data=result)


@router.get("/documents/{doc_id}", response_model=APIResponse)
async def get_document_source(doc_id: str):
    result = get_parent_child_repository().get_document_source(doc_id)
    if result is None:
        raise HTTPException(status_code=404, detail="原始文档不存在")
    return APIResponse(data=result)
