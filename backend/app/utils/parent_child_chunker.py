"""Build stable document -> parent -> child relationships for legal data."""

from __future__ import annotations

import hashlib
import re
import uuid
from dataclasses import dataclass

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import settings
from app.context.compactor import estimate_tokens
from app.db.models import ChildChunk, ParentChunk, SourceDocument
from app.utils.legal_chunker import split_legal_document


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _stable_id(namespace: uuid.UUID, value: str) -> str:
    return str(uuid.uuid5(namespace, value))


def _without_context_header(text: str) -> str:
    return re.sub(r"^\[[^\n]+\]\n", "", text, count=1).strip()


def _line_number(text: str, position: int) -> int:
    return text.count("\n", 0, max(0, position)) + 1


@dataclass
class ParentChildTree:
    document: SourceDocument
    vector_documents: list[Document]
    child_ids: list[str]


def build_parent_child_tree(
    text: str,
    *,
    doc_type: str,
    source_file: str,
    source_path: str = "",
    dataset_name: str = "",
    source_record_id: str = "",
) -> ParentChildTree:
    """Create MySQL entities plus child Documents ready for vector indexing."""
    content_hash = _sha256(text)
    identity = f"{doc_type}:{source_file}:{source_record_id}:{content_hash}"
    doc_id = _stable_id(uuid.NAMESPACE_URL, identity)

    parent_documents = split_legal_document(text, doc_type=doc_type, filename=source_file)
    title = ""
    if parent_documents:
        title = str(
            parent_documents[0].metadata.get("law_name")
            or parent_documents[0].metadata.get("case_title")
            or source_file
        )
    source = SourceDocument(
        doc_id=doc_id,
        doc_type=doc_type,
        title=title or source_file,
        source_file=source_file,
        source_path=source_path,
        dataset_name=dataset_name,
        raw_content=text,
        content_hash=content_hash,
        document_metadata={"source_record_id": source_record_id} if source_record_id else {},
        chunker_version=settings.CHUNKER_VERSION,
        status="chunked",
    )

    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHILD_CHUNK_SIZE,
        chunk_overlap=settings.CHILD_CHUNK_OVERLAP,
        separators=["\n\n", "。", "；", "，", "\n", ""],
        keep_separator=True,
    )
    vector_documents: list[Document] = []
    child_ids: list[str] = []
    parent_search_start = 0

    for parent_index, parent_doc in enumerate(parent_documents):
        parent_text = _without_context_header(parent_doc.page_content)
        parent_hash = _sha256(parent_text)
        parent_id = _stable_id(
            uuid.UUID(doc_id),
            f"{settings.CHUNKER_VERSION}:parent:{parent_index}:{parent_hash}",
        )
        found = text.find(parent_text, max(0, parent_search_start - 128))
        parent_start = found if found >= 0 else None
        parent_end = parent_start + len(parent_text) if parent_start is not None else None
        if parent_end is not None:
            parent_search_start = parent_end

        metadata = dict(parent_doc.metadata)
        parent = ParentChunk(
            parent_chunk_id=parent_id,
            doc_id=doc_id,
            parent_index=parent_index,
            parent_type=str(metadata.get("section_type") or ("article" if doc_type == "law" else "case_section")),
            parent_text=parent_text,
            content_hash=parent_hash,
            char_start=parent_start,
            char_end=parent_end,
            line_start=_line_number(text, parent_start) if parent_start is not None else None,
            line_end=_line_number(text, parent_end) if parent_end is not None else None,
            token_count=estimate_tokens(parent_text),
            law_name=str(metadata.get("law_name", "")),
            article_number=str(metadata.get("article_number", "")),
            chapter=str(metadata.get("chapter", "")),
            case_number=str(metadata.get("case_number", "")),
            guiding_number=str(metadata.get("guiding_number", "")),
            section_type=str(metadata.get("section_type", "")),
            chunk_metadata=metadata,
            chunker_version=settings.CHUNKER_VERSION,
        )
        source.parents.append(parent)

        child_texts = child_splitter.split_text(parent_text) or [parent_text]
        child_search_start = 0
        for child_index, child_text in enumerate(child_texts):
            child_text = child_text.strip()
            if not child_text:
                continue
            child_hash = _sha256(child_text)
            child_id = _stable_id(
                uuid.UUID(parent_id),
                f"child:{child_index}:{child_hash}",
            )
            local_start = parent_text.find(child_text, max(0, child_search_start - settings.CHILD_CHUNK_OVERLAP))
            local_start = local_start if local_start >= 0 else None
            local_end = local_start + len(child_text) if local_start is not None else None
            if local_end is not None:
                child_search_start = local_end
            absolute_start = (
                parent_start + local_start
                if parent_start is not None and local_start is not None
                else None
            )
            absolute_end = (
                parent_start + local_end
                if parent_start is not None and local_end is not None
                else None
            )
            child_meta = {
                "child_chunk_id": child_id,
                "parent_chunk_id": parent_id,
                "doc_id": doc_id,
                "doc_type": doc_type,
                "source_file": source_file,
                "law_name": parent.law_name,
                "article_number": parent.article_number,
                "case_number": parent.case_number,
                "guiding_number": parent.guiding_number,
                "section_type": parent.section_type,
            }
            parent.children.append(ChildChunk(
                child_chunk_id=child_id,
                parent_chunk_id=parent_id,
                doc_id=doc_id,
                child_index=child_index,
                child_text=child_text,
                content_hash=child_hash,
                char_start=absolute_start,
                char_end=absolute_end,
                token_count=estimate_tokens(child_text),
                child_metadata=child_meta,
            ))
            header_bits = [
                value for value in (
                    parent.law_name,
                    f"第{parent.article_number}条" if parent.article_number else "",
                    parent.section_type,
                ) if value
            ]
            vector_text = f"[{' / '.join(header_bits)}]\n{child_text}" if header_bits else child_text
            vector_documents.append(Document(page_content=vector_text, metadata=child_meta))
            child_ids.append(child_id)

    return ParentChildTree(source, vector_documents, child_ids)
