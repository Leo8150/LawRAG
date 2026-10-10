"""Repository for authoritative MySQL parent-child content."""

from __future__ import annotations

from functools import lru_cache

from langchain_core.documents import Document
from sqlalchemy import create_engine, delete, func, select
from sqlalchemy.orm import Session, joinedload

from app.config import settings
from app.db.models import Base, ChildChunk, ParentChunk, SourceDocument


class ParentChildRepository:
    def __init__(self, database_url: str, *, create_schema: bool = False):
        connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
        self.engine = create_engine(
            database_url,
            echo=settings.MYSQL_ECHO,
            pool_pre_ping=True,
            connect_args=connect_args,
        )
        if create_schema:
            self.create_schema()

    def create_schema(self) -> None:
        Base.metadata.create_all(self.engine)

    def save_tree(self, document: SourceDocument) -> None:
        """Idempotently replace one document and its complete parent-child tree."""
        with Session(self.engine, expire_on_commit=False) as session, session.begin():
            existing = session.get(SourceDocument, document.doc_id)
            if existing is not None:
                session.delete(existing)
                session.flush()
            session.add(document)

    def update_status(self, doc_id: str, status: str) -> None:
        with Session(self.engine) as session, session.begin():
            item = session.get(SourceDocument, doc_id)
            if item is not None:
                item.status = status

    def fetch_children(self, child_ids: list[str]) -> list[Document]:
        if not child_ids:
            return []
        with Session(self.engine) as session:
            rows = session.scalars(
                select(ChildChunk)
                .where(ChildChunk.child_chunk_id.in_(child_ids))
                .options(joinedload(ChildChunk.parent).joinedload(ParentChunk.document))
            ).all()
            by_id = {row.child_chunk_id: row for row in rows}
            results: list[Document] = []
            for child_id in child_ids:
                row = by_id.get(child_id)
                if row is None:
                    continue
                parent = row.parent
                source = parent.document
                metadata = {
                    **(parent.chunk_metadata or {}),
                    **(row.child_metadata or {}),
                    "child_chunk_id": row.child_chunk_id,
                    "parent_chunk_id": row.parent_chunk_id,
                    "doc_id": row.doc_id,
                    "chunk_index": row.child_index,
                    "doc_type": source.doc_type,
                    "source_file": source.source_file,
                    "law_name": parent.law_name,
                    "article_number": parent.article_number,
                    "case_number": parent.case_number,
                    "guiding_number": parent.guiding_number,
                    "section_type": parent.section_type,
                }
                results.append(Document(page_content=row.child_text, metadata=metadata))
            return results

    def fetch_parents(self, parent_ids: list[str]) -> list[Document]:
        if not parent_ids:
            return []
        with Session(self.engine) as session:
            rows = session.scalars(
                select(ParentChunk)
                .where(ParentChunk.parent_chunk_id.in_(parent_ids))
                .options(joinedload(ParentChunk.document))
            ).all()
            by_id = {row.parent_chunk_id: row for row in rows}
            results: list[Document] = []
            for parent_id in parent_ids:
                row = by_id.get(parent_id)
                if row is None:
                    continue
                source = row.document
                metadata = {
                    **(row.chunk_metadata or {}),
                    "parent_chunk_id": row.parent_chunk_id,
                    "doc_id": row.doc_id,
                    "doc_type": source.doc_type,
                    "source_file": source.source_file,
                    "source_path": source.source_path,
                    "law_name": row.law_name,
                    "article_number": row.article_number,
                    "case_number": row.case_number,
                    "guiding_number": row.guiding_number,
                    "section_type": row.section_type,
                    "char_start": row.char_start,
                    "char_end": row.char_end,
                }
                results.append(Document(page_content=row.parent_text, metadata=metadata))
            return results

    def list_children(self, doc_types: list[str] | None = None) -> list[Document]:
        with Session(self.engine) as session:
            statement = select(ChildChunk).options(
                joinedload(ChildChunk.parent).joinedload(ParentChunk.document)
            )
            if doc_types:
                statement = statement.join(ChildChunk.parent).join(ParentChunk.document).where(
                    SourceDocument.doc_type.in_(doc_types)
                )
            rows = session.scalars(statement).unique().all()
            return self.fetch_children([row.child_chunk_id for row in rows])

    def get_document(self, doc_id: str) -> SourceDocument | None:
        with Session(self.engine) as session:
            return session.get(SourceDocument, doc_id)

    def get_chunk_source(self, chunk_id: str) -> dict | None:
        """Return a child/parent citation without loading unrelated documents."""
        with Session(self.engine) as session:
            child = session.scalar(
                select(ChildChunk)
                .where(ChildChunk.child_chunk_id == chunk_id)
                .options(joinedload(ChildChunk.parent).joinedload(ParentChunk.document))
            )
            if child is not None:
                parent = child.parent
                source = parent.document
                return {
                    "chunk_level": "child",
                    "child_chunk_id": child.child_chunk_id,
                    "parent_chunk_id": child.parent_chunk_id,
                    "doc_id": child.doc_id,
                    "chunk_text": child.child_text,
                    "parent_text": parent.parent_text,
                    "char_start": child.char_start,
                    "char_end": child.char_end,
                    "title": source.title,
                    "source_file": source.source_file,
                    "source_path": source.source_path,
                }
            parent = session.scalar(
                select(ParentChunk)
                .where(ParentChunk.parent_chunk_id == chunk_id)
                .options(joinedload(ParentChunk.document))
            )
            if parent is None:
                return None
            source = parent.document
            return {
                "chunk_level": "parent",
                "parent_chunk_id": parent.parent_chunk_id,
                "doc_id": parent.doc_id,
                "parent_text": parent.parent_text,
                "char_start": parent.char_start,
                "char_end": parent.char_end,
                "title": source.title,
                "source_file": source.source_file,
                "source_path": source.source_path,
            }

    def get_document_source(self, doc_id: str) -> dict | None:
        with Session(self.engine) as session:
            source = session.get(SourceDocument, doc_id)
            if source is None:
                return None
            return {
                "doc_id": source.doc_id,
                "doc_type": source.doc_type,
                "title": source.title,
                "source_file": source.source_file,
                "source_path": source.source_path,
                "dataset_name": source.dataset_name,
                "raw_content": source.raw_content,
                "metadata": source.document_metadata or {},
                "document_version": source.document_version,
                "chunker_version": source.chunker_version,
            }

    def child_ids_for_document(self, doc_id: str) -> list[str]:
        with Session(self.engine) as session:
            return list(session.scalars(
                select(ChildChunk.child_chunk_id).where(ChildChunk.doc_id == doc_id)
            ).all())

    def delete_document(self, doc_id: str) -> bool:
        with Session(self.engine) as session, session.begin():
            result = session.execute(delete(SourceDocument).where(SourceDocument.doc_id == doc_id))
            return bool(result.rowcount)

    def document_ids_by_source_file(self, source_file: str) -> list[tuple[str, str]]:
        with Session(self.engine) as session:
            return list(session.execute(
                select(SourceDocument.doc_id, SourceDocument.doc_type).where(
                    SourceDocument.source_file == source_file
                )
            ).all())

    def reset_all(self) -> None:
        with Session(self.engine) as session, session.begin():
            session.execute(delete(ChildChunk))
            session.execute(delete(ParentChunk))
            session.execute(delete(SourceDocument))

    def counts(self) -> dict[str, int]:
        with Session(self.engine) as session:
            return {
                "documents": session.scalar(select(func.count(SourceDocument.doc_id))) or 0,
                "parents": session.scalar(select(func.count(ParentChunk.parent_chunk_id))) or 0,
                "children": session.scalar(select(func.count(ChildChunk.child_chunk_id))) or 0,
            }

    def child_counts_by_doc_type(self) -> dict[str, int]:
        with Session(self.engine) as session:
            rows = session.execute(
                select(SourceDocument.doc_type, func.count(ChildChunk.child_chunk_id))
                .join(ParentChunk, ParentChunk.doc_id == SourceDocument.doc_id)
                .join(ChildChunk, ChildChunk.parent_chunk_id == ParentChunk.parent_chunk_id)
                .group_by(SourceDocument.doc_type)
            ).all()
            return {str(doc_type): int(count) for doc_type, count in rows}


@lru_cache(maxsize=1)
def get_parent_child_repository() -> ParentChildRepository:
    return ParentChildRepository(settings.MYSQL_URL)
