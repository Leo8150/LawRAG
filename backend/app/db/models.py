"""SQLAlchemy models for documents and parent-child chunks."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, ForeignKey, Index, Integer, JSON, String, Text, UniqueConstraint
from sqlalchemy.dialects.mysql import LONGTEXT
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


LONG_TEXT = Text().with_variant(LONGTEXT(), "mysql")


class Base(DeclarativeBase):
    pass


class SourceDocument(Base):
    __tablename__ = "documents"

    doc_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    doc_type: Mapped[str] = mapped_column(String(32), index=True)
    title: Mapped[str] = mapped_column(String(512), default="")
    source_file: Mapped[str] = mapped_column(String(512))
    source_path: Mapped[str] = mapped_column(String(1024), default="")
    dataset_name: Mapped[str] = mapped_column(String(128), default="")
    raw_content: Mapped[str] = mapped_column(LONG_TEXT)
    content_hash: Mapped[str] = mapped_column(String(64), unique=True, index=True)
    document_metadata: Mapped[dict] = mapped_column("metadata", JSON, default=dict)
    document_version: Mapped[int] = mapped_column(Integer, default=1)
    chunker_version: Mapped[str] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(32), default="pending", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    parents: Mapped[list["ParentChunk"]] = relationship(
        back_populates="document", cascade="all, delete-orphan"
    )


class ParentChunk(Base):
    __tablename__ = "parent_chunks"
    __table_args__ = (
        UniqueConstraint("doc_id", "parent_index", "chunker_version", name="uk_document_parent"),
        Index("idx_law_article", "law_name", "article_number"),
    )

    parent_chunk_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    doc_id: Mapped[str] = mapped_column(ForeignKey("documents.doc_id", ondelete="CASCADE"), index=True)
    parent_index: Mapped[int] = mapped_column(Integer)
    parent_type: Mapped[str] = mapped_column(String(64), default="")
    parent_text: Mapped[str] = mapped_column(LONG_TEXT)
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    char_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    char_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    line_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    line_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    law_name: Mapped[str] = mapped_column(String(512), default="")
    article_number: Mapped[str] = mapped_column(String(64), default="")
    chapter: Mapped[str] = mapped_column(String(512), default="")
    case_number: Mapped[str] = mapped_column(String(256), default="")
    guiding_number: Mapped[str] = mapped_column(String(128), default="")
    section_type: Mapped[str] = mapped_column(String(64), default="")
    chunk_metadata: Mapped[dict] = mapped_column("metadata", JSON, default=dict)
    chunker_version: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    document: Mapped[SourceDocument] = relationship(back_populates="parents")
    children: Mapped[list["ChildChunk"]] = relationship(
        back_populates="parent", cascade="all, delete-orphan"
    )


class ChildChunk(Base):
    __tablename__ = "child_chunks"
    __table_args__ = (
        UniqueConstraint("parent_chunk_id", "child_index", name="uk_parent_child"),
    )

    child_chunk_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    parent_chunk_id: Mapped[str] = mapped_column(
        ForeignKey("parent_chunks.parent_chunk_id", ondelete="CASCADE"), index=True
    )
    doc_id: Mapped[str] = mapped_column(ForeignKey("documents.doc_id", ondelete="CASCADE"), index=True)
    child_index: Mapped[int] = mapped_column(Integer)
    child_text: Mapped[str] = mapped_column(Text)
    content_hash: Mapped[str] = mapped_column(String(64), index=True)
    char_start: Mapped[int | None] = mapped_column(Integer, nullable=True)
    char_end: Mapped[int | None] = mapped_column(Integer, nullable=True)
    token_count: Mapped[int] = mapped_column(Integer, default=0)
    child_metadata: Mapped[dict] = mapped_column("metadata", JSON, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    parent: Mapped[ParentChunk] = relationship(back_populates="children")
