"""MySQL repository for long-term memory metadata and content."""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache

from sqlalchemy import create_engine, delete, select
from sqlalchemy.orm import Session

from app.config import settings
from app.db.models import Base, MemoryEntry


class LongTermMemoryRepository:
    def __init__(self, database_url: str, *, create_schema: bool = True):
        connect_args = {"check_same_thread": False} if database_url.startswith("sqlite") else {}
        self.engine = create_engine(
            database_url,
            echo=settings.MYSQL_ECHO,
            pool_pre_ping=True,
            connect_args=connect_args,
        )
        if create_schema:
            Base.metadata.create_all(self.engine, tables=[MemoryEntry.__table__])

    def upsert(self, entry: MemoryEntry) -> tuple[MemoryEntry, bool]:
        """Deduplicate by user, memory type and normalized content hash."""
        with Session(self.engine, expire_on_commit=False) as session, session.begin():
            existing = session.scalar(select(MemoryEntry).where(
                MemoryEntry.user_id == entry.user_id,
                MemoryEntry.memory_type == entry.memory_type,
                MemoryEntry.content_hash == entry.content_hash,
            ))
            if existing is None:
                session.add(entry)
                return entry, True
            existing.content = entry.content
            existing.category = entry.category
            existing.source_kind = entry.source_kind
            existing.is_confirmed = existing.is_confirmed or entry.is_confirmed
            existing.confidence = max(existing.confidence, entry.confidence)
            existing.memory_metadata = {
                **(existing.memory_metadata or {}),
                **(entry.memory_metadata or {}),
            }
            existing.is_active = True
            existing.updated_at = datetime.utcnow()
            return existing, False

    def fetch(self, user_id: str, memory_ids: list[str]) -> list[MemoryEntry]:
        if not memory_ids:
            return []
        with Session(self.engine) as session:
            rows = session.scalars(select(MemoryEntry).where(
                MemoryEntry.user_id == user_id,
                MemoryEntry.memory_id.in_(memory_ids),
                MemoryEntry.is_active.is_(True),
            )).all()
            by_id = {row.memory_id: row for row in rows}
            return [by_id[item] for item in memory_ids if item in by_id]

    def mark_indexed(self, memory_id: str, status: str = "indexed") -> None:
        with Session(self.engine) as session, session.begin():
            entry = session.get(MemoryEntry, memory_id)
            if entry is not None:
                entry.index_status = status

    def record_access(self, memory_ids: list[str]) -> None:
        if not memory_ids:
            return
        with Session(self.engine) as session, session.begin():
            rows = session.scalars(select(MemoryEntry).where(
                MemoryEntry.memory_id.in_(memory_ids)
            )).all()
            now = datetime.utcnow()
            for row in rows:
                row.access_count += 1
                row.last_accessed_at = now

    def list_user(self, user_id: str, limit: int = 100) -> list[MemoryEntry]:
        with Session(self.engine) as session:
            return list(session.scalars(
                select(MemoryEntry)
                .where(MemoryEntry.user_id == user_id, MemoryEntry.is_active.is_(True))
                .order_by(MemoryEntry.updated_at.desc())
                .limit(limit)
            ).all())

    def delete(self, user_id: str, memory_id: str) -> bool:
        with Session(self.engine) as session, session.begin():
            result = session.execute(delete(MemoryEntry).where(
                MemoryEntry.user_id == user_id,
                MemoryEntry.memory_id == memory_id,
            ))
            return bool(result.rowcount)


@lru_cache(maxsize=1)
def get_long_term_memory_repository() -> LongTermMemoryRepository:
    return LongTermMemoryRepository(settings.MYSQL_URL)
