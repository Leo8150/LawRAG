"""MySQL repository for long-term memory metadata and content."""

from __future__ import annotations

from datetime import datetime
from functools import lru_cache

from sqlalchemy import create_engine, delete, inspect, select, text
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
            self.ensure_lifecycle_schema()

    def ensure_lifecycle_schema(self) -> None:
        """Add lifecycle columns when upgrading an existing memory table."""
        inspector = inspect(self.engine)
        if "memory_entries" not in inspector.get_table_names():
            return
        existing = {column["name"] for column in inspector.get_columns("memory_entries")}
        definitions = {
            "importance": "FLOAT NOT NULL DEFAULT 0.5",
            "half_life_days": "INTEGER NOT NULL DEFAULT 90",
            "forgotten_at": "DATETIME NULL",
        }
        missing = [name for name in definitions if name not in existing]
        if not missing:
            return
        with self.engine.begin() as connection:
            for name in missing:
                connection.execute(text(
                    f"ALTER TABLE memory_entries ADD COLUMN {name} {definitions[name]}"
                ))
            assignments: list[str] = []
            if "importance" in missing:
                assignments.append(
                    "importance = CASE category "
                    "WHEN 'case_fact' THEN :case_importance "
                    "WHEN 'preference' THEN :preference_importance "
                    "ELSE :episodic_importance END"
                )
            if "half_life_days" in missing:
                assignments.append(
                    "half_life_days = CASE category "
                    "WHEN 'case_fact' THEN :case_half_life "
                    "WHEN 'preference' THEN :preference_half_life "
                    "ELSE :episodic_half_life END"
                )
            if assignments:
                connection.execute(
                    text(f"UPDATE memory_entries SET {', '.join(assignments)}"),
                    {
                        "case_importance": settings.MEMORY_CASE_FACT_IMPORTANCE,
                        "preference_importance": settings.MEMORY_PREFERENCE_IMPORTANCE,
                        "episodic_importance": settings.MEMORY_EPISODIC_IMPORTANCE,
                        "case_half_life": settings.MEMORY_CASE_FACT_HALF_LIFE_DAYS,
                        "preference_half_life": settings.MEMORY_PREFERENCE_HALF_LIFE_DAYS,
                        "episodic_half_life": settings.MEMORY_EPISODIC_HALF_LIFE_DAYS,
                    },
                )

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
            existing.importance = entry.importance
            existing.half_life_days = entry.half_life_days
            existing.source_kind = entry.source_kind
            existing.is_confirmed = existing.is_confirmed or entry.is_confirmed
            existing.confidence = max(existing.confidence, entry.confidence)
            existing.memory_metadata = {
                **(existing.memory_metadata or {}),
                **(entry.memory_metadata or {}),
            }
            existing.is_active = True
            existing.forgotten_at = None
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

    def record_access(self, user_id: str, memory_ids: list[str]) -> None:
        if not memory_ids:
            return
        with Session(self.engine) as session, session.begin():
            rows = session.scalars(select(MemoryEntry).where(
                MemoryEntry.user_id == user_id,
                MemoryEntry.memory_id.in_(memory_ids)
            )).all()
            now = datetime.utcnow()
            for row in rows:
                row.access_count += 1
                row.last_accessed_at = now

    def list_active_for_maintenance(self, limit: int = 500) -> list[MemoryEntry]:
        with Session(self.engine) as session:
            return list(session.scalars(
                select(MemoryEntry)
                .where(MemoryEntry.is_active.is_(True))
                .order_by(MemoryEntry.updated_at.asc())
                .limit(limit)
            ).all())

    def soft_forget(self, memory_ids: list[str], forgotten_at: datetime) -> int:
        if not memory_ids:
            return 0
        with Session(self.engine) as session, session.begin():
            rows = session.scalars(select(MemoryEntry).where(
                MemoryEntry.memory_id.in_(memory_ids),
                MemoryEntry.is_active.is_(True),
            )).all()
            for row in rows:
                row.is_active = False
                row.forgotten_at = forgotten_at
            return len(rows)

    def purge_forgotten(self, before: datetime, limit: int = 500) -> int:
        with Session(self.engine) as session, session.begin():
            memory_ids = list(session.scalars(
                select(MemoryEntry.memory_id)
                .where(
                    MemoryEntry.is_active.is_(False),
                    MemoryEntry.forgotten_at.is_not(None),
                    MemoryEntry.forgotten_at < before,
                )
                .limit(limit)
            ).all())
            if not memory_ids:
                return 0
            result = session.execute(delete(MemoryEntry).where(
                MemoryEntry.memory_id.in_(memory_ids)
            ))
            return int(result.rowcount or 0)

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
