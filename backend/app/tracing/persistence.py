"""Best-effort, non-blocking trace persistence."""

from __future__ import annotations

import asyncio
from typing import Callable

from app.tracing.repository import TraceRepository


class TracePersistenceService:
    def __init__(self, repository_factory: Callable[[], TraceRepository] = TraceRepository):
        self.repository_factory = repository_factory
        self._tasks: set[asyncio.Task] = set()
        self._pending: list[dict] = []
        self._worker: asyncio.Task | None = None

    def enqueue(self, snapshot: dict) -> None:
        """Schedule one transaction containing a trace and all of its spans."""
        try:
            loop = asyncio.get_running_loop()
            self._pending.append(snapshot)
            if self._worker is None or self._worker.done():
                self._worker = loop.create_task(self._drain())
                self._tasks.add(self._worker)
                self._worker.add_done_callback(self._tasks.discard)
        except Exception:
            # Observability must never change the answer path.
            return

    async def _drain(self) -> None:
        # Small window coalesces concurrent FastAPI requests without delaying
        # the answer path; every batch is committed atomically.
        await asyncio.sleep(0.05)
        while self._pending:
            batch, self._pending = self._pending, []
            try:
                await asyncio.to_thread(self.repository_factory().save_many, batch)
            except Exception:
                # Drop only telemetry; never retry on the request path.
                continue

    async def flush(self) -> None:
        tasks = list(self._tasks)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        if self._pending:
            await self._drain()


trace_persistence = TracePersistenceService()
