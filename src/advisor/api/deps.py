"""Shared backend helpers: TastyTrade session, research store, and a job registry.

Single-user, local app — no auth, no DB connection pool. The session is created
lazily and reused; the research store is opened per call (SQLite + WAL).
"""

from __future__ import annotations

import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

# ── Research store ──────────────────────────────────────────────────────────────


def db_path() -> Path:
    from advisor.research.config import get_settings

    return get_settings().db_path


# ── TastyTrade session (lazy, reused) ───────────────────────────────────────────

_session: Any = None
_session_lock = threading.Lock()


async def get_tt_session():
    """Return a cached authenticated TastyTrade session, creating it on first use."""
    global _session
    if _session is None:
        from advisor.market.tastytrade_client import get_session

        with _session_lock:
            if _session is None:
                _session = await get_session()
    return _session


# ── Job registry (background refreshes) ─────────────────────────────────────────

_jobs: dict[str, dict] = {}
_jobs_lock = threading.Lock()


def new_job(kind: str, target: str = "") -> str:
    job_id = uuid.uuid4().hex[:12]
    with _jobs_lock:
        _jobs[job_id] = {
            "id": job_id,
            "kind": kind,
            "target": target,
            "status": "running",
            "message": "starting…",
            "error": None,
            "started_at": datetime.now().isoformat(),
            "finished_at": None,
        }
    return job_id


def update_job(job_id: str, **fields) -> None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        if job is not None:
            job.update(fields)


def get_job(job_id: str) -> dict | None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        return dict(job) if job else None
