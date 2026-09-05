"""Shared backend helpers: TastyTrade session, research store, and a job registry.

Single-user, local app — no auth, no DB connection pool. The session is created
lazily and reused; the research store is opened per call (SQLite + WAL).
"""

from __future__ import annotations

import asyncio
import logging
import threading
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

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


async def try_get_tt_session():
    """Best-effort cached TastyTrade session; None if credentials are absent."""
    try:
        return await get_tt_session()
    except Exception as exc:  # noqa: BLE001
        logger.warning("TastyTrade session unavailable: %s", exc)
        return None


async def account_risk_state(session) -> tuple[float, dict[str, float]]:
    """Net liq + open notional per underlying, for sizing the risk gate.

    Best-effort: an unavailable session (or API hiccup) just means every
    signal gets gated with net_liq=0, i.e. blocked rather than sized blind.
    """
    if session is None:
        return 0.0, {}
    try:
        from advisor.market.tastytrade_client import get_balances, get_positions

        balances, positions = await asyncio.gather(get_balances(session), get_positions(session))
        open_notional: dict[str, float] = {}
        for p in positions:
            price = p["mark"] or p["close_price"] or p["average_open_price"]
            notional = abs(p["quantity"]) * price * p["multiplier"]
            open_notional[p["underlying_symbol"]] = (
                open_notional.get(p["underlying_symbol"], 0.0) + notional
            )
        return balances["net_liq"], open_notional
    except Exception as exc:  # noqa: BLE001
        logger.warning("risk gate: could not load account state: %s", exc)
        return 0.0, {}


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


# ── Symbol universe resolution (shared by scalping/swing scan requests) ─────────


def resolve_symbols(universe: str, symbols: list[str], cap: int) -> list[str]:
    """Custom symbols or a named universe, de-duped and order-preserved, capped."""
    if universe == "custom":
        syms = [s.strip().upper() for s in symbols if s.strip()]
    else:
        from advisor.data.universe import fetch_universe

        syms = [s.symbol for s in fetch_universe(universe)]
    seen: set[str] = set()
    out: list[str] = []
    for s in syms:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out[:cap]
