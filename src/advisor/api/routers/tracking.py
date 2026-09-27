"""Tracking endpoints: entry, risk and close per name, and whether the system is current.

Both read only the local store (and one local ``git`` read for the code
revisions), so the frontend can poll them.
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/api/tracking", tags=["tracking"])


def _db():
    from advisor.research.config import get_settings

    return get_settings().db_path


@router.get("/board")
async def board() -> dict:
    """One row per name held or proposed on lately: the call, the risk, the trades."""
    import asyncio

    from advisor.daemon.market_calendar import now_et
    from advisor.entry.board import load_board

    now = now_et()
    rows = await asyncio.to_thread(load_board, _db(), now)
    return {"asof": now.isoformat(), "rows": [r.model_dump(mode="json") for r in rows]}


@router.get("/status")
async def status() -> dict:
    """Code, jobs, rules and inputs, each against what current means."""
    import asyncio

    from advisor.daemon.uptodate import system_status

    st = await asyncio.to_thread(system_status, _db())
    return st.model_dump(mode="json")
