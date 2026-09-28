"""Breadth endpoints: the stored picks and the evidence behind them.

Reads only the breadth store; the picks are built nightly by ``breadth_sync``
(or ``advisor breadth picks --build``), so the page can poll.
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/api/breadth", tags=["breadth"])


def _db():
    from advisor.research.config import get_settings

    return get_settings().db_path


def _latest() -> dict:
    from advisor.breadth.picks import latest_picks
    from advisor.breadth.store import BreadthStore, breadth_path

    path = breadth_path(_db())
    if not path.exists():
        return {"day": None, "picks": [], "track_record": [], "caveats": []}
    with BreadthStore(path) as store:
        return latest_picks(store)


@router.get("/picks")
async def picks() -> dict:
    """The latest picks, each with its reasons, plus the group's measured record."""
    import asyncio

    return await asyncio.to_thread(_latest)
