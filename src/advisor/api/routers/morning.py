"""The morning read: one page, each holding, what changed since the close, with sources.

Reads only the local stores, so the page can be opened (and polled) any time.
"""

from __future__ import annotations

from fastapi import APIRouter

router = APIRouter(prefix="/api/morning", tags=["morning"])


def _db():
    from advisor.research.config import get_settings

    return get_settings().db_path


@router.get("")
async def morning() -> dict:
    """What to do, the book, and per holding: the call, what is new, the week, what is next."""
    import asyncio

    from advisor.daemon.market_calendar import now_et
    from advisor.story.morning import load

    return await asyncio.to_thread(load, _db(), now_et())
