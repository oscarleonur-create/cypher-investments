"""Actionables: what to do now on the book and on what is tracked, and the answers to them.

Reads only the local stores (the book as the daemon last saved it), so the
frontend can poll it.
"""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter(prefix="/api/actionables", tags=["actionables"])


def _db():
    from advisor.research.config import get_settings

    return get_settings().db_path


@router.get("")
async def actionables() -> dict:
    """SELL, TRIM, DECIDE, READ and BUY, in that order; and what has been answered."""
    import asyncio

    from advisor.daemon.market_calendar import now_et
    from advisor.entry.actionables import load

    return await asyncio.to_thread(load, _db(), now_et())


class AnswerInput(BaseModel):
    id: str
    answer: str  # KEEP | DONE | SKIP
    note: str = ""


def _answer(body: AnswerInput) -> dict:
    from advisor.daemon.market_calendar import now_et
    from advisor.daemon.store import DaemonStore
    from advisor.entry.actionables import Actionable, decision_for, load

    # The item is rebuilt here rather than taken from the request: what was
    # answered is the system's situation, not something a caller asserts.
    current = {a["id"]: a for a in load(_db(), now_et())["items"]}
    item = current.get(body.id)
    if item is None:
        raise HTTPException(404, f"no open actionable {body.id!r}")
    try:
        decisions = decision_for(Actionable.model_validate(item), body.answer, body.note)
    except ValueError as exc:
        raise HTTPException(400, str(exc)) from None
    store = DaemonStore(_db())
    try:
        for d in decisions:
            store.record_decision(d)
    finally:
        store.close()
    return {"id": body.id, "decisions": [d.model_dump(mode="json") for d in decisions]}


@router.post("/answer")
async def answer(body: AnswerInput) -> dict:
    """KEEP (with your reason), DONE or SKIP an open actionable."""
    import asyncio

    return await asyncio.to_thread(_answer, body)
