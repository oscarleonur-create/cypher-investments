"""Breadth endpoints: the picks, their history, and evaluating one as a position.

Reading is instant (the store); building picks on live prices and evaluating
a position are network-bound, so they run as background jobs to poll.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from advisor.api import deps

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/breadth", tags=["breadth"])


def _db():
    from advisor.research.config import get_settings

    return get_settings().db_path


def _latest(day: str | None = None) -> dict:
    from advisor.breadth.picks import latest_picks
    from advisor.breadth.store import BreadthStore, breadth_path

    path = breadth_path(_db())
    if not path.exists():
        return {"day": None, "days": [], "picks": [], "track_record": [], "caveats": []}
    with BreadthStore(path) as store:
        return latest_picks(store, day)


@router.get("/picks")
async def picks(day: str | None = None) -> dict:
    """The picks of ``day`` (default: the newest), the days on file, and the evidence."""
    import asyncio

    return await asyncio.to_thread(_latest, day)


def _refresh() -> dict:
    from advisor.breadth.picks import refresh_picks
    from advisor.breadth.store import BreadthStore, breadth_path
    from advisor.daemon.market_calendar import now_et

    with BreadthStore(breadth_path(_db())) as store:
        return refresh_picks(store, now_et(), _db())


@router.post("/picks/refresh")
async def refresh() -> dict:
    """Rebuild the picks now: on live prices while the session runs. Poll the job."""
    import asyncio

    job_id = deps.new_job("picks", target="today")

    def _run() -> None:
        try:
            out = _refresh()
            if out.get("ok"):
                deps.update_job(
                    job_id,
                    status="done",
                    message=f"{len(out['picks'])} picks for {out['day']}"
                    + (" (provisional)" if out.get("provisional") else ""),
                )
            else:
                deps.update_job(job_id, status="error", error=out.get("error"), message="failed")
        except Exception as exc:  # noqa: BLE001
            logger.exception("picks refresh failed")
            deps.update_job(job_id, status="error", error=str(exc), message="failed")

    asyncio.create_task(asyncio.to_thread(_run))
    return {"job_id": job_id}


@router.post("/evaluate/{symbol}")
async def evaluate(symbol: str) -> dict:
    """Run the entry engine for one name (sizing, stop, zone, reading). Poll the job."""
    import asyncio

    sym = symbol.strip().upper()
    if not sym.replace("-", "").replace(".", "").isalnum():
        raise HTTPException(status_code=400, detail="not a symbol")
    job_id = deps.new_job("evaluate", target=sym)

    def _run() -> None:
        from advisor.breadth.position import evaluate_position
        from advisor.daemon.market_calendar import now_et

        try:
            deps.update_job(job_id, message="building the sheet and reading…")
            out = evaluate_position(_db(), sym, now_et())
            if out["proposal"] is None:
                deps.update_job(job_id, status="error", error="; ".join(out["errors"]),
                                message="no proposal")  # fmt: skip
            else:
                deps.update_job(
                    job_id,
                    status="done",
                    message=f"{sym}: {out['proposal']['action']}",
                    errors=out["errors"],
                )
        except Exception as exc:  # noqa: BLE001
            logger.exception("evaluation failed for %s", sym)
            deps.update_job(job_id, status="error", error=str(exc), message="failed")

    asyncio.create_task(asyncio.to_thread(_run))
    return {"job_id": job_id}


@router.get("/evaluation/{symbol}")
async def evaluation(symbol: str) -> dict:
    """The newest proposal on file for ``symbol`` (null when none)."""
    import asyncio

    from advisor.breadth.position import latest_evaluation

    return {"proposal": await asyncio.to_thread(latest_evaluation, _db(), symbol)}
