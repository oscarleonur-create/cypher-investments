"""Research endpoints: cached full report per symbol + background rebuilds + jobs."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException

from advisor.api import deps

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["research"])


@router.get("/research")
async def list_reports(limit: int = 100) -> dict:
    from advisor.research.store import ResearchStore

    store = ResearchStore(deps.db_path())
    try:
        return {"reports": store.list_reports(limit=limit)}
    finally:
        store.close()


@router.get("/research/{symbol}")
async def get_report(symbol: str) -> dict:
    from advisor.research.store import ResearchStore

    store = ResearchStore(deps.db_path())
    try:
        report = store.load_latest_report(symbol.upper())
    finally:
        store.close()

    if report is None:
        raise HTTPException(
            status_code=404,
            detail=f"No cached research for {symbol.upper()}. Rebuild it from the ticker page.",
        )
    return report.model_dump(mode="json")


@router.get("/research/{symbol}/price-history")
async def get_price_history(symbol: str) -> dict:
    """Daily price series + fundamentals overlay for the ticker chart.

    Cached as a research artifact and refetched from yfinance only when the
    cached copy is older than 12h, so range-tab switching (done client-side)
    and revisits are instant.
    """
    import asyncio
    from datetime import datetime, timedelta

    from advisor.research.models import PriceHistoryResult
    from advisor.research.price_history import build_price_history
    from advisor.research.store import ResearchStore

    sym = symbol.upper()
    store = ResearchStore(deps.db_path())
    try:
        cached = store.load_artifact(sym, "price_history", "daily")
        if cached is not None:
            payload_json, fetched_at = cached
            if datetime.now() - fetched_at < timedelta(hours=12):
                result = PriceHistoryResult.model_validate_json(payload_json)
                return {
                    "result": result.model_dump(mode="json"),
                    "fetched_at": fetched_at.isoformat(),
                }

        report = store.load_latest_report(sym)
        if report is None:
            raise HTTPException(
                status_code=404,
                detail=f"No cached research for {sym}. Rebuild it from the ticker page.",
            )
        # yfinance download is blocking I/O — keep the event loop free.
        result = await asyncio.to_thread(build_price_history, sym, report)
        store.save_artifact(sym, "price_history", "daily", result.model_dump_json())
        return {
            "result": result.model_dump(mode="json"),
            "fetched_at": result.fetched_at.isoformat(),
        }
    finally:
        store.close()


@router.post("/research/{symbol}/refresh")
async def refresh_report(symbol: str) -> dict:
    """Rebuild the full 7-layer report in the background; poll the returned job id."""
    import asyncio

    sym = symbol.upper()
    job_id = deps.new_job("research", target=sym)

    def _progress(event: str, label: str) -> None:
        if event == "started":
            deps.update_job(job_id, message=f"building {label}…")

    def _run() -> None:
        try:
            from advisor.research.report import build_report

            build_report(sym, force_refresh=True, progress=_progress)
            deps.update_job(job_id, status="done", message="report ready")
        except Exception as exc:  # noqa: BLE001
            logger.exception("research rebuild failed for %s", sym)
            deps.update_job(job_id, status="error", error=str(exc), message="failed")

    asyncio.create_task(asyncio.to_thread(_run))
    return {"job_id": job_id}


@router.get("/jobs/{job_id}")
async def job_status(job_id: str) -> dict:
    job = deps.get_job(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail="Unknown job id")
    return job
