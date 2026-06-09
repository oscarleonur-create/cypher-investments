"""Watchlist endpoints: follow arbitrary tickers; adding one builds full research."""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from advisor.api import deps

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api", tags=["watchlist"])


class AddRequest(BaseModel):
    symbol: str
    note: str = ""


def _attention_for(status: str) -> str:
    return {"INVALIDATED": "HIGH", "CAUTION": "MEDIUM"}.get(status, "LOW")


def _watchlist_summary(report) -> dict:
    """A compact research/fair-value summary for one watched ticker."""
    status = "UNKNOWN"
    alerts: list[str] = []
    if report.kpi_monitor:
        status = report.kpi_monitor.thesis_status or "UNKNOWN"
        alerts = report.kpi_monitor.alerts or []

    base_upside = None
    current_price = None
    if report.dcf:
        current_price = report.dcf.current_price
        if report.dcf.base:
            base_upside = report.dcf.base.upside_pct

    fair_price = fair_upside = None
    if report.fair_price:
        fair_price = report.fair_price.fair_price
        fair_upside = report.fair_price.upside_pct
        current_price = report.fair_price.current_price or current_price

    return {
        "has_report": True,
        "thesis_status": status,
        "attention": _attention_for(status),
        "conviction": report.thesis.conviction if report.thesis else None,
        "base_upside": base_upside,
        "current_price": current_price,
        "fair_price": fair_price,
        "fair_upside": fair_upside,
        "kpi_alerts": alerts,
    }


@router.get("/watchlist")
async def list_watchlist() -> dict:
    from advisor.research.store import ResearchStore

    store = ResearchStore(deps.db_path())
    try:
        items = []
        for entry in store.load_watchlist():
            sym = entry["symbol"]
            report = store.load_latest_report(sym)
            items.append(
                {
                    "symbol": sym,
                    "note": entry["note"],
                    "added_at": entry["added_at"],
                    "research": _watchlist_summary(report) if report else None,
                }
            )
        return {"watchlist": items}
    finally:
        store.close()


@router.post("/watchlist")
async def add_watchlist(req: AddRequest) -> dict:
    """Add a ticker and kick off a full research build in the background."""
    import asyncio

    from advisor.research.store import ResearchStore

    sym = (req.symbol or "").strip().upper()
    if not sym:
        raise HTTPException(status_code=400, detail="Empty symbol.")

    store = ResearchStore(deps.db_path())
    try:
        store.add_to_watchlist(sym, req.note)
    finally:
        store.close()

    # Build the full report so the ticker page populates (best-effort per layer).
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
            logger.exception("watchlist build failed for %s", sym)
            deps.update_job(job_id, status="error", error=str(exc), message="failed")

    asyncio.create_task(asyncio.to_thread(_run))
    return {"symbol": sym, "job_id": job_id}


@router.delete("/watchlist/{symbol}")
async def remove_watchlist(symbol: str) -> dict:
    from advisor.research.store import ResearchStore

    store = ResearchStore(deps.db_path())
    try:
        store.remove_from_watchlist(symbol.upper())
    finally:
        store.close()
    return {"symbol": symbol.upper(), "removed": True}
