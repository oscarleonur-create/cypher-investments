"""Depth on demand for any list of names: the news agent and the deep-research report.

Both are network- and model-bound, so each runs as a background job that goes
through the names one at a time and reports progress; the page polls the job
and then reads the results. ``/status`` is a cheap store read for badges.

Costs, so the page can say them before a bulk run: the news agent pulls one
Tavily search per name before judging; the full report (which carries the
deep research) makes several searches and model calls per name and takes
minutes.
"""

from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from advisor.api import deps

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/depth", tags=["depth"])

MAX_NAMES = 40


class Names(BaseModel):
    symbols: list[str] = Field(default_factory=list)


def _clean(symbols: list[str]) -> list[str]:
    out = []
    for s in symbols:
        sym = s.strip().upper()
        if not sym:
            continue
        if not sym.replace("-", "").replace(".", "").isalnum() or len(sym) > 10:
            raise HTTPException(status_code=400, detail=f"not a symbol: {s!r}")
        if sym not in out:
            out.append(sym)
    if not out:
        raise HTTPException(status_code=400, detail="no symbols")
    if len(out) > MAX_NAMES:
        raise HTTPException(status_code=400, detail=f"at most {MAX_NAMES} names per run")
    return out


def _run_each(job_id: str, symbols: list[str], label: str, work) -> None:
    """Run ``work(symbol)`` for each name in turn; keep going past a failure."""
    results, failed = [], []
    for n, sym in enumerate(symbols, 1):
        deps.update_job(job_id, message=f"{label} {sym} ({n}/{len(symbols)})…")
        try:
            results.append(work(sym))
        except Exception as exc:  # noqa: BLE001
            logger.exception("%s failed for %s", label, sym)
            failed.append(sym)
            results.append({"symbol": sym, "problems": [str(exc)]})
    done = len(symbols) - len(failed)
    note = f"; failed {', '.join(failed)}" if failed else ""
    deps.update_job(
        job_id,
        status="done" if done else "error",
        message=f"{label}: {done}/{len(symbols)} done{note}",
        error=None if done else "every name failed",
        results=results,
    )


@router.post("/news")
async def run_news(body: Names) -> dict:
    """Pull and judge recent news for each name (one Tavily search per name). Poll the job."""
    import asyncio

    symbols = _clean(body.symbols)
    job_id = deps.new_job("news", target=",".join(symbols))

    def work(sym: str) -> dict:
        from advisor.daemon.market_calendar import now_et
        from advisor.news.agent_run import run_news_agent

        return run_news_agent(deps.db_path(), sym, now_et())

    asyncio.create_task(asyncio.to_thread(_run_each, job_id, symbols, "news agent", work))
    return {"job_id": job_id, "names": len(symbols)}


@router.get("/news/{symbol}")
async def news(symbol: str) -> dict:
    """The name's judgments of the last two weeks and its weekly synthesis."""
    import asyncio

    from advisor.daemon.market_calendar import now_et
    from advisor.news.agent_run import news_view

    sym = _clean([symbol])[0]
    return await asyncio.to_thread(news_view, deps.db_path(), sym, now_et())


@router.post("/research")
async def run_research(body: Names) -> dict:
    """Rebuild the full report — deep research included — for each name. Minutes per name."""
    import asyncio

    symbols = _clean(body.symbols)
    job_id = deps.new_job("research", target=",".join(symbols))

    def work(sym: str) -> dict:
        from advisor.research.report import build_report

        build_report(sym, force_refresh=True)
        return {"symbol": sym, "problems": []}

    asyncio.create_task(asyncio.to_thread(_run_each, job_id, symbols, "deep research", work))
    return {"job_id": job_id, "names": len(symbols)}


def status_of(db_path, symbols: list[str]) -> dict:
    """Per name: news judged in the last two weeks, and when its report was last built."""
    import sqlite3
    from datetime import timedelta

    from advisor.daemon.market_calendar import now_et

    since = (now_et() - timedelta(days=14)).isoformat()
    conn = sqlite3.connect(str(db_path))
    out = {}
    try:
        tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
        for sym in symbols:
            row = {"news_judged": 0, "news_material": 0, "news_last": None, "report_at": None,
                   "deep_research": False}  # fmt: skip
            if "news_judgments" in tables:
                n, material, last = conn.execute(
                    "SELECT COUNT(*), SUM(CASE WHEN json_extract(payload_json, '$.about_company') "
                    "AND json_extract(payload_json, '$.materiality') != 'LOW' THEN 1 ELSE 0 END), "
                    "MAX(judged_at) FROM news_judgments WHERE symbol = ? AND published_at >= ?",
                    (sym, since),
                ).fetchone()
                row.update(news_judged=n or 0, news_material=material or 0, news_last=last)
            if "research_reports" in tables:
                r = conn.execute(
                    "SELECT created_at, json_extract(report_json, '$.deep_research') IS NOT NULL "
                    "FROM research_reports WHERE symbol = ? ORDER BY created_at DESC LIMIT 1",
                    (sym,),
                ).fetchone()
                if r:
                    row["report_at"] = r[0].replace(" ", "T") + "Z" if r[0] else None
                    row["deep_research"] = bool(r[1])
            out[sym] = row
    finally:
        conn.close()
    return out


@router.get("/status")
async def status(symbols: str) -> dict:
    """``?symbols=A,B``: what depth each name already has."""
    import asyncio

    syms = _clean(symbols.split(","))
    return {"status": await asyncio.to_thread(status_of, deps.db_path(), syms)}
