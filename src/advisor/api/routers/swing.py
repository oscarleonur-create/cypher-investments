"""Swing trade confluence scanner endpoints.

Runs the full confluence pipeline (technical + sentiment + fundamental + ML)
across a symbol universe in parallel threads, returning ENTER/CAUTION/PASS
verdicts with reasoning. Scans run as background jobs; results are cached
in-memory.
"""

from __future__ import annotations

import asyncio
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from advisor.api import deps
from advisor.confluence.orchestrator import run_confluence

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/swing", tags=["swing"])

_MAX_SYMBOLS = 50

_SWING_STRATEGIES: dict[str, dict] = {
    "momentum_breakout": {
        "label": "Momentum Breakout",
        "description": (
            "Price closing above 20-day SMA with volume expansion. Best in trending markets."
        ),
        "typical_hold_days": "3–7 days",
    },
    "buy_the_dip": {
        "label": "Buy the Dip",
        "description": (
            "Oversold pullback in an uptrend with dip-screener confirmation "
            "(safety + value trap layers)."
        ),
        "typical_hold_days": "5–14 days",
    },
    "pead": {
        "label": "Post-Earnings Drift",
        "description": (
            "Stocks that gapped and faded after a strong earnings beat — capture the late drift."
        ),
        "typical_hold_days": "30–45 days",
    },
    "mean_reversion": {
        "label": "Mean Reversion",
        "description": "Oversold RSI in range-bound equities reverting to the mean.",
        "typical_hold_days": "3–5 days",
    },
    "sma_crossover": {
        "label": "SMA Crossover",
        "description": "Golden cross (SMA-20 crossing above SMA-50) with trend confirmation.",
        "typical_hold_days": "5–10 days",
    },
}

_results: dict[str, dict] = {}
_latest_job: str | None = None


class ScanRequest(BaseModel):
    universe: str = "semiconductors"
    symbols: list[str] = []
    strategy: str = "momentum_breakout"
    max_symbols: int = 30
    include_ml: bool = True


def _resolve_symbols(req: ScanRequest) -> list[str]:
    cap = min(req.max_symbols, _MAX_SYMBOLS)
    return deps.resolve_symbols(req.universe, req.symbols, cap)


def _scan_symbol(sym: str, strategy: str, include_ml: bool) -> dict | None:
    try:
        # force_all=True: run sentiment + fundamental even without a technical
        # breakout so the scanner surfaces setup stocks, not just confirmed entries.
        r = run_confluence(sym, strategy_name=strategy, force_all=True, include_ml=include_ml)
        ml = r.ml_signal
        return {
            "symbol": sym,
            "strategy": strategy,
            "verdict": r.verdict.value,
            "reasoning": r.reasoning,
            "suggested_hold_days": r.suggested_hold_days,
            "technical_signal": r.technical.signal,
            "technical_price": r.technical.price,
            "technical_bullish": r.technical.is_bullish,
            "volume_ratio": r.technical.volume_ratio,
            "sentiment_bullish": r.sentiment.is_bullish,
            "sentiment_pct": r.sentiment.positive_pct,
            "sentiment_headlines": r.sentiment.key_headlines[:3],
            "fundamental_clear": r.fundamental.is_clear,
            "earnings_within_7_days": r.fundamental.earnings_within_7_days,
            "earnings_date": (
                r.fundamental.earnings_date.isoformat() if r.fundamental.earnings_date else None
            ),
            "insider_buying": r.fundamental.insider_buying_detected,
            "ml_available": ml is not None and ml.is_available,
            "ml_signal": ml.signal if ml else "NEUTRAL",
            "ml_win_prob": ml.win_probability if (ml and ml.is_available) else None,
            "scanned_at": r.scanned_at.isoformat(),
        }
    except Exception as exc:
        logger.warning("Swing scan failed for %s: %s", sym, exc)
        return {"_error": sym, "_msg": str(exc)}


def _run_scan(job_id: str, symbols: list[str], strategy: str, include_ml: bool) -> None:
    signals: list[dict] = []
    errors: list[str] = []
    completed = 0

    with ThreadPoolExecutor(max_workers=8) as pool:
        futures = {pool.submit(_scan_symbol, sym, strategy, include_ml): sym for sym in symbols}
        for fut in as_completed(futures):
            completed += 1
            deps.update_job(job_id, message=f"scanned {completed}/{len(symbols)}…")
            result = fut.result()
            if result is None:
                errors.append(futures[fut])
            elif "_error" in result:
                errors.append(f"{result['_error']}: {result['_msg']}")
            else:
                signals.append(result)

    _order = {"ENTER": 0, "CAUTION": 1, "PASS": 2}
    signals.sort(key=lambda s: (_order.get(s["verdict"], 3), -s["suggested_hold_days"]))

    enter_n = sum(1 for s in signals if s["verdict"] == "ENTER")
    caution_n = sum(1 for s in signals if s["verdict"] == "CAUTION")

    global _latest_job
    _results[job_id] = {
        "scanned_at": datetime.now().isoformat(),
        "strategy": strategy,
        "symbols_scanned": len(symbols),
        "signals": signals,
        "errors": errors,
    }
    _latest_job = job_id
    deps.update_job(
        job_id,
        status="done",
        message=f"{enter_n} ENTER · {caution_n} CAUTION · {len(symbols)} scanned",
    )


@router.get("/strategies")
async def list_strategies() -> dict:
    return {"strategies": [{"name": k, **v} for k, v in _SWING_STRATEGIES.items()]}


@router.post("/scan")
async def run_scan(req: ScanRequest) -> dict:
    symbols = _resolve_symbols(req)
    if not symbols:
        raise HTTPException(status_code=400, detail="No symbols to scan.")
    if req.strategy not in _SWING_STRATEGIES:
        raise HTTPException(status_code=400, detail=f"Unknown strategy: {req.strategy}")

    job_id = deps.new_job("swing", target=req.universe)
    deps.update_job(job_id, message=f"queued {len(symbols)} symbols via {req.strategy}…")

    def _run() -> None:
        try:
            _run_scan(job_id, symbols, req.strategy, req.include_ml)
        except Exception as exc:
            logger.exception("swing scan failed")
            deps.update_job(job_id, status="error", error=str(exc), message="failed")

    asyncio.create_task(asyncio.to_thread(_run))
    return {"job_id": job_id, "symbols": len(symbols)}


@router.get("/signals")
async def latest_signals(job_id: str | None = Query(default=None)) -> dict:
    key = job_id or _latest_job
    result = _results.get(key) if key else None
    return {"result": result}
