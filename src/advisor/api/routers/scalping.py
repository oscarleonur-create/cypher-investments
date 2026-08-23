"""Scalping endpoints: intraday equity scanner + strategy previews.

Runs the pure-pandas scalp engine (`advisor.scalping`) over a universe, fed by
the cached TastyTrade candle stream with a yfinance fallback. Scans run as
background jobs; the latest result is cached in-memory (single-user local app).
"""

from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel

from advisor.api import deps
from advisor.risk.gate import gate_signals
from advisor.scalping.models import ScalpScanResult
from advisor.scalping.scanner import ScalpScanner
from advisor.scalping.strategies import SCALP_STRATEGIES, default_params

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/api/scalping", tags=["scalping"])

# Cap how many symbols a single scan will subscribe to (TastyTrade candle
# streaming is per-symbol; large universes are slow and noisy for scalping).
_MAX_SYMBOLS = 60

# Latest completed scan, keyed by job id, plus a pointer to the most recent.
_results: dict[str, ScalpScanResult] = {}
_latest_job: str | None = None


class RunRequest(BaseModel):
    universe: str = "semiconductors"  # "semiconductors" | "sp500" | "custom"
    symbols: list[str] = []  # used when universe == "custom"
    interval: str = "5m"
    strategies: list[str] = []  # empty → all
    min_rvol: float = 1.5  # RVOL gate: drop triggers below this relative volume
    enrich_llm: bool = False  # also LLM-score news sentiment on survivors (slower)


def _resolve_symbols(req: RunRequest) -> list[str]:
    return deps.resolve_symbols(req.universe, req.symbols, _MAX_SYMBOLS)


@router.get("/strategies")
async def list_strategies() -> dict:
    return {
        "strategies": [
            {
                "name": name,
                "label": info["label"],
                "description": info["description"],
                "defaults": info["defaults"],
            }
            for name, info in SCALP_STRATEGIES.items()
        ]
    }


@router.post("/run")
async def run_scan(req: RunRequest) -> dict:
    symbols = _resolve_symbols(req)
    if not symbols:
        raise HTTPException(status_code=400, detail="No symbols to scan.")

    bad = [s for s in req.strategies if s not in SCALP_STRATEGIES]
    if bad:
        raise HTTPException(status_code=400, detail=f"Unknown strategies: {', '.join(bad)}")

    session = await _try_session()
    net_liq, open_notional = await _account_risk_state(session)
    job_id = deps.new_job("scalp", target=req.universe)
    deps.update_job(job_id, message=f"scanning {len(symbols)} symbols…")

    def _run() -> None:
        try:
            result = ScalpScanner().scan(
                symbols,
                interval=req.interval,
                strategy_names=req.strategies or None,
                session=session,
                catalysts=True,
                min_rvol=req.min_rvol,
                use_llm=req.enrich_llm,
            )
            result.signals = gate_signals(
                result.signals, net_liq=net_liq, open_notional_by_symbol=open_notional
            )
            global _latest_job
            _results[job_id] = result
            _latest_job = job_id
            approved = sum(1 for s in result.signals if s.risk_approved)
            msg = f"{len(result.signals)} signals, {approved} risk-approved ({result.source})"
            deps.update_job(job_id, status="done", message=msg)
        except Exception as exc:  # noqa: BLE001
            logger.exception("scalp scan failed")
            deps.update_job(job_id, status="error", error=str(exc), message="failed")

    asyncio.create_task(asyncio.to_thread(_run))
    return {"job_id": job_id, "symbols": len(symbols)}


@router.get("/signals")
async def latest_signals(job_id: str | None = Query(default=None)) -> dict:
    key = job_id or _latest_job
    result = _results.get(key) if key else None
    if result is None:
        return {"result": None}
    return {"result": result.model_dump(mode="json")}


@router.get("/preview/{symbol}")
async def preview(
    symbol: str,
    interval: str = Query(default="5m"),
    strategy: str = Query(default="vwap_reversion"),
) -> dict:
    """Candles + the signal a single strategy produces — for the detail chart."""
    if strategy not in SCALP_STRATEGIES:
        raise HTTPException(status_code=404, detail=f"Unknown strategy: {strategy}")

    from advisor.scalping import strategies as strat
    from advisor.scalping.data import candles_to_records, fetch_intraday_candles
    from advisor.scalping.scanner import _LOOKBACK_MIN_BY_INTERVAL

    sym = symbol.strip().upper()
    session = await _try_session()
    lookback = _LOOKBACK_MIN_BY_INTERVAL.get(interval, 2880)

    def _work() -> dict:
        candles, source = fetch_intraday_candles(
            [sym], interval=interval, lookback_minutes=lookback, session=session
        )
        df = candles.get(sym)
        if df is None or df.empty:
            return {
                "symbol": sym,
                "interval": interval,
                "source": source,
                "candles": [],
                "signal": None,
            }
        sig = strat.get_strategy(strategy)(sym, df, default_params(strategy))
        if sig is not None:
            from advisor.scalping.catalysts import enrich_signals

            enriched, _ = enrich_signals([sig], {sym: df}, min_rvol=0.0, use_llm=True)
            sig = enriched[0] if enriched else sig
        return {
            "symbol": sym,
            "interval": interval,
            "source": source,
            "candles": candles_to_records(df),
            "signal": sig.model_dump(mode="json") if sig else None,
        }

    return await asyncio.to_thread(_work)


async def _try_session():
    """Best-effort cached TastyTrade session; None if credentials are absent."""
    try:
        return await deps.get_tt_session()
    except Exception as exc:  # noqa: BLE001
        logger.warning("TastyTrade session unavailable, scalp scan will use yfinance: %s", exc)
        return None


async def _account_risk_state(session) -> tuple[float, dict[str, float]]:
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
