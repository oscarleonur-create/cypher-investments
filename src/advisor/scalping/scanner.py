"""Scalp scanner — run scalp strategies across a universe of symbols."""

from __future__ import annotations

import logging

from advisor.scalping import strategies as strat
from advisor.scalping.data import fetch_intraday_candles
from advisor.scalping.models import ScalpScanResult, ScalpSignal

logger = logging.getLogger(__name__)

# How far back to pull candles per interval (minutes). Enough bars to warm up
# indicators and cover the opening range.
_LOOKBACK_MIN_BY_INTERVAL = {"1m": 1440, "5m": 2880, "15m": 7200, "30m": 14400, "1h": 28800}


class ScalpScanner:
    """Scan symbols against scalp strategies to produce ranked signals."""

    def scan(
        self,
        symbols: list[str],
        interval: str = "5m",
        strategy_names: list[str] | None = None,
        params: dict[str, dict] | None = None,
        session=None,
        catalysts: bool = False,
        min_rvol: float = 1.5,
        use_llm: bool = False,
    ) -> ScalpScanResult:
        names = strategy_names or list(strat.SCALP_STRATEGIES.keys())
        params = params or {}
        lookback = _LOOKBACK_MIN_BY_INTERVAL.get(interval, 2880)

        candles, source = fetch_intraday_candles(
            symbols, interval=interval, lookback_minutes=lookback, session=session
        )

        signals: list[ScalpSignal] = []
        errors: list[str] = []
        for sym in symbols:
            df = candles.get(sym)
            if df is None or df.empty:
                errors.append(f"{sym}: no candle data")
                continue
            for name in names:
                try:
                    fn = strat.get_strategy(name)
                    sig = fn(sym, df, {**strat.default_params(name), **params.get(name, {})})
                    if sig is not None:
                        signals.append(sig)
                except Exception as exc:  # noqa: BLE001 — one bad symbol shouldn't sink the scan
                    logger.warning("scalp %s/%s failed: %s", sym, name, exc)
                    errors.append(f"{sym}/{name}: {exc}")

        gated_out = 0
        if catalysts:
            from advisor.scalping.catalysts import enrich_signals

            signals, gated_out = enrich_signals(
                signals, candles, min_rvol=min_rvol, use_llm=use_llm
            )
        else:
            signals.sort(key=lambda s: s.score, reverse=True)

        return ScalpScanResult(
            interval=interval,
            symbols_scanned=len(symbols),
            source=source,
            min_rvol=min_rvol if catalysts else 0.0,
            gated_out=gated_out,
            signals=signals,
            errors=errors,
        )
