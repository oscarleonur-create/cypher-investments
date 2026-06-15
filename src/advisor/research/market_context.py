"""Broad-market context — VIX snapshot + (optional) HMM volatility regime.

The VIX snapshot is always available (a direct ``^VIX`` download). The regime
read reuses the trained HMM in :mod:`advisor.ml.regime` when a model exists on
disk, and is simply omitted otherwise — no auto-training here, so the dashboard
panel is fast and never blocks on a 5-year fit.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

logger = logging.getLogger(__name__)

_REGIME_LABEL = {
    "low_vol": "Calm",
    "normal": "Normal",
    "high_vol": "Stressed",
}


def get_vix_snapshot() -> dict | None:
    """Current VIX level, 1y percentile, 20d SMA and history.

    Returns ``None`` if the download fails. ``history`` is a list of
    ``{"date": iso, "vix": float}`` for charting.
    """
    try:
        import yfinance as yf

        end = date.today() + timedelta(days=1)
        start = end - timedelta(days=400)
        df = yf.download("^VIX", start=str(start), end=str(end), progress=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning("VIX download failed: %s", exc)
        return None

    if df is None or df.empty:
        return None

    if hasattr(df.columns, "get_level_values"):  # flatten MultiIndex
        df.columns = df.columns.get_level_values(0)

    close = df["Close"].dropna()
    if close.empty:
        return None

    current = float(close.iloc[-1])
    sma20 = float(close.tail(20).mean())
    # 1y percentile rank of the current level.
    last_year = close.tail(252)
    percentile = float((last_year <= current).mean()) if len(last_year) else 0.0

    history = [
        {"date": idx.date().isoformat(), "vix": float(val)} for idx, val in close.tail(252).items()
    ]

    return {
        "current": round(current, 2),
        "sma20": round(sma20, 2),
        "percentile_1y": round(percentile, 3),
        "history": history,
    }


def get_regime() -> dict | None:
    """Latest HMM regime read, or ``None`` if no trained model is on disk."""
    try:
        from advisor.ml.regime import RegimeDetector

        if not RegimeDetector.model_exists():
            return None
        info = RegimeDetector.load().detect_regime()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Regime detection failed: %s", exc)
        return None

    info["label"] = _REGIME_LABEL.get(info.get("regime_name", ""), info.get("regime_name", "—"))
    return info


def get_market_context() -> dict:
    """Combined VIX snapshot + optional regime, for the dashboard panel."""
    return {"vix": get_vix_snapshot(), "regime": get_regime()}
