"""Market data snapshot — short interest, float, volume.

Data source: yfinance (free, no API key).  All fields are best-effort;
None is returned for any field yfinance doesn't report.
"""

from __future__ import annotations

import logging

from advisor.research.models import MarketDataSnapshot

logger = logging.getLogger(__name__)


def get_market_data(symbol: str) -> MarketDataSnapshot:
    """Return short-interest + float + volume snapshot from yfinance."""
    import yfinance as yf

    snap = MarketDataSnapshot(symbol=symbol.upper())

    try:
        ticker = yf.Ticker(symbol)
        info = ticker.info or {}

        snap.float_shares = _f(info, "floatShares")
        snap.shares_outstanding = _f(info, "sharesOutstanding")
        snap.avg_volume_10d = _f(info, "averageVolume10days") or _f(info, "averageDailyVolume10Day")
        snap.avg_volume_90d = _f(info, "averageVolume")

        short_pct = _f(info, "shortPercentOfFloat")
        if short_pct is not None:
            snap.short_interest_pct = short_pct * 100  # convert 0-1 → percentage

        short_ratio = _f(info, "shortRatio")
        snap.days_to_cover = short_ratio  # yfinance shortRatio == days-to-cover

    except Exception as exc:  # noqa: BLE001
        logger.warning("Market data fetch failed for %s: %s", symbol, exc)

    return snap


def _f(info: dict, key: str) -> float | None:
    v = info.get(key)
    if v is None:
        return None
    try:
        f = float(v)
        return None if f != f else f  # NaN guard
    except (TypeError, ValueError):
        return None
