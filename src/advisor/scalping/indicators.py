"""Intraday indicators for scalping strategies.

Thin wrappers that reuse the project's indicator implementations
(`advisor.ml.features`) and add intraday-specific helpers (session VWAP,
opening range).
"""

from __future__ import annotations

import pandas as pd

from advisor.ml.features import _compute_atr, _compute_rsi


def rsi(close: pd.Series, period: int = 2) -> pd.Series:
    """RSI over a close series (defaults to the fast RSI(2) used for scalps)."""
    return _compute_rsi(close, period=period)


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range from an OHLC DataFrame."""
    return _compute_atr(df["high"], df["low"], df["close"], period=period)


def vwap(df: pd.DataFrame) -> pd.Series:
    """Session-anchored VWAP.

    Prefers a feed-provided ``vwap`` column (TastyTrade supplies one per candle);
    otherwise computes a cumulative VWAP that resets each trading session.
    """
    if "vwap" in df.columns and df["vwap"].notna().any():
        return df["vwap"].astype(float)

    typical = (df["high"] + df["low"] + df["close"]) / 3.0
    vol = df["volume"].fillna(0.0)
    session = pd.Index(df.index).normalize()  # group by calendar day
    pv = (typical * vol).groupby(session).cumsum()
    cum_vol = vol.groupby(session).cumsum().replace(0, pd.NA)
    return (pv / cum_vol).astype(float)


def opening_range(df: pd.DataFrame, minutes: int) -> tuple[float, float] | None:
    """High/low of the first ``minutes`` of the most recent session.

    Returns ``(or_high, or_low)`` or ``None`` if the latest session hasn't yet
    produced a full opening-range window.
    """
    if df.empty:
        return None

    last_day = pd.Timestamp(df.index[-1]).normalize()
    session = df[pd.Index(df.index).normalize() == last_day]
    if session.empty:
        return None

    window_end = session.index[0] + pd.Timedelta(minutes=minutes)
    window = session[session.index < window_end]
    if window.empty:
        return None

    return float(window["high"].max()), float(window["low"].min())
