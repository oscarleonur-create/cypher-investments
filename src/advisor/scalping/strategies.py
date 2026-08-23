"""Scalp strategies — pure functions over an intraday candle DataFrame.

Each strategy has signature ``fn(symbol, df, params) -> ScalpSignal | None`` and
returns ``None`` when there's no actionable setup on the latest bar. Register new
setups in ``SCALP_STRATEGIES``.
"""

from __future__ import annotations

from typing import Callable

import pandas as pd

from advisor.scalping import indicators
from advisor.scalping.models import ScalpAction, ScalpSignal

StrategyFn = Callable[[str, pd.DataFrame, dict], "ScalpSignal | None"]


def _clamp_score(v: float) -> float:
    return round(max(0.0, min(100.0, v)), 1)


def _last_valid(series: pd.Series) -> float | None:
    s = series.dropna()
    return float(s.iloc[-1]) if not s.empty else None


# ── VWAP reversion / bounce ─────────────────────────────────────────────────


def vwap_reversion(symbol: str, df: pd.DataFrame, params: dict) -> ScalpSignal | None:
    """Fade price back toward VWAP when it's stretched by ≥ k·ATR."""
    if len(df) < 20:
        return None
    stretch = float(params.get("stretch_atr", 1.5))
    atr_period = int(params.get("atr_period", 14))

    vwap = _last_valid(indicators.vwap(df))
    atr = _last_valid(indicators.atr(df, period=atr_period))
    if vwap is None or atr is None or atr <= 0:
        return None

    close = float(df["close"].iloc[-1])
    dist_atr = (close - vwap) / atr
    if abs(dist_atr) < stretch:
        return None

    if dist_atr <= -stretch:  # stretched below VWAP → mean-revert up
        action, stop = ScalpAction.LONG, close - atr
    else:  # stretched above VWAP → mean-revert down
        action, stop = ScalpAction.SHORT, close + atr

    score = _clamp_score(50 + (abs(dist_atr) - stretch) * 25)
    return ScalpSignal(
        symbol=symbol,
        strategy="vwap_reversion",
        action=action,
        reason=f"Price {dist_atr:+.1f} ATR from VWAP ({vwap:.2f}); fade toward VWAP",
        price=close,
        entry=close,
        stop=round(stop, 4),
        target=round(vwap, 4),
        score=score,
        bar_time=df.index[-1].to_pydatetime(),
    )


# ── RSI(2) mean reversion ───────────────────────────────────────────────────


def rsi2_mean_reversion(symbol: str, df: pd.DataFrame, params: dict) -> ScalpSignal | None:
    """Snap-back trade when fast RSI hits an extreme."""
    rsi_period = int(params.get("rsi_period", 2))
    oversold = float(params.get("oversold", 10))
    overbought = float(params.get("overbought", 90))
    atr_period = int(params.get("atr_period", 14))
    rr = float(params.get("target_rr", 1.5))

    if len(df) < max(atr_period, rsi_period) + 5:
        return None

    rsi = _last_valid(indicators.rsi(df["close"], period=rsi_period))
    atr = _last_valid(indicators.atr(df, period=atr_period))
    if rsi is None or atr is None or atr <= 0:
        return None

    close = float(df["close"].iloc[-1])
    if rsi <= oversold:
        action = ScalpAction.LONG
        stop, target = close - atr, close + rr * atr
        score = _clamp_score(50 + (oversold - rsi) * 4)
    elif rsi >= overbought:
        action = ScalpAction.SHORT
        stop, target = close + atr, close - rr * atr
        score = _clamp_score(50 + (rsi - overbought) * 4)
    else:
        return None

    return ScalpSignal(
        symbol=symbol,
        strategy="rsi2_mean_reversion",
        action=action,
        reason=f"RSI({rsi_period})={rsi:.0f} extreme; mean-reversion {action.value.lower()}",
        price=close,
        entry=close,
        stop=round(stop, 4),
        target=round(target, 4),
        score=score,
        bar_time=df.index[-1].to_pydatetime(),
    )


# ── Opening Range Breakout ──────────────────────────────────────────────────


def opening_range_breakout(symbol: str, df: pd.DataFrame, params: dict) -> ScalpSignal | None:
    """Break of the first-N-minute high/low of the latest session."""
    or_minutes = int(params.get("or_minutes", 15))

    orng = indicators.opening_range(df, or_minutes)
    if orng is None:
        return None
    or_high, or_low = orng
    span = or_high - or_low
    if span <= 0:
        return None

    close = float(df["close"].iloc[-1])
    # Only fire once the opening-range window itself is closed.
    last_day = pd.Timestamp(df.index[-1]).normalize()
    session = df[pd.Index(df.index).normalize() == last_day]
    if df.index[-1] < session.index[0] + pd.Timedelta(minutes=or_minutes):
        return None

    if close > or_high:
        action, stop, target = ScalpAction.LONG, or_low, close + span
        score = _clamp_score(50 + ((close - or_high) / span) * 100)
    elif close < or_low:
        action, stop, target = ScalpAction.SHORT, or_high, close - span
        score = _clamp_score(50 + ((or_low - close) / span) * 100)
    else:
        return None

    return ScalpSignal(
        symbol=symbol,
        strategy="opening_range_breakout",
        action=action,
        reason=f"Broke {or_minutes}m opening range [{or_low:.2f}, {or_high:.2f}]",
        price=close,
        entry=close,
        stop=round(stop, 4),
        target=round(target, 4),
        score=score,
        bar_time=df.index[-1].to_pydatetime(),
    )


# ── Registry ────────────────────────────────────────────────────────────────

SCALP_STRATEGIES: dict[str, dict] = {
    "vwap_reversion": {
        "label": "VWAP Reversion",
        "description": "Fade price back toward intraday VWAP when stretched ≥ k·ATR away.",
        "fn": vwap_reversion,
        "defaults": {"stretch_atr": 1.5, "atr_period": 14},
    },
    "rsi2_mean_reversion": {
        "label": "RSI(2) Mean Reversion",
        "description": "Snap-back when the fast RSI(2) reaches an oversold/overbought extreme.",
        "fn": rsi2_mean_reversion,
        "defaults": {
            "rsi_period": 2,
            "oversold": 10,
            "overbought": 90,
            "atr_period": 14,
            "target_rr": 1.5,
        },
    },
    "opening_range_breakout": {
        "label": "Opening Range Breakout",
        "description": "Trade a break of the first-N-minute high/low of the session.",
        "fn": opening_range_breakout,
        "defaults": {"or_minutes": 15},
    },
}


def get_strategy(name: str) -> StrategyFn:
    info = SCALP_STRATEGIES.get(name)
    if info is None:
        raise KeyError(f"Unknown scalp strategy: {name}")
    return info["fn"]


def default_params(name: str) -> dict:
    return dict(SCALP_STRATEGIES[name]["defaults"])
