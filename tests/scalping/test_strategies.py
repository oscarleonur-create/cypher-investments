"""Unit tests for the scalp strategy functions (offline, deterministic)."""

from __future__ import annotations

import pandas as pd
from advisor.scalping import strategies as strat
from advisor.scalping.models import ScalpAction
from advisor.scalping.scanner import ScalpScanner


def build_df(closes: list[float], freq: str = "5min", rng: float = 0.1) -> pd.DataFrame:
    """A single-session intraday OHLCV frame from a list of closes."""
    idx = pd.date_range("2026-06-15 09:30", periods=len(closes), freq=freq)
    close = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame(
        {
            "open": close.shift(1).fillna(close.iloc[0]),
            "high": close + rng,
            "low": close - rng,
            "close": close,
            "volume": 1000.0,
        },
        index=idx,
    )


# ── VWAP reversion ──────────────────────────────────────────────────────────


def test_vwap_reversion_long_when_stretched_below():
    df = build_df([100.0] * 29 + [95.0])
    sig = strat.vwap_reversion("TEST", df, strat.default_params("vwap_reversion"))
    assert sig is not None
    assert sig.action == ScalpAction.LONG
    assert sig.target > sig.entry  # target is VWAP, above the stretched-down price
    assert sig.stop < sig.entry


def test_vwap_reversion_short_when_stretched_above():
    df = build_df([100.0] * 29 + [105.0])
    sig = strat.vwap_reversion("TEST", df, strat.default_params("vwap_reversion"))
    assert sig is not None
    assert sig.action == ScalpAction.SHORT
    assert sig.target < sig.entry


def test_vwap_reversion_flat_market_no_signal():
    df = build_df([100.0] * 30)
    assert strat.vwap_reversion("TEST", df, strat.default_params("vwap_reversion")) is None


# ── RSI(2) mean reversion ───────────────────────────────────────────────────


def test_rsi2_long_on_oversold():
    closes = [100.0] * 15 + [101 - i for i in range(15)]  # monotonic decline at the end
    df = build_df(closes)
    sig = strat.rsi2_mean_reversion("TEST", df, strat.default_params("rsi2_mean_reversion"))
    assert sig is not None
    assert sig.action == ScalpAction.LONG
    assert sig.target > sig.entry > sig.stop


def test_rsi2_short_on_overbought():
    # Rising into the close, with an early dip so RSI is defined (no losses → NaN).
    closes = [100.0] * 14 + [99.0] + [100.0 + i for i in range(15)]
    df = build_df(closes)
    sig = strat.rsi2_mean_reversion("TEST", df, strat.default_params("rsi2_mean_reversion"))
    assert sig is not None
    assert sig.action == ScalpAction.SHORT
    assert sig.target < sig.entry < sig.stop


# ── Opening range breakout ──────────────────────────────────────────────────


def test_orb_long_breakout():
    # First 3 bars (15m at 5m freq) sit in [100, 101]; price later breaks above.
    closes = [100.5, 100.8, 100.6] + [101.5, 102.0, 102.5, 103.0]
    df = build_df(closes)
    sig = strat.opening_range_breakout("TEST", df, strat.default_params("opening_range_breakout"))
    assert sig is not None
    assert sig.action == ScalpAction.LONG
    assert sig.stop < sig.entry < sig.target


def test_orb_no_signal_inside_range():
    closes = [100.5, 100.8, 100.6, 100.7, 100.55]
    df = build_df(closes)
    assert (
        strat.opening_range_breakout("TEST", df, strat.default_params("opening_range_breakout"))
        is None
    )


# ── Risk/reward + scanner integration ───────────────────────────────────────


def test_signal_risk_reward():
    df = build_df([100.0] * 29 + [105.0])
    sig = strat.vwap_reversion("TEST", df, strat.default_params("vwap_reversion"))
    assert sig.risk_reward >= 0


def test_scanner_ranks_signals_by_score(monkeypatch):
    frames = {
        "AAA": build_df([100.0] * 29 + [95.0]),
        "BBB": build_df([100.0] * 30),  # flat → no signal
    }
    monkeypatch.setattr(
        "advisor.scalping.scanner.fetch_intraday_candles",
        lambda symbols, **kw: ({s: frames[s] for s in symbols if s in frames}, "yfinance"),
    )
    result = ScalpScanner().scan(["AAA", "BBB"], interval="5m")
    assert result.symbols_scanned == 2
    assert result.source == "yfinance"
    syms = {s.symbol for s in result.signals}
    assert "AAA" in syms and "BBB" not in syms
    scores = [s.score for s in result.signals]
    assert scores == sorted(scores, reverse=True)
