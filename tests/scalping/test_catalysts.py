"""Catalyst layer: volume context (pure) + gate/rank logic (network mocked)."""

from __future__ import annotations

from datetime import datetime

import pandas as pd
from advisor.scalping import catalysts as cat
from advisor.scalping.models import ScalpAction, ScalpSignal


def _vol_df(recent_vol: float, base_vol: float, n: int = 12) -> pd.DataFrame:
    idx = pd.date_range("2026-06-15 09:30", periods=n, freq="5min")
    vols = [base_vol] * (n - 3) + [recent_vol] * 3
    close = pd.Series([100.0] * n, index=idx)
    return pd.DataFrame(
        {"open": close, "high": close + 0.1, "low": close - 0.1, "close": close, "volume": vols},
        index=idx,
    )


def _sig(symbol: str, strategy: str, action: ScalpAction, score: float = 60.0) -> ScalpSignal:
    return ScalpSignal(
        symbol=symbol,
        strategy=strategy,
        action=action,
        reason="x",
        price=100.0,
        entry=100.0,
        stop=99.0,
        target=101.0,
        score=score,
        bar_time=datetime(2026, 6, 15, 10, 0),
    )


# ── Tier 1 (pure) ───────────────────────────────────────────────────────────


def test_rvol_and_gap_from_candles():
    rvol, _ = cat.volume_context(_vol_df(recent_vol=400, base_vol=100))
    assert rvol == 4.0

    # Two-session frame → gap = day-2 open vs day-1 close.
    idx = pd.to_datetime(["2026-06-12 15:55", "2026-06-15 09:30", "2026-06-15 09:35"])
    df = pd.DataFrame(
        {
            "open": [99.0, 103.0, 103.5],
            "high": [100.0, 104.0, 104.0],
            "low": [98.0, 102.0, 103.0],
            "close": [100.0, 103.5, 103.8],
            "volume": [1000, 1000, 1000],
        },
        index=idx,
    )
    _, gap = cat.volume_context(df)
    assert gap == 0.03  # (103 - 100) / 100


# ── Gate on volume ──────────────────────────────────────────────────────────


def test_gate_drops_low_rvol(monkeypatch):
    monkeypatch.setattr(cat, "earnings_context", lambda s: (False, None))
    monkeypatch.setattr(cat, "news_headlines", lambda s, **k: [])

    candles = {
        "HI": _vol_df(recent_vol=400, base_vol=100),  # rvol 4 → kept
        "LO": _vol_df(recent_vol=50, base_vol=100),  # rvol 0.5 → gated
    }
    signals = [
        _sig("HI", "opening_range_breakout", ScalpAction.LONG),
        _sig("LO", "opening_range_breakout", ScalpAction.LONG),
    ]
    kept, gated = cat.enrich_signals(signals, candles, min_rvol=1.5, use_llm=False)

    assert gated == 1
    assert [s.symbol for s in kept] == ["HI"]
    hi = kept[0]
    assert hi.technical_score == 60.0
    assert hi.score > 60.0  # RVOL conviction boost
    assert "RVOL" in hi.catalyst_note


# ── Rank on news (direction-aware) ──────────────────────────────────────────


def test_sentiment_boosts_aligned_penalizes_fade(monkeypatch):
    monkeypatch.setattr(cat, "earnings_context", lambda s: (False, None))
    monkeypatch.setattr(cat, "news_headlines", lambda s, **k: ["headline"])
    # Strongly bullish news.
    monkeypatch.setattr(cat, "llm_sentiment", lambda s: (80.0, True, ["bull"]))

    # Low RVOL so the volume-conviction boost doesn't mask the news effect.
    candles = {
        "MOM": _vol_df(recent_vol=110, base_vol=100),
        "FADE": _vol_df(recent_vol=110, base_vol=100),
    }
    signals = [
        _sig("MOM", "opening_range_breakout", ScalpAction.LONG),  # aligned with bull news
        _sig("FADE", "vwap_reversion", ScalpAction.SHORT),  # fading into bull news
    ]
    kept, _ = cat.enrich_signals(signals, candles, min_rvol=0.0, use_llm=True)
    by_sym = {s.symbol: s for s in kept}

    assert by_sym["MOM"].score > by_sym["MOM"].technical_score  # boosted
    assert by_sym["FADE"].score < by_sym["FADE"].technical_score  # penalized
    # The confirmed momentum trade should rank above the news-fighting fade.
    assert kept[0].symbol == "MOM"
