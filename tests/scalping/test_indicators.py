"""Unit tests for scalping indicators (offline, deterministic)."""

from __future__ import annotations

import pandas as pd
from advisor.scalping.indicators import vwap


def _df(closes: list[float], volumes: list[float], freq: str = "5min") -> pd.DataFrame:
    idx = pd.date_range("2026-06-15 09:30", periods=len(closes), freq=freq)
    close = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.1,
            "low": close - 0.1,
            "close": close,
            "volume": volumes,
        },
        index=idx,
    )


def test_vwap_computes_cleanly_with_zero_volume_bars():
    """Regression test: a zero-volume bar (real when a batch download hits a
    thinly-traded symbol or a pre-market bar) used to crash with
    "float() argument must be a string or a real number, not 'NAType'"
    because the zero-cumvol guard replaced 0 with pd.NA instead of np.nan,
    which .astype(float) can't convert. Reproduced live scanning a real
    S&P 500 universe."""
    df = _df([100.0, 100.5, 101.0, 100.8], [0.0, 0.0, 1000.0, 500.0])
    result = vwap(df)
    assert len(result) == 4
    assert result.dtype == float
    # First two bars have zero cumulative volume -> undefined VWAP -> NaN,
    # not a crash and not a fabricated number.
    assert pd.isna(result.iloc[0])
    assert pd.isna(result.iloc[1])
    assert not pd.isna(result.iloc[2])
    assert not pd.isna(result.iloc[3])


def test_vwap_prefers_feed_provided_column_when_present():
    df = _df([100.0, 101.0], [1000.0, 1000.0])
    df["vwap"] = [99.5, 100.2]
    result = vwap(df)
    assert result.tolist() == [99.5, 100.2]


def test_vwap_computes_session_anchored_average_with_normal_volume():
    df = _df([100.0, 102.0], [1000.0, 1000.0])
    result = vwap(df)
    assert result.iloc[0] == 100.0  # single bar so far -> VWAP == its own typical price
    assert result.iloc[1] == 101.0  # cumulative average of the two bars
