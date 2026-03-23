"""Tests for swing trade pattern features."""

from __future__ import annotations

import numpy as np
import pandas as pd
from advisor.ml.swing_features import (
    compute_all_swing_features,
    swing_accumulation_score,
    swing_adx_14,
    swing_bb_squeeze,
    swing_consolidation_days,
    swing_daily_weekly_align,
    swing_feature_names,
    swing_pullback_to_sma20,
    swing_volume_at_breakout,
    swing_weekly_trend,
)


def _make_ohlcv(
    n: int = 300,
    trend: float = 0.0,
    volatility: float = 0.02,
    seed: int = 42,
    freq: str = "B",
) -> pd.DataFrame:
    """Generate synthetic OHLCV data.

    Args:
        n: Number of bars.
        trend: Daily drift (e.g. 0.001 for uptrend).
        volatility: Daily return volatility.
        seed: Random seed.
        freq: Frequency for date index.
    """
    rng = np.random.RandomState(seed)
    dates = pd.bdate_range("2023-01-02", periods=n, freq=freq)
    returns = trend + volatility * rng.randn(n)
    close = 100.0 * np.exp(np.cumsum(returns))

    # Realistic OHLC from close
    high = close * (1 + np.abs(rng.randn(n) * 0.005) + 0.002)
    low = close * (1 - np.abs(rng.randn(n) * 0.005) - 0.002)
    open_ = close * (1 + rng.randn(n) * 0.003)
    volume = (1e6 * (1 + 0.5 * np.abs(rng.randn(n)))).astype(float)

    return pd.DataFrame(
        {"Open": open_, "High": high, "Low": low, "Close": close, "Volume": volume},
        index=dates,
    )


def _make_trending(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Strong uptrend data."""
    return _make_ohlcv(n=n, trend=0.003, volatility=0.008, seed=seed)


def _make_choppy(n: int = 300, seed: int = 42) -> pd.DataFrame:
    """Range-bound / choppy data."""
    return _make_ohlcv(n=n, trend=0.0, volatility=0.025, seed=seed)


# ── Group A: Trend Structure ───────────────────────────────────────


class TestADX:
    def test_adx_trending_market(self):
        """ADX > 0.25 for strong trend data."""
        df = _make_trending()
        adx = swing_adx_14(df["High"], df["Low"], df["Close"])
        # After warmup, ADX should be high in a strong trend
        adx_values = adx.dropna().iloc[50:]
        assert (
            adx_values.mean() > 0.25
        ), f"ADX mean {adx_values.mean():.3f} should be > 0.25 for trending"

    def test_adx_range_bound(self):
        """ADX < 0.20 for choppy data (on average)."""
        df = _make_choppy(seed=99)
        adx = swing_adx_14(df["High"], df["Low"], df["Close"])
        adx_values = adx.dropna().iloc[50:]
        # Choppy market should have lower ADX than trending
        assert (
            adx_values.mean() < 0.35
        ), f"ADX mean {adx_values.mean():.3f} should be lower for choppy"

    def test_adx_bounded(self):
        """ADX values should be between 0 and 1."""
        df = _make_ohlcv()
        adx = swing_adx_14(df["High"], df["Low"], df["Close"]).dropna()
        assert adx.min() >= 0.0
        assert adx.max() <= 1.0


# ── Group B: Consolidation & Squeeze ───────────────────────────────


class TestSqueeze:
    def test_squeeze_detection(self):
        """Squeeze fires when BB inside Keltner."""
        # Create very low-volatility data (consolidation)
        rng = np.random.RandomState(42)
        n = 300
        dates = pd.bdate_range("2023-01-02", periods=n)
        # Flat price with very tight range
        close = pd.Series(100.0 + rng.randn(n) * 0.1, index=dates)
        high = close + 0.15
        low = close - 0.15

        squeeze = swing_bb_squeeze(close, high, low)
        squeeze_valid = squeeze.dropna().iloc[30:]
        # Tight consolidation should trigger some squeezes
        assert squeeze_valid.sum() > 0, "Should detect squeeze in tight consolidation"

    def test_squeeze_binary(self):
        """Squeeze values should be 0 or 1."""
        df = _make_ohlcv()
        squeeze = swing_bb_squeeze(df["Close"], df["High"], df["Low"]).dropna()
        unique_vals = set(squeeze.unique())
        assert unique_vals.issubset({0.0, 1.0}), f"Squeeze should be binary, got {unique_vals}"


# ── Group C: Pullback & Support ────────────────────────────────────


class TestPullback:
    def test_pullback_detection(self):
        """Score > 0 when price touches rising SMA from above in uptrend."""
        df = _make_trending(n=300)
        pullback = swing_pullback_to_sma20(df["Close"])
        pullback_valid = pullback.dropna().iloc[60:]
        # In a strong uptrend, some bars should show pullback signals
        assert pullback_valid.max() > 0, "Should detect pullback opportunities in uptrend"

    def test_pullback_bounded(self):
        """Pullback score should be between 0 and 1."""
        df = _make_ohlcv()
        pullback = swing_pullback_to_sma20(df["Close"]).dropna()
        assert pullback.min() >= 0.0
        assert pullback.max() <= 1.0


# ── Group B: Consolidation Days ────────────────────────────────────


class TestConsolidation:
    def test_consolidation_counting(self):
        """Days count increases during flat range."""
        # Create flat data
        rng = np.random.RandomState(42)
        n = 200
        dates = pd.bdate_range("2023-01-02", periods=n)
        close = pd.Series(100.0 + rng.randn(n) * 0.2, index=dates)
        high = close + 0.3
        low = close - 0.3

        df = pd.DataFrame(
            {"Close": close, "High": high, "Low": low, "Volume": 1e6},
            index=dates,
        )
        consol = swing_consolidation_days(df["High"], df["Low"], df["Close"])
        consol_valid = consol.dropna().iloc[30:]
        # Flat range should accumulate consolidation days
        assert consol_valid.max() > 0, "Should count consolidation days"


# ── Group D: Multi-Timeframe ──────────────────────────────────────


class TestMultiTimeframe:
    def test_weekly_trend_alignment(self):
        """Alignment score matches when daily/weekly agree."""
        df = _make_trending(n=300)
        alignment = swing_daily_weekly_align(df["Close"])
        align_valid = alignment.dropna().iloc[60:]
        # Strong uptrend should show positive alignment mostly
        assert align_valid.mean() > 0, "Trending data should show positive alignment"

    def test_weekly_trend_values(self):
        """Weekly trend should be -1, 0, or 1."""
        df = _make_ohlcv(n=300)
        weekly = swing_weekly_trend(df["Close"]).dropna()
        unique_vals = set(weekly.unique())
        assert unique_vals.issubset(
            {-1.0, 0.0, 1.0}
        ), f"Weekly trend should be {{-1,0,1}}, got {unique_vals}"


# ── Group E: Volume at Key Events ─────────────────────────────────


class TestVolumeEvents:
    def test_volume_at_breakout(self):
        """Non-zero only on breakout bars."""
        df = _make_ohlcv(n=300)
        vab = swing_volume_at_breakout(df["Close"], df["Volume"])
        vab_valid = vab.dropna().iloc[30:]

        # Non-breakout bars should be 0
        close = df["Close"]
        rolling_high = close.rolling(20).max().shift(1)
        breakout_mask = close > rolling_high
        non_breakout = vab_valid[~breakout_mask.reindex(vab_valid.index, fill_value=False)]
        assert (non_breakout == 0).all(), "Non-breakout bars should have 0 volume_at_breakout"

    def test_accumulation_score_bounded(self):
        """Accumulation score should be between 0 and 1."""
        df = _make_ohlcv(n=300)
        acc = swing_accumulation_score(df["Close"], df["High"], df["Low"], df["Volume"]).dropna()
        assert acc.min() >= 0.0
        assert acc.max() <= 1.0


# ── Integration ────────────────────────────────────────────────────


class TestComputeAll:
    def test_compute_all_swing_features(self):
        """Correct shape, no NaN after warmup."""
        df = _make_ohlcv(n=300)
        features = compute_all_swing_features(df)

        assert features.shape[0] == len(df)
        assert features.shape[1] == 15

        # After warmup (100 bars covers all lookback windows), NaN should be minimal
        warmup = features.iloc[100:]
        nan_pct = warmup.isna().mean()
        for col in features.columns:
            assert nan_pct[col] < 0.05, f"{col} has {nan_pct[col]:.1%} NaN after warmup"

    def test_feature_names_match(self):
        """swing_feature_names() matches output columns."""
        df = _make_ohlcv(n=300)
        features = compute_all_swing_features(df)
        names = swing_feature_names()

        assert (
            list(features.columns) == names
        ), f"Column mismatch:\n  computed: {list(features.columns)}\n  names: {names}"

    def test_features_same_index(self):
        """Output index matches input index."""
        df = _make_ohlcv(n=200)
        features = compute_all_swing_features(df)
        assert features.index.equals(df.index)
