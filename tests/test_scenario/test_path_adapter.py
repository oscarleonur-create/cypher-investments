"""Tests for scenario path adapter."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from advisor.scenario.models import BUILTIN_SCENARIOS, ScenarioConfig
from advisor.scenario.path_adapter import (
    build_full_feeds,
    synthesize_ohlcv,
)


class TestSynthesizeOHLCV:
    """Tests for OHLCV synthesis from close-only paths."""

    def test_shape_and_columns(self):
        """Synthesized DataFrames have correct shape and OHLCV columns."""
        n_paths, n_days = 5, 20
        rng = np.random.default_rng(42)
        # Simple random walk
        close_paths = 100 * np.exp(np.cumsum(0.01 * rng.standard_normal((n_paths, n_days)), axis=1))

        result = synthesize_ohlcv(close_paths, start_date=date(2025, 1, 6), seed=42)

        assert len(result) == n_paths
        for df in result:
            assert list(df.columns) == ["Open", "High", "Low", "Close", "Volume"]
            assert len(df) == n_days

    def test_ohlcv_validity(self):
        """High >= max(Open, Close) and Low <= min(Open, Close)."""
        n_paths, n_days = 10, 30
        rng = np.random.default_rng(123)
        close_paths = 100 * np.exp(np.cumsum(0.01 * rng.standard_normal((n_paths, n_days)), axis=1))

        result = synthesize_ohlcv(close_paths, start_date=date(2025, 1, 6), seed=123)

        for df in result:
            assert (df["High"] >= df["Close"]).all(), "High must be >= Close"
            assert (df["High"] >= df["Open"]).all(), "High must be >= Open"
            assert (df["Low"] <= df["Close"]).all(), "Low must be <= Close"
            assert (df["Low"] <= df["Open"]).all(), "Low must be <= Open"

    def test_volume_positive(self):
        """Volume is always positive."""
        n_paths, n_days = 3, 15
        rng = np.random.default_rng(7)
        close_paths = 100 * np.exp(np.cumsum(0.01 * rng.standard_normal((n_paths, n_days)), axis=1))

        result = synthesize_ohlcv(close_paths, start_date=date(2025, 1, 6), seed=7)

        for df in result:
            assert (df["Volume"] > 0).all()

    def test_open_equals_prev_close(self):
        """Open[t] == Close[t-1] for t > 0."""
        rng = np.random.default_rng(99)
        close_paths = 100 * np.exp(np.cumsum(0.01 * rng.standard_normal((2, 10)), axis=1))

        result = synthesize_ohlcv(close_paths, start_date=date(2025, 1, 6), seed=99)

        for df in result:
            np.testing.assert_array_almost_equal(df["Open"].values[1:], df["Close"].values[:-1])

    def test_reproducibility_with_seed(self):
        """Same seed produces identical output."""
        rng = np.random.default_rng(42)
        close_paths = 100 * np.exp(np.cumsum(0.01 * rng.standard_normal((3, 10)), axis=1))

        result1 = synthesize_ohlcv(close_paths, start_date=date(2025, 1, 6), seed=42)
        result2 = synthesize_ohlcv(close_paths, start_date=date(2025, 1, 6), seed=42)

        for df1, df2 in zip(result1, result2):
            pd.testing.assert_frame_equal(df1, df2)


class TestBuildFullFeeds:
    """Tests for prepending warmup data to simulated feeds."""

    def test_warmup_prepended(self):
        """Warmup data is prepended before simulated data."""
        warmup_dates = pd.bdate_range("2024-06-01", periods=200, freq="B")
        warmup_df = pd.DataFrame(
            {
                "Open": 100.0,
                "High": 101.0,
                "Low": 99.0,
                "Close": 100.5,
                "Volume": 1_000_000,
            },
            index=warmup_dates,
        )

        sim_dates = pd.bdate_range("2025-03-10", periods=30, freq="B")
        sim_df = pd.DataFrame(
            {
                "Open": 150.0,
                "High": 151.0,
                "Low": 149.0,
                "Close": 150.5,
                "Volume": 2_000_000,
            },
            index=sim_dates,
        )

        result = build_full_feeds(warmup_df, [sim_df])
        assert len(result) == 1
        combined = result[0]
        assert len(combined) == len(warmup_df) + len(sim_df)
        assert combined.index[0] == warmup_dates[0]
        assert combined.index[-1] == sim_dates[-1]

    def test_no_overlap(self):
        """Warmup data trimmed if overlapping with simulation dates."""
        overlap_dates = pd.bdate_range("2025-03-01", periods=20, freq="B")
        warmup_df = pd.DataFrame(
            {"Open": 100.0, "High": 101.0, "Low": 99.0, "Close": 100.5, "Volume": 1e6},
            index=overlap_dates,
        )

        sim_dates = pd.bdate_range("2025-03-10", periods=10, freq="B")
        sim_df = pd.DataFrame(
            {"Open": 150.0, "High": 151.0, "Low": 149.0, "Close": 150.5, "Volume": 2e6},
            index=sim_dates,
        )

        result = build_full_feeds(warmup_df, [sim_df])
        combined = result[0]
        # No duplicate dates
        assert combined.index.is_unique


class TestGenerateScenarioPaths:
    """Tests for MC path generation with scenario overrides."""

    @patch("advisor.scenario.path_adapter.calibrate")
    @patch("advisor.scenario.path_adapter.yf")
    def test_drift_bias(self, mock_yf, mock_calibrate):
        """Bull paths should have higher mean terminal price than bear paths."""
        from advisor.simulator.models import SimConfig

        mock_calibrate.return_value = SimConfig(vol_mean_level=0.25, seed=42)
        # Mock yfinance
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame(
            {"Close": [150.0]}, index=pd.DatetimeIndex(["2025-03-07"])
        )
        mock_yf.Ticker.return_value = mock_ticker

        from advisor.scenario.path_adapter import generate_scenario_paths

        config = ScenarioConfig(dte=30, n_paths=200, seed=42)

        bull = BUILTIN_SCENARIOS["bull"]
        bear = BUILTIN_SCENARIOS["bear"]

        bull_paths = generate_scenario_paths("AAPL", bull, config, SimConfig(seed=42))
        bear_paths = generate_scenario_paths("AAPL", bear, config, SimConfig(seed=42))

        bull_mean_terminal = bull_paths[:, -1].mean()
        bear_mean_terminal = bear_paths[:, -1].mean()

        assert bull_mean_terminal > bear_mean_terminal, (
            f"Bull terminal ({bull_mean_terminal:.2f}) should exceed "
            f"bear terminal ({bear_mean_terminal:.2f})"
        )

    @patch("advisor.scenario.path_adapter.calibrate")
    @patch("advisor.scenario.path_adapter.yf")
    def test_path_shape(self, mock_yf, mock_calibrate):
        """Output shape is (n_paths, dte+1)."""
        from advisor.simulator.models import SimConfig

        mock_calibrate.return_value = SimConfig(seed=42)
        mock_ticker = MagicMock()
        mock_ticker.history.return_value = pd.DataFrame(
            {"Close": [100.0]}, index=pd.DatetimeIndex(["2025-03-07"])
        )
        mock_yf.Ticker.return_value = mock_ticker

        from advisor.scenario.path_adapter import generate_scenario_paths

        config = ScenarioConfig(dte=20, n_paths=50, seed=42)
        paths = generate_scenario_paths(
            "TEST", BUILTIN_SCENARIOS["sideways"], config, SimConfig(seed=42)
        )

        assert paths.shape == (50, 21)  # n_paths x (dte + 1)
