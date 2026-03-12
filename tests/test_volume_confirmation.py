"""Tests for the volume confirmation layer."""

from __future__ import annotations

from unittest.mock import patch

import numpy as np
import pandas as pd
from advisor.confluence.volume_confirmation import (
    _compute_obv,
    check_volume_confirmation,
)


def _make_df(closes: list[float], volumes: list[float]) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame."""
    n = len(closes)
    return pd.DataFrame(
        {
            "Open": closes,
            "High": [c * 1.01 for c in closes],
            "Low": [c * 0.99 for c in closes],
            "Close": closes,
            "Volume": volumes,
        },
        index=pd.date_range("2025-01-01", periods=n, freq="B"),
    )


class TestComputeOBV:
    def test_rising_prices(self):
        close = np.array([10.0, 11.0, 12.0, 13.0])
        volume = np.array([100.0, 200.0, 300.0, 400.0])
        obv = _compute_obv(close, volume)
        assert obv[-1] == 900.0  # 0 + 200 + 300 + 400

    def test_falling_prices(self):
        close = np.array([13.0, 12.0, 11.0, 10.0])
        volume = np.array([100.0, 200.0, 300.0, 400.0])
        obv = _compute_obv(close, volume)
        assert obv[-1] == -900.0

    def test_flat_prices(self):
        close = np.array([10.0, 10.0, 10.0])
        volume = np.array([100.0, 200.0, 300.0])
        obv = _compute_obv(close, volume)
        assert obv[-1] == 0.0


class TestCheckVolumeConfirmation:
    @patch("advisor.confluence.volume_confirmation.YahooDataProvider")
    def test_capitulation_detected(self, mock_provider_cls):
        """Volume spike on a down day within last 5 days → capitulation."""
        n = 30
        closes = [100.0 - i * 0.2 for i in range(n)]
        volumes = [1_000_000.0] * n
        # Spike volume on last down day
        volumes[-2] = 3_000_000.0
        closes[-2] = closes[-3] - 2.0  # down day

        mock_provider_cls.return_value.get_stock_history.return_value = _make_df(closes, volumes)

        result = check_volume_confirmation("TEST")
        assert result.capitulation_detected is True
        assert result.capitulation_ratio >= 2.0
        assert result.score > 0

    @patch("advisor.confluence.volume_confirmation.YahooDataProvider")
    def test_volume_dryup(self, mock_provider_cls):
        """3 consecutive days of declining volume → dry-up."""
        n = 30
        closes = [100.0] * n
        volumes = [1_000_000.0] * n
        # Last 3 days: declining volume
        volumes[-3] = 900_000.0
        volumes[-2] = 700_000.0
        volumes[-1] = 500_000.0

        mock_provider_cls.return_value.get_stock_history.return_value = _make_df(closes, volumes)

        result = check_volume_confirmation("TEST")
        assert result.volume_dryup is True
        assert result.score >= 25.0

    @patch("advisor.confluence.volume_confirmation.YahooDataProvider")
    def test_obv_divergence(self, mock_provider_cls):
        """Price down but OBV rising → accumulation divergence."""
        n = 30
        closes = list(np.linspace(110, 100, n))  # price trending down
        volumes = [1_000_000.0] * n
        # Make most recent days have higher volume on up-ticks
        # to create positive OBV slope while price falls
        for i in range(-10, 0):
            if i % 2 == 0:
                closes[i] = closes[i - 1] + 0.1  # small up day
                volumes[i] = 2_000_000.0  # big volume on up days
            else:
                closes[i] = closes[i - 1] - 0.05  # small down day
                volumes[i] = 500_000.0  # low volume on down days
        # Ensure price is still lower than 10 days ago
        closes[-1] = min(closes[-1], closes[-11] - 1.0)

        mock_provider_cls.return_value.get_stock_history.return_value = _make_df(closes, volumes)

        result = check_volume_confirmation("TEST")
        assert result.obv_divergence is True
        assert result.score >= 35.0

    @patch("advisor.confluence.volume_confirmation.YahooDataProvider")
    def test_no_signals(self, mock_provider_cls):
        """Flat volume, flat price → no signals."""
        n = 30
        closes = [100.0] * n
        volumes = [1_000_000.0] * n

        mock_provider_cls.return_value.get_stock_history.return_value = _make_df(closes, volumes)

        result = check_volume_confirmation("TEST")
        assert result.capitulation_detected is False
        assert result.volume_dryup is False
        assert result.obv_divergence is False
        assert result.score == 0.0

    @patch("advisor.confluence.volume_confirmation.YahooDataProvider")
    def test_insufficient_data(self, mock_provider_cls):
        """Too few bars → empty result."""
        mock_provider_cls.return_value.get_stock_history.return_value = _make_df(
            [100.0] * 10, [1_000_000.0] * 10
        )

        result = check_volume_confirmation("TEST")
        assert result.score == 0.0
