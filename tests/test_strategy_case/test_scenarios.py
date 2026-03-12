"""Tests for scenario detection — Stage 1."""

from datetime import date
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from advisor.strategy_case.models import ScenarioType
from advisor.strategy_case.scenarios import detect_scenario


def _make_hist(prices: list[float], days: int = 60) -> pd.DataFrame:
    """Create a mock history DataFrame."""
    idx = pd.date_range(end=date.today(), periods=days, freq="B")
    if len(prices) < days:
        prices = [prices[0]] * (days - len(prices)) + prices
    return pd.DataFrame({"Close": prices[-days:]}, index=idx[-days:])


class TestScenarioDetection:
    @patch("advisor.strategy_case.scenarios.yf.Ticker")
    def test_iv_spike_detected(self, mock_ticker_cls):
        """High IV percentile should trigger IV_SPIKE."""
        prices = [100.0] * 60
        hist = _make_hist(prices)
        mock_ticker_cls.return_value.history.return_value = hist

        mock_iv_result = MagicMock()
        mock_iv_result.iv_percentile = 80.0

        with (
            patch(
                "advisor.market.iv_analysis.compute_iv_percentile",
                return_value=mock_iv_result,
            ),
            patch(
                "advisor.confluence.pead_screener.check_pead_fundamental",
                side_effect=Exception("skip"),
            ),
            patch(
                "advisor.confluence.dip_screener.check_dip_fundamental",
                side_effect=Exception("skip"),
            ),
        ):
            result = detect_scenario("AAPL")

        assert result.scenario_type == ScenarioType.IV_SPIKE
        assert result.iv_percentile == 80.0

    @patch("advisor.strategy_case.scenarios.yf.Ticker")
    def test_range_bound_default(self, mock_ticker_cls):
        """Flat price action should default to RANGE_BOUND."""
        prices = [100.0 + np.sin(i / 5) * 2 for i in range(60)]
        hist = _make_hist(prices)
        mock_ticker_cls.return_value.history.return_value = hist

        mock_iv_result = MagicMock()
        mock_iv_result.iv_percentile = 45.0

        with (
            patch(
                "advisor.market.iv_analysis.compute_iv_percentile",
                return_value=mock_iv_result,
            ),
            patch(
                "advisor.confluence.pead_screener.check_pead_fundamental",
                side_effect=Exception("skip"),
            ),
            patch(
                "advisor.confluence.dip_screener.check_dip_fundamental",
                side_effect=Exception("skip"),
            ),
        ):
            result = detect_scenario("AAPL")

        assert result.scenario_type == ScenarioType.RANGE_BOUND

    @patch("advisor.strategy_case.scenarios.yf.Ticker")
    def test_momentum_detected(self, mock_ticker_cls):
        """Trending up above SMA-20 should detect MOMENTUM."""
        prices = [90 + i * 0.5 for i in range(60)]
        hist = _make_hist(prices)
        mock_ticker_cls.return_value.history.return_value = hist

        mock_iv_result = MagicMock()
        mock_iv_result.iv_percentile = 40.0

        with (
            patch(
                "advisor.market.iv_analysis.compute_iv_percentile",
                return_value=mock_iv_result,
            ),
            patch(
                "advisor.confluence.pead_screener.check_pead_fundamental",
                side_effect=Exception("skip"),
            ),
            patch(
                "advisor.confluence.dip_screener.check_dip_fundamental",
                side_effect=Exception("skip"),
            ),
        ):
            result = detect_scenario("AAPL")

        assert result.scenario_type == ScenarioType.MOMENTUM

    @patch("advisor.strategy_case.scenarios.yf.Ticker")
    def test_insufficient_data(self, mock_ticker_cls):
        """Too little data should return RANGE_BOUND with low confidence."""
        hist = pd.DataFrame({"Close": [100.0] * 5})
        mock_ticker_cls.return_value.history.return_value = hist

        result = detect_scenario("XYZ")
        assert result.scenario_type == ScenarioType.RANGE_BOUND
        assert result.confidence == 0.3
