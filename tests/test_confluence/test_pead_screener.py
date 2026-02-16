"""Tests for the PEAD (Post-Earnings Announcement Drift) screener."""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest
from advisor.confluence.models import (
    EarningsSurpriseResult,
    FadeSetupResult,
)
from advisor.confluence.pead_screener import (
    _check_earnings_surprise,
    _check_fade_setup,
    _check_revenue_surprise,
    _compute_pead_score,
    check_pead_fundamental,
)

# ── Helpers ────────────────────────────────────────────────────────────────


def _make_earnings_dates(
    eps_estimate: float = 1.0,
    eps_actual: float = 1.10,
    days_ago: int = 3,
) -> pd.DataFrame:
    """Build a mock earnings_dates DataFrame."""
    report_date = pd.Timestamp(date.today() - timedelta(days=days_ago))
    future_date = pd.Timestamp(date.today() + timedelta(days=90))
    return pd.DataFrame(
        {
            "EPS Estimate": [eps_estimate, 1.0],
            "Reported EPS": [eps_actual, pd.NA],
        },
        index=pd.DatetimeIndex([report_date, future_date]),
    )


def _make_history(
    pre_prices: list[float],
    post_prices: list[float],
    report_date: date,
) -> pd.DataFrame:
    """Build a mock history DataFrame spanning pre- and post-earnings."""
    n_total = len(pre_prices) + len(post_prices)
    all_prices = pre_prices + post_prices

    # Generate enough contiguous business days and split at report_date boundary
    start = report_date - timedelta(days=len(pre_prices) * 3)
    all_bdays = pd.bdate_range(start=start, periods=n_total * 3)

    pre_bdays = all_bdays[all_bdays < pd.Timestamp(report_date)][-len(pre_prices) :]
    post_bdays = all_bdays[all_bdays >= pd.Timestamp(report_date)][: len(post_prices)]
    selected = pre_bdays.append(post_bdays)

    return pd.DataFrame(
        {
            "Open": all_prices,
            "High": [p * 1.01 for p in all_prices],
            "Low": [p * 0.99 for p in all_prices],
            "Close": all_prices,
            "Volume": [1_000_000] * len(all_prices),
        },
        index=selected,
    )


def _make_quarterly_financials(revenues: list[float]) -> pd.DataFrame:
    """Build a mock quarterly_financials DataFrame with revenue rows."""
    dates = pd.date_range(end=date.today(), periods=len(revenues), freq="QE")[::-1]
    return pd.DataFrame(
        {"Total Revenue": revenues},
        index=pd.Index(["Total Revenue"]),
        columns=dates,
    ).T.T  # Transpose trick to get labels as index


# ===========================================================================
# Layer 1: Earnings Surprise
# ===========================================================================


class TestCheckEarningsSurprise:
    def test_passes_with_large_beat(self):
        """EPS beat >5% within 7 days should pass."""
        ticker = MagicMock()
        ticker.earnings_dates = _make_earnings_dates(eps_estimate=1.0, eps_actual=1.10, days_ago=3)
        ticker.quarterly_financials = None
        ticker.info = {"revenueGrowth": 0.05}

        result = _check_earnings_surprise(ticker)

        assert result.passes is True
        assert result.eps_surprise_pct == pytest.approx(10.0)
        assert result.days_since_report == 3

    def test_fails_with_small_beat(self):
        """EPS beat <5% should fail."""
        ticker = MagicMock()
        ticker.earnings_dates = _make_earnings_dates(eps_estimate=1.0, eps_actual=1.03, days_ago=3)
        ticker.quarterly_financials = None
        ticker.info = {"revenueGrowth": 0.05}

        result = _check_earnings_surprise(ticker)

        assert result.passes is False
        assert result.eps_surprise_pct == pytest.approx(3.0)

    def test_fails_with_miss(self):
        """EPS miss should fail."""
        ticker = MagicMock()
        ticker.earnings_dates = _make_earnings_dates(eps_estimate=1.0, eps_actual=0.90, days_ago=3)
        ticker.quarterly_financials = None
        ticker.info = {"revenueGrowth": -0.05}

        result = _check_earnings_surprise(ticker)

        assert result.passes is False
        assert result.eps_surprise_pct == pytest.approx(-10.0)

    def test_fails_when_too_old(self):
        """Earnings reported >7 days ago should fail."""
        ticker = MagicMock()
        ticker.earnings_dates = _make_earnings_dates(eps_estimate=1.0, eps_actual=1.20, days_ago=10)
        ticker.quarterly_financials = None
        ticker.info = {"revenueGrowth": 0.05}

        result = _check_earnings_surprise(ticker)

        assert result.passes is False

    def test_fails_with_no_earnings_data(self):
        """No earnings_dates should return default (fails)."""
        ticker = MagicMock()
        ticker.earnings_dates = None

        result = _check_earnings_surprise(ticker)

        assert result.passes is False

    def test_revenue_surprise_negative(self):
        """Negative revenue growth should set revenue_surprise=False."""
        ticker = MagicMock()
        ticker.earnings_dates = _make_earnings_dates(eps_estimate=1.0, eps_actual=1.10, days_ago=3)
        ticker.quarterly_financials = None
        ticker.info = {"revenueGrowth": -0.02}

        result = _check_earnings_surprise(ticker)

        assert result.passes is True  # EPS still passes
        assert result.revenue_surprise is False


# ===========================================================================
# Revenue surprise (quarterly financials)
# ===========================================================================


class TestCheckRevenueSurprise:
    @staticmethod
    def _make_qf(revenues: list[float]) -> pd.DataFrame:
        """Build a mock quarterly_financials (items as rows, dates as cols)."""
        dates = pd.date_range(end="2025-12-31", periods=len(revenues), freq="QE")[::-1]
        return pd.DataFrame(
            [revenues],
            index=pd.Index(["Total Revenue"]),
            columns=dates,
        )

    def test_quarterly_yoy_growth(self):
        """Latest quarter revenue > year-ago quarter = True."""
        ticker = MagicMock()
        ticker.quarterly_financials = self._make_qf([120.0, 115.0, 110.0, 105.0, 100.0])
        ticker.info = {}

        assert _check_revenue_surprise(ticker) is True

    def test_quarterly_yoy_decline(self):
        """Latest quarter revenue < year-ago quarter = False."""
        ticker = MagicMock()
        ticker.quarterly_financials = self._make_qf([90.0, 95.0, 100.0, 105.0, 110.0])
        ticker.info = {}

        assert _check_revenue_surprise(ticker) is False

    def test_fallback_to_revenue_growth(self):
        """When quarterly_financials is None, falls back to ticker.info."""
        ticker = MagicMock()
        ticker.quarterly_financials = None
        ticker.info = {"revenueGrowth": 0.08}

        assert _check_revenue_surprise(ticker) is True

    def test_fallback_negative_growth(self):
        """Fallback with negative revenueGrowth returns False."""
        ticker = MagicMock()
        ticker.quarterly_financials = None
        ticker.info = {"revenueGrowth": -0.03}

        assert _check_revenue_surprise(ticker) is False

    def test_no_data_returns_none(self):
        """No quarterly data and no info field returns None."""
        ticker = MagicMock()
        ticker.quarterly_financials = None
        ticker.info = {}

        assert _check_revenue_surprise(ticker) is None


# ===========================================================================
# Layer 2: Fade Setup
# ===========================================================================


class TestCheckFadeSetup:
    def test_passes_with_gap_and_fade(self):
        """Stock that gapped up then faded back should pass."""
        report_date = date.today() - timedelta(days=3)
        ticker = MagicMock()

        pre_prices = [98.0, 99.0, 100.0, 101.0, 100.0]
        post_prices = [108.0, 104.0, 99.0, 98.0]
        ticker.history.return_value = _make_history(pre_prices, post_prices, report_date)

        result = _check_fade_setup(ticker, report_date)

        assert result.passes is True
        assert result.has_faded is True
        assert result.gap_and_fade is True
        # Pre-earnings high is max of High column = 101 * 1.01 = 102.01
        assert result.pre_earnings_high == pytest.approx(102.01, abs=0.1)

    def test_fails_when_price_above_pre_earnings_high(self):
        """Stock still above pre-earnings high should fail."""
        report_date = date.today() - timedelta(days=3)
        ticker = MagicMock()

        pre_prices = [98.0, 99.0, 100.0, 101.0, 100.0]
        post_prices = [108.0, 106.0, 105.0, 103.0]
        ticker.history.return_value = _make_history(pre_prices, post_prices, report_date)

        result = _check_fade_setup(ticker, report_date)

        assert result.passes is False
        assert result.has_faded is False

    def test_fails_when_too_soon(self):
        """Only 1 day since earnings should fail (need 2+)."""
        report_date = date.today() - timedelta(days=1)
        ticker = MagicMock()

        pre_prices = [98.0, 99.0, 100.0, 101.0, 100.0]
        post_prices = [108.0, 99.0]
        ticker.history.return_value = _make_history(pre_prices, post_prices, report_date)

        result = _check_fade_setup(ticker, report_date)

        assert result.passes is False

    def test_fails_when_window_expired(self):
        """More than 7 days since earnings should fail."""
        report_date = date.today() - timedelta(days=10)
        ticker = MagicMock()

        pre_prices = [98.0, 99.0, 100.0, 101.0, 100.0]
        post_prices = [108.0] + [99.0] * 10
        ticker.history.return_value = _make_history(pre_prices, post_prices, report_date)

        result = _check_fade_setup(ticker, report_date)

        assert result.passes is False

    def test_uses_high_column_for_pre_earnings(self):
        """Pre-earnings high should use High column, not Close."""
        report_date = date.today() - timedelta(days=3)
        ticker = MagicMock()

        # Close=100 but High=101 (100 * 1.01 from helper)
        pre_prices = [95.0, 96.0, 97.0, 98.0, 100.0]
        post_prices = [108.0, 104.0, 99.0, 98.0]
        ticker.history.return_value = _make_history(pre_prices, post_prices, report_date)

        result = _check_fade_setup(ticker, report_date)

        # High = 100 * 1.01 = 101.0
        assert result.pre_earnings_high == pytest.approx(101.0, abs=0.1)


# ===========================================================================
# Scoring
# ===========================================================================


class TestComputePeadScore:
    def test_fail_when_surprise_does_not_pass(self):
        surprise = EarningsSurpriseResult(passes=False)
        assert _compute_pead_score(surprise, None) == "FAIL"

    def test_fail_when_fade_does_not_pass(self):
        """Score requires BOTH layers to pass."""
        surprise = EarningsSurpriseResult(passes=True, eps_surprise_pct=15.0, revenue_surprise=True)
        fade = FadeSetupResult(passes=False)
        assert _compute_pead_score(surprise, fade) == "FAIL"

    def test_fail_when_fade_is_none(self):
        surprise = EarningsSurpriseResult(passes=True, eps_surprise_pct=15.0, revenue_surprise=True)
        assert _compute_pead_score(surprise, None) == "FAIL"

    def test_strong_buy_with_all_signals(self):
        """EPS >10% (+2), revenue (+1), gap-and-fade (+1), deep fade (+1) = 5 → STRONG_BUY."""
        surprise = EarningsSurpriseResult(
            passes=True,
            eps_surprise_pct=15.0,
            revenue_surprise=True,
        )
        fade = FadeSetupResult(
            passes=True,
            gap_and_fade=True,
            fade_pct=-0.08,
        )
        assert _compute_pead_score(surprise, fade) == "STRONG_BUY"

    def test_buy_with_three_points(self):
        """EPS >10% (+2) + revenue (+1) = 3 → BUY."""
        surprise = EarningsSurpriseResult(
            passes=True,
            eps_surprise_pct=12.0,
            revenue_surprise=True,
        )
        fade = FadeSetupResult(passes=True, fade_pct=-0.01)
        assert _compute_pead_score(surprise, fade) == "BUY"

    def test_lean_buy_with_two_points(self):
        """EPS >5% (+1) + revenue (+1) = 2 → LEAN_BUY."""
        surprise = EarningsSurpriseResult(
            passes=True,
            eps_surprise_pct=7.0,
            revenue_surprise=True,
        )
        fade = FadeSetupResult(passes=True, fade_pct=-0.01)
        assert _compute_pead_score(surprise, fade) == "LEAN_BUY"

    def test_watch_with_one_point(self):
        """EPS >5% (+1) only = 1 → WATCH."""
        surprise = EarningsSurpriseResult(
            passes=True,
            eps_surprise_pct=6.0,
            revenue_surprise=False,
        )
        fade = FadeSetupResult(passes=True, fade_pct=-0.01)
        assert _compute_pead_score(surprise, fade) == "WATCH"


# ===========================================================================
# Top-level check_pead_fundamental
# ===========================================================================


class TestCheckPeadFundamental:
    @patch("advisor.confluence.pead_screener.yf.Ticker")
    def test_returns_fundamental_result_with_passing_screener(self, mock_ticker_cls):
        ticker = MagicMock()
        mock_ticker_cls.return_value = ticker

        # Earnings surprise passes
        ticker.earnings_dates = _make_earnings_dates(eps_estimate=1.0, eps_actual=1.15, days_ago=3)
        ticker.quarterly_financials = None
        ticker.info = {"revenueGrowth": 0.10}

        # Fade setup passes
        report_date = date.today() - timedelta(days=3)
        pre_prices = [98.0, 99.0, 100.0, 101.0, 100.0]
        post_prices = [108.0, 104.0, 99.0, 98.0]
        ticker.history.return_value = _make_history(pre_prices, post_prices, report_date)

        result = check_pead_fundamental("AAPL")

        assert result.is_clear is True
        assert result.earnings_within_7_days is True
        assert result.pead_screener is not None
        assert result.pead_screener.overall_score != "FAIL"
        assert result.pead_screener.earnings_surprise.passes is True
        assert result.pead_screener.fade_setup is not None
        assert result.pead_screener.fade_setup.passes is True

    @patch("advisor.confluence.pead_screener.yf.Ticker")
    def test_returns_failing_result_with_small_beat(self, mock_ticker_cls):
        ticker = MagicMock()
        mock_ticker_cls.return_value = ticker

        ticker.earnings_dates = _make_earnings_dates(eps_estimate=1.0, eps_actual=1.02, days_ago=3)
        ticker.quarterly_financials = None
        ticker.info = {"revenueGrowth": 0.05}

        result = check_pead_fundamental("AAPL")

        assert result.is_clear is False
        assert result.pead_screener is not None
        assert result.pead_screener.overall_score == "FAIL"
        assert result.pead_screener.rejection_reason is not None

    @patch("advisor.confluence.pead_screener.yf.Ticker")
    def test_earnings_within_7_days_reflects_reality(self, mock_ticker_cls):
        """earnings_within_7_days should be False when no recent earnings."""
        ticker = MagicMock()
        mock_ticker_cls.return_value = ticker

        ticker.earnings_dates = _make_earnings_dates(eps_estimate=1.0, eps_actual=1.20, days_ago=30)
        ticker.quarterly_financials = None
        ticker.info = {"revenueGrowth": 0.05}

        result = check_pead_fundamental("AAPL")

        assert result.earnings_within_7_days is False

    @patch("advisor.confluence.pead_screener.yf.Ticker")
    def test_score_is_fail_when_fade_does_not_pass(self, mock_ticker_cls):
        """Score should be FAIL when surprise passes but fade doesn't."""
        ticker = MagicMock()
        mock_ticker_cls.return_value = ticker

        # EPS beats
        ticker.earnings_dates = _make_earnings_dates(eps_estimate=1.0, eps_actual=1.20, days_ago=3)
        ticker.quarterly_financials = None
        ticker.info = {"revenueGrowth": 0.10}

        # But price hasn't faded (still above pre-earnings high)
        report_date = date.today() - timedelta(days=3)
        pre_prices = [98.0, 99.0, 100.0, 101.0, 100.0]
        post_prices = [108.0, 106.0, 105.0, 104.0]  # stayed above
        ticker.history.return_value = _make_history(pre_prices, post_prices, report_date)

        result = check_pead_fundamental("AAPL")

        assert result.is_clear is False
        assert result.pead_screener.overall_score == "FAIL"
