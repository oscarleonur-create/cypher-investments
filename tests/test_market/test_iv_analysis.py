"""Tests for IV analysis module — percentile, term structure, expected move, earnings."""

from __future__ import annotations

from datetime import date, datetime, timedelta
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
from advisor.market.iv_analysis import (
    classify_term_structure,
    compute_expected_move,
    compute_iv_percentile,
    get_next_earnings_date,
)

# ── Expected Move ─────────────────────────────────────────────────────────────


def test_expected_move_basic():
    # price=100, IV=0.30, DTE=30
    em = compute_expected_move(100, 0.30, 30)
    # 100 * 0.30 * sqrt(30/365) ≈ 8.60
    assert 8.0 < em < 9.5


def test_expected_move_zero_inputs():
    assert compute_expected_move(0, 0.30, 30) == 0.0
    assert compute_expected_move(100, 0, 30) == 0.0
    assert compute_expected_move(100, 0.30, 0) == 0.0


def test_expected_move_scales_with_sqrt_dte():
    em_30 = compute_expected_move(100, 0.30, 30)
    em_120 = compute_expected_move(100, 0.30, 120)
    # 120 DTE should be ~2x the 30 DTE move (sqrt(4) = 2)
    ratio = em_120 / em_30
    assert 1.9 < ratio < 2.1


def test_expected_move_scales_with_iv():
    em_low = compute_expected_move(100, 0.20, 30)
    em_high = compute_expected_move(100, 0.40, 30)
    assert abs(em_high / em_low - 2.0) < 0.01


# ── IV Percentile ─────────────────────────────────────────────────────────────


@patch("advisor.market.iv_analysis.yf.Ticker")
def test_iv_percentile_tastytrade_shortcut(mock_ticker_cls):
    """When TastyTrade data is provided, skip yfinance entirely."""
    tt_data = {
        "PLTR": {"iv_percentile": 0.72, "iv_index": 0.55, "iv_rank": 0.68},
    }
    result = compute_iv_percentile("PLTR", tt_data=tt_data)
    assert result.source == "tastytrade"
    assert result.iv_percentile == 72.0
    assert result.iv_rank == 0.68
    mock_ticker_cls.assert_not_called()


@patch("advisor.market.iv_analysis._extract_atm_iv")
@patch("advisor.market.iv_analysis._get_price")
@patch("advisor.market.iv_analysis.yf.Ticker")
def test_iv_percentile_estimated(mock_ticker_cls, mock_get_price, mock_extract_iv):
    """Test estimated IV percentile using HV30 distribution."""
    mock_ticker = MagicMock()
    mock_ticker_cls.return_value = mock_ticker
    mock_get_price.return_value = 25.0
    mock_ticker.options = ("2026-04-03",)
    mock_extract_iv.return_value = 0.60  # current ATM IV = 60%

    # Build 1yr history where HV30 ranges 20%-50%
    # IV of 60% should be above most of the HV30 distribution
    np.random.seed(42)
    dates = pd.date_range("2025-03-01", periods=252, freq="B")
    prices = 25.0 * np.exp(np.cumsum(np.random.normal(0.0002, 0.01, 252)))
    hist_df = pd.DataFrame({"Close": prices}, index=dates)
    mock_ticker.history.return_value = hist_df

    result = compute_iv_percentile("PLTR", ticker=mock_ticker)
    assert result.source == "estimated"
    assert result.current_iv == 0.60
    # With low HV, a 60% IV should produce high percentile
    assert result.iv_percentile > 50


@patch("advisor.market.iv_analysis._get_price")
@patch("advisor.market.iv_analysis.yf.Ticker")
def test_iv_percentile_no_options(mock_ticker_cls, mock_get_price):
    """Returns 50% when no options data is available."""
    mock_ticker = MagicMock()
    mock_ticker_cls.return_value = mock_ticker
    mock_get_price.return_value = 25.0
    mock_ticker.options = ()

    result = compute_iv_percentile("PLTR", ticker=mock_ticker)
    assert result.iv_percentile == 50.0


# ── Term Structure ────────────────────────────────────────────────────────────


@patch("advisor.market.iv_analysis._extract_atm_iv")
@patch("advisor.market.iv_analysis._get_price")
def test_term_structure_contango(mock_get_price, mock_extract_iv):
    """Rising IV across expirations = contango."""
    mock_ticker = MagicMock()
    mock_get_price.return_value = 100.0

    today = date.today()
    expirations = [(today + timedelta(days=d)).isoformat() for d in [30, 60, 90, 120]]
    mock_ticker.options = expirations

    # IV rises with DTE: 0.25, 0.30, 0.35, 0.40
    mock_extract_iv.side_effect = [0.25, 0.30, 0.35, 0.40]

    result = classify_term_structure("TEST", ticker=mock_ticker, price=100.0)
    assert result.classification == "contango"
    assert result.slope > 0


@patch("advisor.market.iv_analysis._extract_atm_iv")
@patch("advisor.market.iv_analysis._get_price")
def test_term_structure_backwardation(mock_get_price, mock_extract_iv):
    """Falling IV across expirations = backwardation."""
    mock_ticker = MagicMock()
    mock_get_price.return_value = 100.0

    today = date.today()
    expirations = [(today + timedelta(days=d)).isoformat() for d in [30, 60, 90, 120]]
    mock_ticker.options = expirations

    # IV falls with DTE: 0.50, 0.40, 0.30, 0.25
    mock_extract_iv.side_effect = [0.50, 0.40, 0.30, 0.25]

    result = classify_term_structure("TEST", ticker=mock_ticker, price=100.0)
    assert result.classification == "backwardation"
    assert result.slope < 0


@patch("advisor.market.iv_analysis._extract_atm_iv")
@patch("advisor.market.iv_analysis._get_price")
def test_term_structure_flat(mock_get_price, mock_extract_iv):
    """Similar IV across expirations = flat."""
    mock_ticker = MagicMock()
    mock_get_price.return_value = 100.0

    today = date.today()
    expirations = [(today + timedelta(days=d)).isoformat() for d in [30, 60, 90]]
    mock_ticker.options = expirations

    mock_extract_iv.side_effect = [0.30, 0.30, 0.31]

    result = classify_term_structure("TEST", ticker=mock_ticker, price=100.0)
    assert result.classification == "flat"


# ── Earnings Date ─────────────────────────────────────────────────────────────


def test_earnings_date_from_calendar():
    """Extract earnings date from yfinance calendar dict."""
    mock_ticker = MagicMock()
    future_date = datetime.now() + timedelta(days=30)
    mock_ticker.calendar = {"Earnings Date": [future_date]}

    result = get_next_earnings_date("PLTR", ticker=mock_ticker)
    assert result == future_date.date()


def test_earnings_date_skips_past():
    """Should skip earnings dates in the past."""
    mock_ticker = MagicMock()
    past = datetime.now() - timedelta(days=10)
    future = datetime.now() + timedelta(days=45)
    mock_ticker.calendar = {"Earnings Date": [past, future]}

    result = get_next_earnings_date("PLTR", ticker=mock_ticker)
    assert result == future.date()


def test_earnings_date_none_calendar():
    """Returns None when calendar is None."""
    mock_ticker = MagicMock()
    mock_ticker.calendar = None

    result = get_next_earnings_date("PLTR", ticker=mock_ticker)
    assert result is None
