"""Tests for the fundamental confluence agent."""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd

from advisor.confluence.fundamental import check_fundamental


class TestCheckFundamental:
    @patch("advisor.confluence.fundamental.yf")
    def test_no_earnings_risk_is_clear(self, mock_yf):
        ticker = MagicMock()
        # Earnings date far in the future
        future_date = date.today() + timedelta(days=30)
        ticker.calendar = {"Earnings Date": [future_date]}
        ticker.insider_transactions = pd.DataFrame(
            {"Transaction": ["Purchase"], "Shares": [1000]}
        )
        mock_yf.Ticker.return_value = ticker

        result = check_fundamental("AAPL")

        assert result.is_clear is True
        assert result.earnings_within_7_days is False
        assert result.insider_buying_detected is True

    @patch("advisor.confluence.fundamental.yf")
    def test_earnings_within_7_days_is_not_clear(self, mock_yf):
        ticker = MagicMock()
        # Earnings in 3 days
        soon_date = date.today() + timedelta(days=3)
        ticker.calendar = {"Earnings Date": [soon_date]}
        ticker.insider_transactions = pd.DataFrame()
        mock_yf.Ticker.return_value = ticker

        result = check_fundamental("AAPL")

        assert result.is_clear is False
        assert result.earnings_within_7_days is True
        assert result.earnings_date == soon_date

    @patch("advisor.confluence.fundamental.yf")
    def test_insider_buying_detected(self, mock_yf):
        ticker = MagicMock()
        ticker.calendar = None
        ticker.insider_transactions = pd.DataFrame(
            {"Transaction": ["Purchase", "Sale"], "Shares": [500, -200]}
        )
        mock_yf.Ticker.return_value = ticker

        result = check_fundamental("AAPL")

        assert result.insider_buying_detected is True

    @patch("advisor.confluence.fundamental.yf")
    def test_no_insider_buying(self, mock_yf):
        ticker = MagicMock()
        ticker.calendar = None
        ticker.insider_transactions = pd.DataFrame(
            {"Transaction": ["Sale", "Sale"], "Shares": [-500, -200]}
        )
        mock_yf.Ticker.return_value = ticker

        result = check_fundamental("AAPL")

        assert result.insider_buying_detected is False

    @patch("advisor.confluence.fundamental.yf")
    def test_empty_insider_transactions(self, mock_yf):
        ticker = MagicMock()
        ticker.calendar = None
        ticker.insider_transactions = pd.DataFrame()
        mock_yf.Ticker.return_value = ticker

        result = check_fundamental("AAPL")

        assert result.insider_buying_detected is False
        assert result.is_clear is True

    @patch("advisor.confluence.fundamental.yf")
    def test_none_calendar_no_crash(self, mock_yf):
        ticker = MagicMock()
        ticker.calendar = None
        ticker.insider_transactions = None
        mock_yf.Ticker.return_value = ticker

        result = check_fundamental("AAPL")

        assert result.is_clear is True
        assert result.earnings_within_7_days is False
        assert result.insider_buying_detected is False

    @patch("advisor.confluence.fundamental.yf")
    def test_empty_transaction_column_falls_back_to_shares(self, mock_yf):
        """When Transaction column exists but is all empty, fall back to Shares > 0."""
        ticker = MagicMock()
        ticker.calendar = None
        ticker.insider_transactions = pd.DataFrame(
            {"Transaction": ["", "", ""], "Shares": [1255, 1255, 3750]}
        )
        mock_yf.Ticker.return_value = ticker

        result = check_fundamental("AAPL")

        assert result.insider_buying_detected is True

    @patch("advisor.confluence.fundamental.yf")
    def test_empty_transaction_with_negative_shares(self, mock_yf):
        """Empty Transaction column with negative shares should not detect buying."""
        ticker = MagicMock()
        ticker.calendar = None
        ticker.insider_transactions = pd.DataFrame(
            {"Transaction": ["", ""], "Shares": [-500, -200]}
        )
        mock_yf.Ticker.return_value = ticker

        result = check_fundamental("AAPL")

        assert result.insider_buying_detected is False

    @patch("advisor.confluence.fundamental.yf")
    def test_calendar_exception_handled(self, mock_yf):
        ticker = MagicMock()
        ticker.calendar = property(lambda self: (_ for _ in ()).throw(RuntimeError("API down")))
        # Make .calendar raise
        type(ticker).calendar = property(lambda self: (_ for _ in ()).throw(RuntimeError("API down")))
        ticker.insider_transactions = None
        mock_yf.Ticker.return_value = ticker

        result = check_fundamental("AAPL")

        assert result.is_clear is True  # defaults to safe
