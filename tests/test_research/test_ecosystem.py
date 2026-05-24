"""Tests for ecosystem modules — holders, insiders (mocked yfinance)."""

from __future__ import annotations

from datetime import date, timedelta
from unittest.mock import MagicMock, patch

import pandas as pd

_RECENT = date.today() - timedelta(days=30)


def _make_holders_df():
    return pd.DataFrame(
        {
            "Holder": ["Vanguard Group", "BlackRock", "State Street"],
            "Shares": [1_200_000_000, 1_000_000_000, 600_000_000],
            "% Out": [0.079, 0.066, 0.040],
            "Value": [250e9, 210e9, 125e9],
            "Date Reported": ["2025-03-31", "2025-03-31", "2025-03-31"],
        }
    )


def _make_insider_df():
    return pd.DataFrame(
        {
            "Insider": ["Tim Cook", "Luca Maestri", "Jeff Williams"],
            "Position": ["CEO", "CFO", "COO"],
            "Transaction": ["Purchase", "Sale", "Purchase"],
            "Shares": [10_000, -5_000, 8_000],
            "Value": [2_100_000, -1_050_000, 1_680_000],
            "Start Date": [_RECENT, _RECENT, _RECENT],
        }
    )


def test_get_holders_parses_yfinance():
    from advisor.research.ecosystem.holders import get_holders

    fake_ticker = MagicMock()
    fake_ticker.institutional_holders = _make_holders_df()
    fake_ticker.major_holders = pd.DataFrame(
        {0: [0.02, 0.63, 0.65, 3825]},
        index=["% Insider", "% Institution", "% Float", "Shares Outstanding"],
    )

    with patch("yfinance.Ticker", return_value=fake_ticker):
        result = get_holders("AAPL")

    assert result.symbol == "AAPL"
    assert len(result.top_holders) == 3
    assert result.top_holders[0].name == "Vanguard Group"
    assert result.top_holders[0].shares == 1_200_000_000


def test_get_holders_empty_df_returns_empty_list():
    from advisor.research.ecosystem.holders import get_holders

    fake_ticker = MagicMock()
    fake_ticker.institutional_holders = pd.DataFrame()
    fake_ticker.major_holders = pd.DataFrame()

    with patch("yfinance.Ticker", return_value=fake_ticker):
        result = get_holders("AAPL")

    assert result.top_holders == []


def test_get_insiders_parses_transactions():
    from advisor.research.ecosystem.insiders import get_insiders

    fake_ticker = MagicMock()
    fake_ticker.insider_transactions = _make_insider_df()

    with patch("yfinance.Ticker", return_value=fake_ticker):
        result = get_insiders("AAPL")

    assert result.symbol == "AAPL"
    assert len(result.transactions) == 3
    purchases = [t for t in result.transactions if t.transaction_type == "Purchase"]
    assert len(purchases) == 2
    assert result.c_suite_buying is True  # CEO purchased


def test_get_insiders_net_buying_sign():
    from advisor.research.ecosystem.insiders import get_insiders

    # Only sales → negative net buying
    df = pd.DataFrame(
        {
            "Insider": ["Director A"],
            "Position": ["Director"],
            "Transaction": ["Sale"],
            "Shares": [10_000],
            "Value": [500_000],
            "Start Date": [_RECENT],
        }
    )
    fake_ticker = MagicMock()
    fake_ticker.insider_transactions = df

    with patch("yfinance.Ticker", return_value=fake_ticker):
        result = get_insiders("TEST")

    assert result.net_buying_usd < 0
