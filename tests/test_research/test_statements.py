"""Statement extraction tests — EDGAR mock + yfinance fallback."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pandas as pd
from advisor.research.statements import extract_statements

# ── EDGAR path ───────────────────────────────────────────────────────────────


def test_edgar_path_populates_bundle():
    fake_client = MagicMock()
    fake_client.get_financials.return_value = {
        "income": [
            {
                "index": "2024-12-31",
                "Total Revenue": 1000,
                "Net Income": 144,
                "Operating Income": 200,
                "Diluted EPS": 1.44,
                "Diluted Average Shares": 100,
                "EBITDA": 240,
            },
            {
                "index": "2023-12-31",
                "Total Revenue": 900,
                "Net Income": 130,
                "Operating Income": 180,
            },
        ],
        "balance": [
            {
                "index": "2024-12-31",
                "Total Assets": 2000,
                "Current Liabilities": 300,
                "Current Assets": 500,
                "Long Term Debt": 450,
                "Stockholders Equity": 1100,
            },
        ],
        "cashflow": [
            {"index": "2024-12-31", "Operating Cash Flow": 200, "Capital Expenditure": -50},
        ],
    }

    bundle = extract_statements("TEST", edgar_client=fake_client, fallback_to_yfinance=False)

    assert bundle.source == "edgar"
    assert len(bundle.income) == 2
    assert bundle.income[0].revenue == 1000
    assert bundle.income[0].net_income == 144
    assert bundle.balance[0].total_assets == 2000
    assert bundle.cashflow[0].operating_cash_flow == 200


def test_edgar_failure_triggers_yfinance_fallback():
    fake_client = MagicMock()
    fake_client.get_financials.side_effect = RuntimeError("network down")

    with patch("advisor.research.statements._from_yfinance") as fake_yf:
        fake_yf.return_value = MagicMock(source="yfinance")
        extract_statements("AAPL", edgar_client=fake_client)
        fake_yf.assert_called_once_with("AAPL", 5)


def test_sparse_edgar_falls_back_to_yfinance():
    fake_client = MagicMock()
    # No revenue / assets / OCF anywhere — _is_populated returns False
    fake_client.get_financials.return_value = {
        "income": [{"index": "2024-12-31"}],
        "balance": [{"index": "2024-12-31"}],
        "cashflow": [{"index": "2024-12-31"}],
    }

    with patch("advisor.research.statements._from_yfinance") as fake_yf:
        fake_yf.return_value = MagicMock(source="yfinance")
        extract_statements("AAPL", edgar_client=fake_client)
        fake_yf.assert_called_once()


# ── yfinance path ────────────────────────────────────────────────────────────


def _yf_df(rows: dict[str, list[float]], cols: list[date]) -> pd.DataFrame:
    """Build a yfinance-shaped DataFrame: rows=labels, cols=period-end dates."""
    return (
        pd.DataFrame(rows, index=list(rows.keys())).T
        if False
        else pd.DataFrame(
            {pd.Timestamp(c): list(vals) for c, vals in zip(cols, zip(*rows.values()))},
            index=list(rows.keys()),
        )
    )


def test_yfinance_extraction_maps_canonical_labels():
    income_df = pd.DataFrame(
        {
            pd.Timestamp("2024-12-31"): [1000.0, 200.0, 144.0],
            pd.Timestamp("2023-12-31"): [900.0, 180.0, 130.0],
        },
        index=["Total Revenue", "Operating Income", "Net Income"],
    )
    balance_df = pd.DataFrame(
        {pd.Timestamp("2024-12-31"): [2000.0, 1100.0, 500.0]},
        index=["Total Assets", "Stockholders Equity", "Current Assets"],
    )
    cashflow_df = pd.DataFrame(
        {pd.Timestamp("2024-12-31"): [200.0, -50.0]},
        index=["Operating Cash Flow", "Capital Expenditure"],
    )

    fake_ticker = MagicMock()
    fake_ticker.income_stmt = income_df
    fake_ticker.balance_sheet = balance_df
    fake_ticker.cashflow = cashflow_df

    with patch("yfinance.Ticker", return_value=fake_ticker):
        bundle = extract_statements("AAPL", edgar_client=None)

    assert bundle.source == "yfinance"
    assert bundle.income[0].revenue == 1000.0
    assert bundle.income[0].net_income == 144.0
    assert bundle.balance[0].total_assets == 2000.0
    assert bundle.cashflow[0].capex == -50.0


def test_yfinance_missing_data_returns_empty_bundle():
    fake_ticker = MagicMock()
    fake_ticker.income_stmt = pd.DataFrame()  # empty
    fake_ticker.balance_sheet = pd.DataFrame()
    fake_ticker.cashflow = pd.DataFrame()

    with patch("yfinance.Ticker", return_value=fake_ticker):
        bundle = extract_statements("XXX", edgar_client=None)

    assert bundle.source == "yfinance"
    assert bundle.income == []
    assert bundle.balance == []
    assert bundle.cashflow == []
