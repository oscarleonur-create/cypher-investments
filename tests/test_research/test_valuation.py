"""Tests for DCF engine, multiples, and reverse-DCF."""

from __future__ import annotations

import pytest
from advisor.research.models import DcfAssumptions, DcfResult

# ── DCF scenario math ─────────────────────────────────────────────────────────


def _make_assumptions(**kwargs) -> DcfAssumptions:
    defaults = dict(
        scenario="base",
        revenue_growth_yr1_3=0.10,
        revenue_growth_yr4_10=0.05,
        target_fcf_margin=0.15,
        capex_intensity=0.03,
        terminal_growth_rate=0.025,
        terminal_exit_multiple=None,
        wacc=0.09,
    )
    defaults.update(kwargs)
    return DcfAssumptions(**defaults)


def test_dcf_projects_ten_fcf_years():
    from advisor.research.valuation.dcf import _run_scenario

    a = _make_assumptions()
    scenario = _run_scenario(
        a,
        base_revenue=1000.0,
        seed_fcf=150.0,
        net_debt=0.0,
        shares=100.0,
        current_price=50.0,
    )
    assert len(scenario.projected_fcf) == 10
    # Revenue should grow ~10% for years 1-3
    assert scenario.projected_fcf[0] > 0
    assert scenario.projected_fcf[2] > scenario.projected_fcf[0]


def test_dcf_terminal_value_is_positive():
    from advisor.research.valuation.dcf import _run_scenario

    a = _make_assumptions()
    scenario = _run_scenario(
        a, base_revenue=1000.0, seed_fcf=150.0, net_debt=0.0, shares=100.0, current_price=50.0
    )
    assert scenario.terminal_value_gordon is not None and scenario.terminal_value_gordon > 0
    assert scenario.enterprise_value > 0
    assert scenario.implied_price > 0


def test_dcf_bull_price_above_base_above_bear():
    from advisor.research.valuation.dcf import _run_scenario

    kwargs = dict(
        base_revenue=1000.0, seed_fcf=150.0, net_debt=0.0, shares=100.0, current_price=50.0
    )
    base = _run_scenario(_make_assumptions(scenario="base"), **kwargs)
    bull = _run_scenario(
        _make_assumptions(
            scenario="bull", revenue_growth_yr1_3=0.15, target_fcf_margin=0.18, wacc=0.08
        ),
        **kwargs,
    )
    bear = _run_scenario(
        _make_assumptions(
            scenario="bear", revenue_growth_yr1_3=0.04, target_fcf_margin=0.10, wacc=0.10
        ),
        **kwargs,
    )

    assert bull.implied_price > base.implied_price > bear.implied_price


def test_dcf_net_debt_reduces_equity_value():
    from advisor.research.valuation.dcf import _run_scenario

    a = _make_assumptions()
    no_debt = _run_scenario(
        a, base_revenue=1000.0, seed_fcf=150.0, net_debt=0.0, shares=100.0, current_price=50.0
    )
    with_debt = _run_scenario(
        a, base_revenue=1000.0, seed_fcf=150.0, net_debt=200.0, shares=100.0, current_price=50.0
    )

    assert no_debt.implied_price > with_debt.implied_price
    assert no_debt.equity_value - with_debt.equity_value == pytest.approx(200.0, rel=0.01)


def test_dcf_exit_multiple_terminal_value():
    from advisor.research.valuation.dcf import _run_scenario

    a = _make_assumptions(terminal_exit_multiple=20.0)
    scenario = _run_scenario(
        a, base_revenue=1000.0, seed_fcf=150.0, net_debt=0.0, shares=100.0, current_price=50.0
    )
    assert scenario.terminal_value_exit is not None and scenario.terminal_value_exit > 0


# ── Reverse-DCF ──────────────────────────────────────────────────────────────


def test_reverse_dcf_returns_float(clean_bundle):
    from unittest.mock import patch

    from advisor.research.valuation.dcf import build_dcf
    from advisor.research.valuation.reverse_dcf import solve_implied_growth

    # Mock yfinance so we don't hit the network
    fake_info = {
        "currentPrice": 50.0,
        "sharesOutstanding": 100e6,
        "totalDebt": 500e6,
        "totalCash": 200e6,
        "beta": 1.2,
        "totalRevenue": 1000e6,
        "freeCashflow": 150e6,
        "revenueGrowth": 0.10,
        "enterpriseToEbitda": 15.0,
    }
    with (
        patch("yfinance.Ticker") as mock_ticker,
        patch("advisor.research.valuation.dcf._risk_free_rate", return_value=0.043),
    ):
        mock_ticker.return_value.info = fake_info
        dcf = build_dcf("TEST", statements=None)

    result = solve_implied_growth(dcf)
    assert result is not None
    assert -0.10 <= result <= 0.50


def test_reverse_dcf_none_when_no_current_price():
    from advisor.research.valuation.reverse_dcf import solve_implied_growth

    dcf = DcfResult(
        symbol="X",
        current_price=0.0,
        shares_outstanding=100.0,
        net_debt=0.0,
        wacc=0.09,
        risk_free_rate=0.04,
    )
    assert solve_implied_growth(dcf) is None


# ── Multiples helpers ─────────────────────────────────────────────────────────


def test_peer_snapshot_fills_from_yfinance():
    from unittest.mock import patch

    from advisor.research.valuation.multiples import _snapshot

    fake_info = {
        "shortName": "Apple Inc.",
        "sector": "Technology",
        "marketCap": 3e12,
        "enterpriseValue": 3.1e12,
        "currentPrice": 210.0,
        "totalRevenue": 400e9,
        "trailingPE": 33.0,
        "forwardPE": 29.0,
        "freeCashflow": 100e9,
        "grossMargins": 0.46,
        "revenueGrowth": 0.06,
    }
    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.info = fake_info
        snap = _snapshot("AAPL")

    assert snap.symbol == "AAPL"
    assert snap.pe_trailing == pytest.approx(33.0)
    assert snap.gross_margin == pytest.approx(0.46)
    assert snap.ev_to_sales == pytest.approx(3.1e12 / 400e9)
