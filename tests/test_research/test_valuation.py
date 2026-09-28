"""Tests for DCF engine, multiples, and reverse-DCF."""

from __future__ import annotations

from datetime import date

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
    from advisor.research.valuation.dcf import compute_dcf_scenario

    a = _make_assumptions()
    scenario = compute_dcf_scenario(
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
    from advisor.research.valuation.dcf import compute_dcf_scenario

    a = _make_assumptions()
    scenario = compute_dcf_scenario(
        a, base_revenue=1000.0, seed_fcf=150.0, net_debt=0.0, shares=100.0, current_price=50.0
    )
    assert scenario.terminal_value_gordon is not None and scenario.terminal_value_gordon > 0
    assert scenario.enterprise_value > 0
    assert scenario.implied_price > 0


def test_dcf_bull_price_above_base_above_bear():
    from advisor.research.valuation.dcf import compute_dcf_scenario

    kwargs = dict(
        base_revenue=1000.0, seed_fcf=150.0, net_debt=0.0, shares=100.0, current_price=50.0
    )
    base = compute_dcf_scenario(_make_assumptions(scenario="base"), **kwargs)
    bull = compute_dcf_scenario(
        _make_assumptions(
            scenario="bull", revenue_growth_yr1_3=0.15, target_fcf_margin=0.18, wacc=0.08
        ),
        **kwargs,
    )
    bear = compute_dcf_scenario(
        _make_assumptions(
            scenario="bear", revenue_growth_yr1_3=0.04, target_fcf_margin=0.10, wacc=0.10
        ),
        **kwargs,
    )

    assert bull.implied_price > base.implied_price > bear.implied_price


def test_dcf_net_debt_reduces_equity_value():
    from advisor.research.valuation.dcf import compute_dcf_scenario

    a = _make_assumptions()
    no_debt = compute_dcf_scenario(
        a, base_revenue=1000.0, seed_fcf=150.0, net_debt=0.0, shares=100.0, current_price=50.0
    )
    with_debt = compute_dcf_scenario(
        a, base_revenue=1000.0, seed_fcf=150.0, net_debt=200.0, shares=100.0, current_price=50.0
    )

    assert no_debt.implied_price > with_debt.implied_price
    assert no_debt.equity_value - with_debt.equity_value == pytest.approx(200.0, rel=0.01)


def test_an_exit_multiple_is_ignored_and_the_perpetuity_values_the_tail():
    """The exit multiple was applied to an EBITDA of 25% of revenue for every
    company. The engine values the tail as a Gordon perpetuity only."""
    from advisor.research.valuation.dcf import compute_dcf_scenario

    kwargs = dict(
        base_revenue=1000.0, seed_fcf=150.0, net_debt=0.0, shares=100.0, current_price=50.0
    )
    with_exit = compute_dcf_scenario(_make_assumptions(terminal_exit_multiple=20.0), **kwargs)
    without = compute_dcf_scenario(_make_assumptions(), **kwargs)
    assert with_exit.terminal_value_exit is None
    assert with_exit.enterprise_value == pytest.approx(without.enterprise_value)


# ── The DCF built from the company's own figures ─────────────────────────────


def test_build_dcf_values_bear_below_base_below_bull():
    from advisor.research.valuation.dcf import build_dcf

    from tests.test_research.dcf_figures import figures

    dcf = build_dcf("TEST", figures=figures())
    assert dcf.note == ""
    assert dcf.bear.implied_price < dcf.base.implied_price < dcf.bull.implied_price
    assert dcf.shares_outstanding == 100e6 and dcf.net_debt == pytest.approx(300e6)
    assert dcf.base_revenue == 1000e6 and dcf.seed_fcf == pytest.approx(150e6)
    # The margins are the company's own: min, median, max.
    margins = [s.assumptions.target_fcf_margin for s in (dcf.bear, dcf.base, dcf.bull)]
    assert margins == pytest.approx([0.12, 0.15, 0.18])
    assert all(s.assumptions.wacc == 0.10 for s in (dcf.bear, dcf.base, dcf.bull))


def test_the_sliders_replay_the_base_scenario_exactly():
    """What the workstation recomputes from the report is what was stored."""
    from advisor.research.models import ResearchReport
    from advisor.research.valuation.dcf import (
        build_dcf,
        compute_dcf_scenario,
        dcf_inputs_from_report,
    )

    from tests.test_research.dcf_figures import figures

    dcf = build_dcf("TEST", figures=figures())
    inputs = dcf_inputs_from_report(ResearchReport(symbol="TEST", as_of=date.today(), dcf=dcf))
    replay = compute_dcf_scenario(dcf.base.assumptions, *inputs)
    assert replay.implied_price == pytest.approx(dcf.base.implied_price, rel=1e-12)


@pytest.mark.parametrize(
    "missing,reason",
    [
        ({"price": None}, "no price"),
        ({"price": 0.0}, "no price"),
        ({"shares": None}, "no share count"),
        ({"shares": 0.0}, "no share count"),
        ({"revenue_base": None}, "no revenue"),
        ({"net_cash": None}, "no balance sheet"),
        ({"revenue_growth": None}, "no year-over-year revenue comparison"),
        ({"margins": []}, "no positive margin"),
    ],
)
def test_a_missing_input_is_a_stated_refusal_never_a_default(missing, reason):
    """Rate-limited, the old DCF defaulted shares to 1.0 and price to 0 and
    published JBL at $0.00. A missing input now yields no scenarios and a note."""
    from advisor.research.valuation.dcf import build_dcf

    from tests.test_research.dcf_figures import figures

    dcf = build_dcf("TEST", figures=figures(**missing))
    assert dcf.base is None and dcf.bull is None and dcf.bear is None
    assert reason in dcf.note
    assert dcf.shares_outstanding != 1.0


def test_only_burning_cash_margins_refuse_a_value():
    from advisor.research.valuation.dcf import build_dcf
    from advisor.valuation.models import OwnMargin

    from tests.test_research.dcf_figures import figures

    burning = [
        OwnMargin(kind="fcf", value=-0.2, label="FCF"),
        OwnMargin(kind="nopat", value=-0.1, label="op"),
    ]
    dcf = build_dcf("TEST", figures=figures(margins=burning, start_margin=-0.2))
    assert dcf.base is None
    assert "no positive margin" in dcf.note


def test_a_report_cached_with_the_old_defaults_is_not_replayed():
    """shares=1.0 and price=0 were the yfinance fallbacks; never recompute on them."""
    from advisor.research.models import ResearchReport
    from advisor.research.valuation.dcf import build_dcf, dcf_inputs_from_report

    from tests.test_research.dcf_figures import figures

    dcf = build_dcf("TEST", figures=figures())
    for broken in ({"shares_outstanding": 1.0}, {"current_price": 0.0}, {"base_revenue": 0.0}):
        report = ResearchReport(
            symbol="TEST", as_of=date.today(), dcf=dcf.model_copy(update=broken)
        )
        assert dcf_inputs_from_report(report) is None


# ── Reverse-DCF ──────────────────────────────────────────────────────────────


def test_reverse_dcf_reproduces_the_price():
    """Run forward at the solved growth, the engine returns today's price."""
    from advisor.research.valuation.dcf import build_dcf, compute_dcf_scenario
    from advisor.research.valuation.reverse_dcf import solve_implied_growth

    from tests.test_research.dcf_figures import figures

    dcf = build_dcf("TEST", figures=figures())
    g = solve_implied_growth(dcf)
    assert g is not None
    a = dcf.base.assumptions.model_copy(
        update={"revenue_growth_yr1_3": g, "revenue_growth_yr4_10": g}
    )
    forward = compute_dcf_scenario(
        a, dcf.base_revenue, dcf.seed_fcf, dcf.net_debt, dcf.shares_outstanding, 50.0
    )
    assert forward.implied_price == pytest.approx(50.0, rel=1e-4)


def test_reverse_dcf_is_none_beyond_the_bracket_not_the_edge():
    """The old solver returned +50% when the price needed more; that is a bound
    dressed as an answer."""
    from advisor.research.valuation.dcf import build_dcf
    from advisor.research.valuation.reverse_dcf import solve_implied_growth

    from tests.test_research.dcf_figures import figures

    dcf = build_dcf("TEST", figures=figures())
    assert solve_implied_growth(dcf.model_copy(update={"current_price": 1e9})) is None


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
