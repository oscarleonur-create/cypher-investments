"""Unit tests for the ratio engine — values verified by hand."""

from __future__ import annotations

import pytest
from advisor.research.ratios import compute_ratios


def test_profitability_ratios(clean_bundle):
    r = compute_ratios(clean_bundle)
    latest = r.latest()
    assert latest is not None

    # gross_margin = 400 / 1000 = 0.40
    assert latest.gross_margin == pytest.approx(0.40)
    # operating_margin = 200 / 1000 = 0.20
    assert latest.operating_margin == pytest.approx(0.20)
    # net_margin = 144 / 1000 = 0.144
    assert latest.net_margin == pytest.approx(0.144)


def test_return_on_capital_ratios(clean_bundle):
    r = compute_ratios(clean_bundle)
    latest = r.latest()
    # avg_assets = avg(2000, 2000) = 2000 → ROA = 144/2000 = 0.072
    assert latest.roa == pytest.approx(0.072)
    # avg_equity = 1100 → ROE = 144/1100
    assert latest.roe == pytest.approx(144 / 1100)
    # NOPAT: effective tax = 36/180 = 0.20, NOPAT = 200 * 0.80 = 160
    # invested capital = 1100 + 50 + 450 = 1600 → ROIC = 160/1600 = 0.10
    assert latest.roic == pytest.approx(0.10)


def test_liquidity_and_leverage(clean_bundle):
    r = compute_ratios(clean_bundle)
    latest = r.latest()
    # current_ratio = 500 / 300
    assert latest.current_ratio == pytest.approx(500 / 300)
    # quick_ratio = (500 - 80) / 300
    assert latest.quick_ratio == pytest.approx(420 / 300)
    # debt = 50 + 450 = 500; D/E = 500/1100
    assert latest.debt_to_equity == pytest.approx(500 / 1100)
    # D/EBITDA = 500/240
    assert latest.debt_to_ebitda == pytest.approx(500 / 240)
    # interest_coverage = 200 / 20
    assert latest.interest_coverage == pytest.approx(10.0)


def test_efficiency_and_cash_quality(clean_bundle):
    r = compute_ratios(clean_bundle)
    latest = r.latest()
    # asset_turnover = 1000 / 2000
    assert latest.asset_turnover == pytest.approx(0.5)
    # inventory_turns = 600 / 80
    assert latest.inventory_turns == pytest.approx(600 / 80)
    # DSO = 100 / 1000 * 365
    assert latest.dso == pytest.approx(36.5)
    # fcf_margin = 150 / 1000
    assert latest.fcf_margin == pytest.approx(0.15)
    # capex_intensity = |-50| / 1000
    assert latest.capex_intensity == pytest.approx(0.05)
    # fcf_to_net_income = 150 / 144
    assert latest.fcf_to_net_income == pytest.approx(150 / 144)


def test_share_count_cagr_zero_when_flat(clean_bundle):
    r = compute_ratios(clean_bundle)
    assert r.share_count_cagr_3y == pytest.approx(0.0, abs=1e-9)


def test_share_count_cagr_positive(income_factory, balance_factory, cashflow_factory):
    from advisor.research.models import StatementBundle

    # 100 → 110 → 121 → 133.1 (10% CAGR)
    years = [2024, 2023, 2022, 2021]
    shares = [133.1, 121.0, 110.0, 100.0]
    bundle = StatementBundle(
        symbol="DIL",
        income=[income_factory(y, shares_diluted=s) for y, s in zip(years, shares)],
        balance=[balance_factory(y) for y in years],
        cashflow=[cashflow_factory(y) for y in years],
    )
    r = compute_ratios(bundle)
    assert r.share_count_cagr_3y == pytest.approx(0.10, rel=1e-3)


def test_missing_inputs_return_none_not_raise(income_factory, balance_factory, cashflow_factory):
    from advisor.research.models import StatementBundle

    bundle = StatementBundle(
        symbol="GAP",
        income=[income_factory(2024, revenue=None, operating_income=None, net_income=None)],
        balance=[balance_factory(2024, total_assets=None)],
        cashflow=[cashflow_factory(2024, operating_cash_flow=None, free_cash_flow=None)],
    )
    r = compute_ratios(bundle)
    latest = r.latest()
    assert latest.gross_margin is None
    assert latest.operating_margin is None
    assert latest.roa is None
    assert latest.fcf_margin is None


def test_fcf_derived_when_only_ocf_and_capex_given(
    income_factory, balance_factory, cashflow_factory
):
    from advisor.research.models import StatementBundle

    bundle = StatementBundle(
        symbol="DRV",
        income=[income_factory(2024)],
        balance=[balance_factory(2024)],
        cashflow=[cashflow_factory(2024, operating_cash_flow=300, capex=-100, free_cash_flow=None)],
    )
    r = compute_ratios(bundle)
    # fcf = 300 - 100 = 200 → fcf_margin = 200/1000
    assert r.latest().fcf_margin == pytest.approx(0.20)
