"""Each detector tested in isolation against a synthetic 2-yr bundle."""

from __future__ import annotations

from advisor.research.models import StatementBundle
from advisor.research.ratios import compute_ratios
from advisor.research.red_flags import detect_red_flags


def _two_year_bundle(income_factory, balance_factory, cashflow_factory, **kwargs):
    """Build a 2-yr bundle where overrides apply to FY2024 (newest)."""
    inc_kwargs = kwargs.get("inc", {})
    bs_kwargs = kwargs.get("bs", {})
    cf_kwargs = kwargs.get("cf", {})
    return StatementBundle(
        symbol="TEST",
        income=[income_factory(2024, **inc_kwargs), income_factory(2023)],
        balance=[balance_factory(2024, **bs_kwargs), balance_factory(2023)],
        cashflow=[cashflow_factory(2024, **cf_kwargs), cashflow_factory(2023)],
    )


def test_clean_bundle_has_no_flags(clean_bundle):
    ratios = compute_ratios(clean_bundle)
    rf = detect_red_flags(clean_bundle, ratios)
    assert rf.flags == []
    assert rf.high_severity_count == 0


def test_dso_drift_triggers(income_factory, balance_factory, cashflow_factory):
    # Revenue flat (1000 → 1000), AR up 20% (100 → 120) → flag
    bundle = _two_year_bundle(
        income_factory,
        balance_factory,
        cashflow_factory,
        bs={"accounts_receivable": 120.0},
    )
    rf = detect_red_flags(bundle, compute_ratios(bundle))
    codes = {f.code for f in rf.flags}
    assert "DSO_DRIFT" in codes


def test_inventory_drift_triggers(income_factory, balance_factory, cashflow_factory):
    # Revenue flat, inventory up 25% (80 → 100)
    bundle = _two_year_bundle(
        income_factory,
        balance_factory,
        cashflow_factory,
        bs={"inventory": 100.0},
    )
    rf = detect_red_flags(bundle, compute_ratios(bundle))
    assert any(f.code == "INVENTORY_DRIFT" for f in rf.flags)


def test_share_dilution_triggers(income_factory, balance_factory, cashflow_factory):
    years = [2024, 2023, 2022, 2021]
    # 100 → 110 → 121 → 133.1 (10% CAGR, well above 3% threshold)
    shares = [133.1, 121.0, 110.0, 100.0]
    bundle = StatementBundle(
        symbol="DIL",
        income=[income_factory(y, shares_diluted=s) for y, s in zip(years, shares)],
        balance=[balance_factory(y) for y in years],
        cashflow=[cashflow_factory(y) for y in years],
    )
    rf = detect_red_flags(bundle, compute_ratios(bundle))
    flag = next((f for f in rf.flags if f.code == "SHARE_DILUTION"), None)
    assert flag is not None
    assert flag.severity.value == "HIGH"  # >7% CAGR


def test_fcf_quality_gap_triggers(income_factory, balance_factory, cashflow_factory):
    years = [2024, 2023, 2022]
    # NI=144, FCF=50 → FCF/NI = 0.35 across 3 periods
    bundle = StatementBundle(
        symbol="GAP",
        income=[income_factory(y) for y in years],
        balance=[balance_factory(y) for y in years],
        cashflow=[cashflow_factory(y, free_cash_flow=50, operating_cash_flow=100) for y in years],
    )
    rf = detect_red_flags(bundle, compute_ratios(bundle))
    assert any(f.code == "FCF_QUALITY_GAP" for f in rf.flags)


def test_leverage_step_up_triggers(income_factory, balance_factory, cashflow_factory):
    # FY2023: D=500, EBITDA=240 → 2.08x. FY2024: D=900, EBITDA=240 → 3.75x
    bundle = _two_year_bundle(
        income_factory,
        balance_factory,
        cashflow_factory,
        bs={"long_term_debt": 850.0},  # 50 + 850 = 900 total
    )
    rf = detect_red_flags(bundle, compute_ratios(bundle))
    assert any(f.code == "LEVERAGE_STEP_UP" for f in rf.flags)


def test_margin_compression_triggers(income_factory, balance_factory, cashflow_factory):
    # FY2023 op_margin = 0.20; FY2024 op_income = 150 → margin = 0.15 (-500 bps)
    bundle = _two_year_bundle(
        income_factory,
        balance_factory,
        cashflow_factory,
        inc={"operating_income": 150.0},
    )
    rf = detect_red_flags(bundle, compute_ratios(bundle))
    assert any(f.code == "MARGIN_COMPRESSION" for f in rf.flags)
