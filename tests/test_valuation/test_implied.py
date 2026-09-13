"""Implied expectations: the arithmetic, and what it refuses to do.

This module never produces a fair value. It answers the other question — what
would have to happen for the price to make sense — because that answer is
falsifiable and a valuation opinion is not.
"""

from __future__ import annotations

from datetime import date

import pytest
from advisor.valuation.implied import DEFAULT_SCENARIOS, build_snapshot, implied_expectations
from advisor.valuation.models import Fundamentals

# Every figure verified against SPCX's 10-Q filed 2026-08-04.
SPCX = dict(
    symbol="SPCX",
    source_accession="0001628280-26-052535",
    period_end=date(2026, 6, 30),
    period_start=date(2026, 4, 1),
    fiscal_period="Q",
    revenue=7_814_000_000.0,
    operating_income=-143_000_000.0,
    net_income=-541_000_000.0,
    cash=93_522_000_000.0,
    marketable_securities=6_487_000_000.0,
    total_debt=39_364_000_000.0,
    shares_outstanding=13_181_779_945.0,
)


class TestTheArithmetic:
    def test_the_real_spcx_snapshot_reproduces_the_filing(self):
        snapshot = build_snapshot(Fundamentals(**SPCX), 151.21)
        assert snapshot.market_cap == pytest.approx(1.993e12, rel=0.01)
        assert snapshot.net_cash == pytest.approx(60.645e9, rel=0.01)
        assert snapshot.enterprise_value == pytest.approx(1.933e12, rel=0.01)
        assert snapshot.ev_to_revenue == pytest.approx(61.8, abs=0.2)

    def test_the_base_case_cagr(self):
        snapshot = build_snapshot(Fundamentals(**SPCX), 151.21)
        assert snapshot.base_case().implied_cagr == pytest.approx(0.258, abs=0.005)

    def test_a_quarter_is_annualised_and_a_year_is_not(self):
        quarterly = Fundamentals(**SPCX)
        annual = Fundamentals(**{**SPCX, "fiscal_period": "FY"})
        assert quarterly.revenue_runrate == pytest.approx(4 * SPCX["revenue"])
        assert annual.revenue_runrate == pytest.approx(SPCX["revenue"])

    def test_a_higher_price_demands_more_growth(self):
        low = build_snapshot(Fundamentals(**SPCX), 100.0).base_case().implied_cagr
        high = build_snapshot(Fundamentals(**SPCX), 200.0).base_case().implied_cagr
        assert high > low

    def test_a_more_generous_terminal_multiple_demands_less_growth(self):
        ev, revenue = 1.933e12, 31.256e9
        generous = implied_expectations(ev, revenue, terminal_multiple=30, fcf_margin=0.30)
        demanding = implied_expectations(ev, revenue, terminal_multiple=20, fcf_margin=0.20)
        assert generous.implied_cagr < demanding.implied_cagr

    def test_net_cash_lowers_the_enterprise_value_below_the_market_cap(self):
        snapshot = build_snapshot(Fundamentals(**SPCX), 151.21)
        assert snapshot.enterprise_value < snapshot.market_cap

    def test_a_net_debt_company_has_an_ev_above_its_market_cap(self):
        indebted = Fundamentals(
            **{**SPCX, "cash": 1e9, "marketable_securities": 0.0, "total_debt": 40e9}
        )
        snapshot = build_snapshot(indebted, 151.21)
        assert snapshot.enterprise_value > snapshot.market_cap


class TestRefusals:
    """A valuation built on a guessed input is worse than no valuation."""

    @pytest.mark.parametrize("field", ["revenue", "cash", "shares_outstanding"])
    def test_a_missing_essential_input_produces_nothing(self, field):
        broken = Fundamentals(**{**SPCX, field: None})
        broken.missing.append(field)
        assert build_snapshot(broken, 151.21) is None

    def test_a_zero_or_negative_price_produces_nothing(self):
        assert build_snapshot(Fundamentals(**SPCX), 0) is None
        assert build_snapshot(Fundamentals(**SPCX), -5) is None

    def test_a_pre_revenue_company_cannot_be_valued_this_way(self):
        """CCXI is clinical-stage: no revenue concept exists in its filing."""
        pre_revenue = Fundamentals(**{**SPCX, "revenue": None})
        pre_revenue.missing.append("revenue")
        assert build_snapshot(pre_revenue, 151.21) is None

    def test_zero_revenue_yields_no_scenario_rather_than_infinity(self):
        assert implied_expectations(1e12, 0.0, terminal_multiple=25, fcf_margin=0.25) is None

    def test_a_negative_enterprise_value_yields_nothing(self):
        """More net cash than market cap: the model has nothing to say."""
        assert implied_expectations(-1e9, 1e9, terminal_multiple=25, fcf_margin=0.25) is None

    @pytest.mark.parametrize(
        "multiple,margin,years", [(0, 0.25, 10), (25, 0, 10), (25, 1.5, 10), (25, 0.25, 0)]
    )
    def test_nonsense_assumptions_are_refused(self, multiple, margin, years):
        assert (
            implied_expectations(
                1e12, 1e10, terminal_multiple=multiple, fcf_margin=margin, years=years
            )
            is None
        )


class TestStaleness:
    def test_a_recent_filing_is_not_stale(self):
        snapshot = build_snapshot(Fundamentals(**SPCX), 151.21, asof=date(2026, 8, 15))
        assert snapshot.is_stale(date(2026, 8, 15)) is False

    def test_an_annual_filing_two_quarters_old_is_stale(self):
        """NBIS files 20-F annually; its figures were 256 days old in September."""
        old = Fundamentals(**{**SPCX, "period_end": date(2025, 12, 31), "fiscal_period": "FY"})
        snapshot = build_snapshot(old, 224.55, asof=date(2026, 9, 13))
        assert snapshot.is_stale(date(2026, 9, 13)) is True
        assert snapshot.period_age_days(date(2026, 9, 13)) == 256

    def test_exactly_at_the_staleness_boundary_is_not_yet_stale(self):
        from advisor.valuation.models import ValuationSnapshot

        end = date(2026, 5, 16)
        today = date(2026, 9, 13)
        assert (today - end).days == ValuationSnapshot.STALE_AFTER_DAYS
        snapshot = build_snapshot(Fundamentals(**{**SPCX, "period_end": end}), 151.21)
        assert snapshot.is_stale(today) is False


class TestScenarios:
    def test_three_readings_are_produced_not_one_false_precision(self):
        snapshot = build_snapshot(Fundamentals(**SPCX), 151.21)
        assert len(snapshot.scenarios) == len(DEFAULT_SCENARIOS) == 3

    def test_the_base_case_is_the_middle_reading(self):
        snapshot = build_snapshot(Fundamentals(**SPCX), 151.21)
        cagrs = sorted(s.implied_cagr for s in snapshot.scenarios)
        assert snapshot.base_case().implied_cagr == cagrs[1]

    def test_every_scenario_states_the_assumptions_that_produced_it(self):
        """A CAGR without its terminal multiple and margin means nothing."""
        snapshot = build_snapshot(Fundamentals(**SPCX), 151.21)
        for scenario in snapshot.scenarios:
            assert scenario.terminal_multiple > 0
            assert 0 < scenario.fcf_margin <= 1
            assert str(scenario.terminal_multiple).rstrip("0").rstrip(".") in scenario.describe()

    def test_the_inputs_travel_with_the_answer(self):
        snapshot = build_snapshot(Fundamentals(**SPCX), 151.21)
        assert snapshot.source_accession == SPCX["source_accession"]
        assert snapshot.period_end == SPCX["period_end"]
