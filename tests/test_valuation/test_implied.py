"""Implied expectations: the arithmetic, and what it refuses to do.

The headline answers what would have to happen for the price to make sense,
because that answer is falsifiable. Since 2026-09-27 it is discounted: the
first version asked what would make the business worth today's price in ten
years — a 0% return — and understated every requirement.
"""

from __future__ import annotations

from datetime import date

import pytest
from advisor.valuation.implied import (
    GENERIC_MARGINS,
    build_snapshot,
    implied_expectations,
    undiscounted_expectations,
)
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

    def test_the_base_case_cagr_is_discounted(self):
        """42.4% a year at 25% FCF, 10% discount, 3% terminal — against the
        25.8% the undiscounted arithmetic reported for the same price."""
        snapshot = build_snapshot(Fundamentals(**SPCX), 151.21)
        assert snapshot.method == "dcf"
        assert snapshot.discount_rate == 0.10 and snapshot.terminal_growth == 0.03
        assert snapshot.base_case().implied_cagr == pytest.approx(0.4235, abs=0.001)

    def test_the_old_rows_reading_is_reproduced_exactly(self):
        """Rows stored before the engine discounted still read as they did."""
        old = undiscounted_expectations(1.933e12, 31.256e9, terminal_multiple=25, fcf_margin=0.25)
        assert old.implied_cagr == pytest.approx(0.258, abs=0.001)
        assert old.discount_rate is None
        assert "25x terminal" in old.describe()

    def test_discounting_always_demands_more_than_the_undiscounted_reading(self):
        snapshot = build_snapshot(Fundamentals(**SPCX), 151.21)
        old = undiscounted_expectations(
            snapshot.enterprise_value, snapshot.revenue_base, terminal_multiple=25, fcf_margin=0.25
        )
        assert snapshot.base_case().implied_cagr > old.implied_cagr

    def test_a_quarter_is_annualised_and_a_year_is_not(self):
        quarterly = Fundamentals(**SPCX)
        annual = Fundamentals(**{**SPCX, "fiscal_period": "FY", "period_start": date(2025, 7, 1)})
        assert quarterly.revenue_runrate == pytest.approx(4 * SPCX["revenue"])
        assert annual.revenue_runrate == pytest.approx(SPCX["revenue"])

    def test_without_a_start_date_the_form_decides(self):
        undated = Fundamentals(**{**SPCX, "period_start": None})
        annual = Fundamentals(**{**SPCX, "period_start": None, "fiscal_period": "FY"})
        assert undated.revenue_runrate == pytest.approx(4 * SPCX["revenue"])
        assert annual.revenue_runrate == pytest.approx(SPCX["revenue"])

    def test_a_quarter_inside_a_10k_is_still_a_quarter(self):
        """The period's own length decides, not the form it came in: a 10-K's
        shortest undimensioned period can be the fourth quarter."""
        q4 = Fundamentals(**{**SPCX, "fiscal_period": "FY"})
        assert q4.revenue_runrate == pytest.approx(4 * SPCX["revenue"])

    def test_a_fresh_start_nine_months_is_scaled_to_a_year(self):
        """WOLF's FY2026 10-K: $468.3M over 2025-09-30 to 2026-06-28 (272 days)
        after leaving bankruptcy. Read as a year it lost a quarter."""
        wolf = Fundamentals(
            **{
                **SPCX,
                "symbol": "WOLF",
                "fiscal_period": "FY",
                "revenue": 468_300_000.0,
                "period_start": date(2025, 9, 30),
                "period_end": date(2026, 6, 28),
            }
        )
        assert wolf.period_days == 272
        assert wolf.revenue_runrate == pytest.approx(468.3e6 * 365 / 272)
        assert "272-day period" in wolf.runrate_label()

    def test_a_start_after_the_end_is_no_period(self):
        broken = Fundamentals(**{**SPCX, "period_start": date(2026, 7, 1)})
        assert broken.period_days is None
        assert broken.revenue_runrate == pytest.approx(4 * SPCX["revenue"])

    def test_a_higher_price_demands_more_growth(self):
        low = build_snapshot(Fundamentals(**SPCX), 100.0).base_case().implied_cagr
        high = build_snapshot(Fundamentals(**SPCX), 200.0).base_case().implied_cagr
        assert high > low

    def test_a_more_generous_margin_demands_less_growth(self):
        ev, revenue = 1.933e12, 31.256e9
        generous = implied_expectations(ev, revenue, fcf_margin=0.30)
        demanding = implied_expectations(ev, revenue, fcf_margin=0.20)
        assert generous.implied_cagr < demanding.implied_cagr

    def test_a_higher_discount_rate_demands_more_growth(self):
        ev, revenue = 1.933e12, 31.256e9
        low = implied_expectations(ev, revenue, fcf_margin=0.25, discount_rate=0.08)
        high = implied_expectations(ev, revenue, fcf_margin=0.25, discount_rate=0.12)
        assert high.implied_cagr > low.implied_cagr

    def test_a_cash_burning_start_demands_more_than_a_steady_one(self):
        """Today's margin fades to the steady state; the years spent burning
        cash have to be paid for by more growth."""
        ev, revenue = 1.933e12, 31.256e9
        steady = implied_expectations(ev, revenue, fcf_margin=0.25)
        burning = implied_expectations(ev, revenue, fcf_margin=0.25, start_margin=-0.30)
        assert burning.implied_cagr > steady.implied_cagr

    def test_the_answer_reproduces_the_price(self):
        """Run forward at the solved growth, the engine returns today's EV."""
        from advisor.valuation.dcf import Path, project

        ev, revenue = 1.933e12, 31.256e9
        answer = implied_expectations(ev, revenue, fcf_margin=0.25, start_margin=0.05)
        g = answer.implied_cagr
        forward = project(revenue, 0.05, Path(g, g, 0.25, 0.10, 0.03))
        assert forward.enterprise_value == pytest.approx(ev, rel=1e-5)
        assert answer.required_revenue == pytest.approx(forward.revenue[-1], rel=1e-9)

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
        assert implied_expectations(1e12, 0.0, fcf_margin=0.25) is None

    def test_a_negative_enterprise_value_yields_nothing(self):
        """More net cash than market cap: the model has nothing to say."""
        assert implied_expectations(-1e9, 1e9, fcf_margin=0.25) is None

    @pytest.mark.parametrize("margin", [0.0, -0.1])
    def test_a_margin_at_or_below_zero_is_refused(self, margin):
        assert implied_expectations(1e12, 1e10, fcf_margin=margin) is None

    def test_a_discount_rate_at_the_terminal_growth_is_refused(self):
        """A perpetuity growing as fast as it is discounted is infinite."""
        assert implied_expectations(1e12, 1e10, fcf_margin=0.25, discount_rate=0.03) is None

    def test_a_price_beyond_any_growth_in_the_bracket_is_none_not_the_edge(self):
        """$1,000tn of EV on $1bn of revenue: no growth up to 150%/yr covers it."""
        assert implied_expectations(1e18, 1e9, fcf_margin=0.25) is None

    def test_a_snapshot_with_no_own_margins_refuses_a_value_range(self):
        """A filing alone carries no margins: the requirement is computed, the
        value range is refused with its reason — never a default margin."""
        snapshot = build_snapshot(Fundamentals(**SPCX), 151.21)
        assert snapshot.value == []
        assert snapshot.value_refused


class TestStaleness:
    def test_a_recent_filing_is_not_stale(self):
        snapshot = build_snapshot(Fundamentals(**SPCX), 151.21, asof=date(2026, 8, 15))
        assert snapshot.is_stale(date(2026, 8, 15)) is False

    def test_an_annual_filing_two_quarters_old_is_stale(self):
        """NBIS files 20-F annually; its figures were 256 days old in September."""
        old = Fundamentals(
            **{
                **SPCX,
                "period_start": date(2025, 1, 1),
                "period_end": date(2025, 12, 31),
                "fiscal_period": "FY",
            }
        )
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
        assert len(snapshot.scenarios) == len(GENERIC_MARGINS) == 3

    def test_the_base_case_is_the_generic_25_percent_margin(self):
        """The quantity a thesis rule tests (user decision, 2026-09-25)."""
        assert build_snapshot(Fundamentals(**SPCX), 151.21).base_case().fcf_margin == 0.25

    def test_the_base_case_is_the_middle_reading(self):
        snapshot = build_snapshot(Fundamentals(**SPCX), 151.21)
        cagrs = sorted(s.implied_cagr for s in snapshot.scenarios)
        assert snapshot.base_case().implied_cagr == cagrs[1]

    def test_every_scenario_states_the_assumptions_that_produced_it(self):
        """A CAGR without its margin, discount rate and terminal means nothing."""
        snapshot = build_snapshot(Fundamentals(**SPCX), 151.21)
        for scenario in snapshot.scenarios:
            assert scenario.terminal_multiple == pytest.approx(1.03 / 0.07, abs=0.01)
            assert 0 < scenario.fcf_margin <= 1
            text = scenario.describe()
            assert "discounted at 10%" in text and "3% terminal growth" in text

    def test_the_inputs_travel_with_the_answer(self):
        snapshot = build_snapshot(Fundamentals(**SPCX), 151.21)
        assert snapshot.source_accession == SPCX["source_accession"]
        assert snapshot.period_end == SPCX["period_end"]
        assert snapshot.revenue_base == pytest.approx(4 * SPCX["revenue"])
        assert "quarter to 2026-06-30 × 4" in snapshot.revenue_base_label
