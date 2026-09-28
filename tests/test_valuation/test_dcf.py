"""The one DCF engine: its arithmetic, its refusals and its boundaries."""

from __future__ import annotations

import math

import pytest
from advisor.valuation.dcf import (
    DISCOUNT_RATE,
    GROWTH_BOUNDS,
    START_MARGIN_BOUNDS,
    TERMINAL_GROWTH,
    Margin,
    Path,
    faded_growth,
    gordon_multiple,
    implied_growth,
    implied_margin,
    market_ev,
    per_share,
    project,
    scenario_growth,
    steady_state_margins,
    value_range,
)


class TestProjection:
    def test_a_flat_business_matches_the_closed_form(self):
        """No growth, a constant margin: an annuity plus a discounted perpetuity."""
        r, g, revenue, m = 0.10, 0.03, 1000.0, 0.20
        p = project(revenue, m, Path(0.0, 0.0, m, r, g))
        fcf = revenue * m
        annuity = fcf * (1 - (1 + r) ** -10) / r
        perpetuity = fcf * (1 + g) / (r - g) / (1 + r) ** 10
        assert p.pv_fcf == pytest.approx(annuity, rel=1e-12)
        assert p.pv_terminal == pytest.approx(perpetuity, rel=1e-12)
        assert p.enterprise_value == pytest.approx(annuity + perpetuity, rel=1e-12)

    def test_growth_steps_after_year_three(self):
        p = project(100.0, 0.1, Path(0.10, 0.02, 0.1))
        assert p.revenue[2] == pytest.approx(100 * 1.1**3)
        assert p.revenue[3] == pytest.approx(100 * 1.1**3 * 1.02)

    def test_the_margin_fades_in_a_straight_line_to_year_ten(self):
        p = project(100.0, 0.0, Path(0.0, 0.0, 0.30))
        margins = [f / r for f, r in zip(p.fcf, p.revenue)]
        assert margins[0] == pytest.approx(0.03)
        assert margins[-1] == pytest.approx(0.30)

    def test_the_start_margin_is_bounded(self):
        """-400% faded over a decade sinks every year; the start is clamped."""
        wild = project(100.0, -4.0, Path(0.0, 0.0, 0.2))
        bounded = project(100.0, START_MARGIN_BOUNDS[0], Path(0.0, 0.0, 0.2))
        assert wild.enterprise_value == pytest.approx(bounded.enterprise_value)

    def test_a_business_burning_cash_in_year_ten_has_no_terminal_value(self):
        p = project(100.0, -0.2, Path(0.0, 0.0, -0.1))
        assert p.pv_terminal == 0.0
        assert p.enterprise_value < 0

    @pytest.mark.parametrize(
        "revenue,path",
        [
            (0.0, Path(0.1, 0.1, 0.2)),
            (-5.0, Path(0.1, 0.1, 0.2)),
            (100.0, Path(0.1, 0.1, 0.2, discount_rate=0.03, terminal_growth=0.03)),
            (100.0, Path(0.1, 0.1, 0.2, discount_rate=0.02, terminal_growth=0.03)),
            (100.0, Path(-1.0, 0.1, 0.2)),
            (float("nan"), Path(0.1, 0.1, 0.2)),
            (100.0, Path(float("inf"), 0.1, 0.2)),
            (100.0, Path(0.1, 0.1, float("nan"))),
        ],
    )
    def test_inputs_that_describe_no_business_are_refused(self, revenue, path):
        assert project(revenue, 0.1, path) is None

    def test_the_gordon_multiple(self):
        assert gordon_multiple(0.10, 0.03) == pytest.approx(1.03 / 0.07)
        assert gordon_multiple(0.03, 0.03) is None


class TestTheBridge:
    def test_net_cash_adds_to_equity(self):
        assert per_share(1000.0, 200.0, 10.0) == pytest.approx(120.0)

    def test_equity_is_floored_at_zero(self):
        """Limited liability: debt beyond the business is not a negative price."""
        assert per_share(100.0, -500.0, 10.0) == 0.0

    @pytest.mark.parametrize("shares", [0.0, -1.0, float("nan")])
    def test_no_share_count_is_no_price(self, shares):
        assert per_share(1000.0, 0.0, shares) is None

    @pytest.mark.parametrize("price", [0.0, -1.0, float("nan")])
    def test_no_market_ev_without_a_price(self, price):
        assert market_ev(price, 10.0, 0.0) is None

    def test_market_ev_subtracts_net_cash(self):
        assert market_ev(50.0, 100.0, 300.0) == pytest.approx(4700.0)


class TestFade:
    def test_the_step_averages_track_a_straight_line_fade(self):
        early, late = faded_growth(0.60, 0.03)
        assert early == pytest.approx(0.60 - 0.57 / 9)
        assert late == pytest.approx(0.60 - 0.57 * 2 / 3)

    def test_growth_already_at_terminal_stays_there(self):
        assert faded_growth(TERMINAL_GROWTH) == pytest.approx((TERMINAL_GROWTH, TERMINAL_GROWTH))

    def test_a_shrinking_business_fades_up_toward_terminal(self):
        early, late = faded_growth(-0.12)
        assert -0.12 < early < late < TERMINAL_GROWTH


class TestBackward:
    def test_implied_growth_reproduces_the_ev(self):
        ev = project(1000.0, 0.1, Path(0.2, 0.2, 0.25)).enterprise_value
        solved = implied_growth(ev, 1000.0, 0.1, 0.25)
        assert solved.value == pytest.approx(0.2, abs=1e-6)

    def test_a_price_beyond_the_bracket_says_which_side(self):
        above = implied_growth(1e15, 1000.0, 0.1, 0.25)
        below = implied_growth(1e-6, 1000.0, 0.1, 0.25)
        assert (above.value, above.beyond) == (None, "above")
        assert (below.value, below.beyond) == (None, "below")
        assert "above" in above.describe()

    @pytest.mark.parametrize("ev", [0.0, -1.0, float("nan")])
    def test_no_growth_answers_a_non_positive_ev(self, ev):
        assert implied_growth(ev, 1000.0, 0.1, 0.25).value is None

    def test_no_growth_answers_a_non_positive_margin(self):
        assert implied_growth(1e4, 1000.0, 0.1, 0.0).value is None

    def test_implied_margin_reproduces_the_ev(self):
        ev = project(1000.0, 0.05, Path(0.1, 0.05, 0.22)).enterprise_value
        solved = implied_margin(ev, 1000.0, 0.05, 0.1, 0.05)
        assert solved.value == pytest.approx(0.22, abs=1e-6)

    def test_implied_margin_without_a_start_applies_from_year_one(self):
        ev = project(1000.0, 0.22, Path(0.1, 0.05, 0.22)).enterprise_value
        assert implied_margin(ev, 1000.0, None, 0.1, 0.05).value == pytest.approx(0.22, abs=1e-6)

    def test_a_price_no_margin_can_justify(self):
        assert implied_margin(1e15, 1000.0, 0.1, 0.05, 0.03).beyond == "above"


class TestMargins:
    def test_bear_base_bull_are_min_median_max(self):
        got = steady_state_margins(
            [Margin(0.20, "fcf"), Margin(0.254, "median"), Margin(0.37, "nopat")]
        )
        assert [m.value for m in got] == [0.20, 0.254, 0.37]
        assert got[1].label == "median"

    def test_two_readings_take_their_midpoint_as_the_base(self):
        bear, base, bull = steady_state_margins([Margin(0.052, "median"), Margin(0.095, "nopat")])
        assert (bear.value, bull.value) == (0.052, 0.095)
        assert base.value == pytest.approx(0.0735)
        assert "median" in base.label and "nopat" in base.label

    def test_a_margin_at_or_below_zero_is_not_a_steady_state(self):
        got = steady_state_margins([Margin(-0.015, "fcf"), Margin(0.0, "x"), Margin(0.05, "m")])
        assert [m.value for m in got] == [0.05, 0.05, 0.05]

    def test_nothing_positive_is_nothing(self):
        assert steady_state_margins([Margin(-0.2, "fcf"), Margin(-2.6, "median")]) is None
        assert steady_state_margins([]) is None


class TestScenarioGrowth:
    def test_a_quarter_either_side(self):
        g = scenario_growth(0.20)
        assert g == pytest.approx({"bear": 0.15, "base": 0.20, "bull": 0.25})

    def test_never_less_than_three_points(self):
        g = scenario_growth(0.04)
        assert g["bear"] == pytest.approx(0.01) and g["bull"] == pytest.approx(0.07)

    def test_bounded_both_ways(self):
        hot = scenario_growth(1.65)
        assert hot["base"] == hot["bull"] == GROWTH_BOUNDS[1]
        cold = scenario_growth(-0.60)
        assert cold["bear"] == cold["base"] == GROWTH_BOUNDS[0]


def _range(**overrides):
    kwargs = dict(
        price=50.0,
        shares=100.0,
        net_cash=0.0,
        base_revenue=1000.0,
        start_margin=0.15,
        current_growth=0.10,
        margins=[Margin(0.12, "a"), Margin(0.15, "b"), Margin(0.18, "c")],
    )
    kwargs.update(overrides)
    return value_range(**kwargs)


class TestValueRange:
    def test_ordered_and_discounted_at_one_rate(self):
        vr = _range()
        bear, base, bull = (vr.get(n) for n in ("bear", "base", "bull"))
        assert bear.value_per_share < base.value_per_share < bull.value_per_share
        assert {s.path.discount_rate for s in vr.scenarios} == {DISCOUNT_RATE}

    def test_upside_is_value_over_price(self):
        base = _range().get("base")
        assert base.upside == pytest.approx(base.value_per_share / 50.0 - 1)

    @pytest.mark.parametrize(
        "override,reason",
        [
            ({"price": None}, "no price"),
            ({"price": 0.0}, "no price"),
            ({"price": -3.0}, "no price"),
            ({"price": float("nan")}, "no price"),
            ({"shares": None}, "no share count"),
            ({"shares": 0.0}, "no share count"),
            ({"base_revenue": 0.0}, "no revenue"),
            ({"net_cash": None}, "no balance sheet"),
            ({"current_growth": None}, "no year-over-year"),
            ({"margins": []}, "no positive margin"),
            ({"margins": [Margin(-0.2, "fcf")]}, "no positive margin"),
        ],
    )
    def test_every_missing_input_is_a_stated_refusal(self, override, reason):
        vr = _range(**override)
        assert vr.scenarios == ()
        assert reason in vr.refused

    def test_growth_exactly_at_the_bound_is_not_reported_as_bounded(self):
        assert _range(current_growth=GROWTH_BOUNDS[1]).notes == ()

    def test_growth_past_the_bound_says_so(self):
        vr = _range(current_growth=1.651)
        assert any("bounded" in n for n in vr.notes)

    def test_no_trailing_cash_flow_starts_every_case_at_the_base_steady_state(self):
        vr = _range(start_margin=None)
        assert vr.scenarios and any("year one" in n for n in vr.notes)

    def test_net_debt_beyond_the_business_values_the_equity_at_zero(self):
        vr = _range(net_cash=-1e9)
        assert all(s.value_per_share == 0.0 for s in vr.scenarios)
        assert all(math.isclose(s.upside, -1.0) for s in vr.scenarios)
