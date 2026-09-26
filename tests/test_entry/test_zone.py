"""The relative entry zone and the absolute context."""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from advisor.entry.zone import (
    MAX_TTM_AGE_DAYS,
    MIN_OBSERVATIONS,
    SHORT_MIN_OBSERVATIONS,
    Scenario,
    absolute_context,
    consensus_growth,
    relative_zone,
    required_at,
)
from advisor.valuation.history import Point, Series
from advisor.valuation.models import ImpliedExpectations, ValuationSnapshot

TODAY = date(2026, 9, 25)


def quarterly(value, first=date(2023, 1, 1), last=TODAY):
    """A TTM point every 90 days, each known on its end: always fresh."""
    out, d = [], first
    while d <= last:
        out.append(Point(end=d, value=value, known=d))
        d += timedelta(days=90)
    return out


def series(rev=100.0, shares=10.0, broken_after=None, shares_path=None, revenue=None):
    start = date(2023, 1, 1)
    return Series(
        symbol="X",
        revenue_ttm=revenue or quarterly(rev),
        shares=shares_path or [Point(end=start, value=shares, known=start)],
        broken_after=broken_after,
    )


def closes(prices):
    """One close per calendar day ending today (weekends included; fine for a median)."""
    n = len(prices)
    return [(TODAY - timedelta(days=n - 1 - i), p) for i, p in enumerate(prices)]


class TestRelativeZone:
    def test_median_percentile_and_top(self):
        hist = closes([10.0] * 300 + [20.0] * 300)  # P/S 1x then 2x
        z = relative_zone(hist, series(), TODAY, 12.0)
        assert z.ps_now == pytest.approx(1.2)
        assert z.median == pytest.approx(1.5)  # 1x and 2x, half each
        assert z.top == pytest.approx(15.0)
        assert z.in_zone and z.distance == pytest.approx(12 / 15 - 1)
        assert z.percentile == pytest.approx(0.5)

    def test_exactly_at_the_median_is_in_zone(self):
        z = relative_zone(closes([10.0] * 400), series(), TODAY, 10.0)
        assert z.in_zone and z.distance == pytest.approx(0.0)

    def test_above_the_median(self):
        z = relative_zone(closes([10.0] * 400), series(), TODAY, 13.0)
        assert not z.in_zone and z.distance == pytest.approx(0.3)

    def test_too_little_history(self):
        hist = closes([10.0] * (SHORT_MIN_OBSERVATIONS - 1))
        assert relative_zone(hist, series(), TODAY, 10) is None

    def test_under_a_year_is_a_short_zone(self):
        """NBIS: five Yahoo quarters. Shown, flagged, never a position leg."""
        z = relative_zone(closes([10.0] * (MIN_OBSERVATIONS - 1)), series(), TODAY, 10)
        assert z is not None and z.short
        assert z.observations == MIN_OBSERVATIONS - 1

    def test_exactly_the_short_floor(self):
        z = relative_zone(closes([10.0] * SHORT_MIN_OBSERVATIONS), series(), TODAY, 10)
        assert z is not None and z.short

    def test_a_full_year_is_not_short(self):
        z = relative_zone(closes([10.0] * MIN_OBSERVATIONS), series(), TODAY, 10)
        assert z is not None and not z.short

    def test_stale_revenue_days_are_excluded(self):
        """The last TTM before Yahoo's window carried forward a year inflates P/S.

        Old sales (100) paired with prices from a year when sales tripled would
        put the median at 3x; only days with fresh TTM may count.
        """
        old_end = TODAY - timedelta(days=700)  # fresh only until day -500
        revenue = [Point(end=old_end, value=100.0, known=old_end)] + quarterly(
            300.0, first=TODAY - timedelta(days=180)
        )
        hist = closes([30.0] * 500)
        z = relative_zone(hist, series(revenue=revenue), TODAY, 30.0)
        assert z is not None
        assert z.median == pytest.approx(1.0)  # 30 * 10 / 300, never 30 * 10 / 100
        assert z.short  # the stale days did not count toward a full year

    def test_stale_revenue_today_means_no_zone(self):
        """A name that stopped reporting: today's P/S would be on old sales."""
        end = TODAY - timedelta(days=MAX_TTM_AGE_DAYS + 1)
        revenue = quarterly(100.0, last=end)
        assert relative_zone(closes([10.0] * 400), series(revenue=revenue), TODAY, 10) is None

    def test_revenue_exactly_at_the_age_limit_counts(self):
        end = TODAY - timedelta(days=MAX_TTM_AGE_DAYS)
        revenue = [
            Point(
                end=date(2023, 1, 1) + timedelta(days=90 * i),
                value=100.0,
                known=date(2023, 1, 1) + timedelta(days=90 * i),
            )
            for i in range(20)
        ]
        revenue = [p for p in revenue if p.end < end] + [Point(end=end, value=100.0, known=end)]
        z = relative_zone(closes([10.0] * 400), series(revenue=revenue), TODAY, 10)
        assert z is not None

    def test_source_is_carried(self):
        s = series().model_copy(update={"source": "yfinance statements"})
        z = relative_zone(closes([10.0] * 400), s, TODAY, 10)
        assert z.source == "yfinance statements"

    def test_only_two_years_count(self):
        old = [(TODAY - timedelta(days=900 + i), 1000.0) for i in range(100)]
        z = relative_zone(old + closes([10.0] * 400), series(), TODAY, 10.0)
        assert z.median == pytest.approx(1.0)

    def test_a_break_trims_the_window(self):
        brk = TODAY - timedelta(days=100)
        assert relative_zone(closes([10.0] * 600), series(broken_after=brk), TODAY, 10) is None

    def test_share_count_as_known_each_day(self):
        """A raise mid-window: older days use the old count, not today's."""
        mid = TODAY - timedelta(days=200)
        path = [
            Point(end=date(2023, 1, 1), value=10.0, known=date(2023, 1, 1)),
            Point(end=mid, value=20.0, known=mid),
        ]
        z = relative_zone(closes([10.0] * 600), series(shares_path=path), TODAY, 10.0)
        assert z.ps_now == pytest.approx(2.0)  # today on 20 shares
        assert z.median == pytest.approx(1.0)  # most of the window on 10

    @pytest.mark.parametrize("price", [0.0, -1.0, None])
    def test_no_price(self, price):
        assert relative_zone(closes([10.0] * 400), series(), TODAY, price) is None

    def test_no_series(self):
        assert relative_zone(closes([10.0] * 400), None, TODAY, 10.0) is None

    def test_zero_or_negative_prices_in_history_skipped(self):
        hist = closes([10.0] * 400 + [0.0, -5.0, float("nan")] * 3)
        z = relative_zone([(d, p) for d, p in hist if p == p], series(), TODAY, 10.0)
        assert z.median == pytest.approx(1.0)


def snapshot(**kw):
    base = dict(
        symbol="X",
        asof=TODAY,
        price=100.0,
        shares_outstanding=10.0,
        market_cap=1000.0,
        net_cash=0.0,
        enterprise_value=1000.0,
        revenue_runrate=100.0,
        ev_to_revenue=10.0,
        source_accession="0000",
        period_end=TODAY - timedelta(days=60),
        revenue_yoy=0.2,
        scenarios=[
            ImpliedExpectations(
                terminal_multiple=m,
                fcf_margin=f,
                years=10,
                required_fcf=1,
                required_revenue=1,
                implied_cagr=c,
            )
            for m, f, c in ((30, 0.3, 0.01), (25, 0.25, 0.02), (20, 0.2, 0.03))
        ],
    )
    base.update(kw)
    return ValuationSnapshot(**base)


class TestAbsolute:
    def test_negative_own_margin_is_left_out_with_the_reason(self):
        """AMZN: -0.3% trailing FCF from AI capex."""
        ctx = absolute_context(snapshot(), 100.0, own_margin=-0.003)
        assert [label for label, _, _ in ctx.readings] == ["generic"]
        assert any("burning cash" in n for n in ctx.notes)

    def test_lower_own_margin_widens_the_range_upward(self):
        ctx = absolute_context(snapshot(), 100.0, own_margin=0.10)
        assert len(ctx.readings) == 2 and ctx.high > ctx.low

    def test_stored_margins_are_used_and_live_one_ignored(self):
        snap = snapshot(
            margin_trailing=0.18,
            margin_trailing_label="t4q",
            margin_median=0.327,
            margin_median_label="FY2023–FY2025 median",
        )
        ctx = absolute_context(snap, 100.0, own_margin=0.99)
        margins = sorted(m for _, m, _ in ctx.readings)
        assert margins == [0.18, 0.25, 0.327]

    def test_unknown_margins_are_said(self):
        ctx = absolute_context(snapshot(), 100.0, own_margin=None)
        assert sum("unavailable" in n for n in ctx.notes) == 2

    def test_no_snapshot(self):
        assert absolute_context(None, 100.0) is None

    def test_required_at_rejects_bad_inputs(self):
        sc = Scenario(terminal_multiple=25, fcf_margin=0.25, years=10)
        assert required_at(snapshot(revenue_runrate=None), 100, sc) is None
        assert required_at(snapshot(), 0, sc) is None
        assert required_at(snapshot(net_cash=5000), 100, sc) is None  # EV negative
        assert required_at(snapshot(), 100, sc.model_copy(update={"fcf_margin": 0})) is None

    def test_consensus_growth_annualised_from_the_run_rate(self):
        from advisor.valuation.consensus import Consensus, RevenueEstimate

        snap = snapshot(period_end=date(2025, 12, 31))
        cons = Consensus(
            symbol="X",
            years=[
                RevenueEstimate(
                    label="FY2027", fiscal_year_end=date(2027, 12, 31), avg=121.0, analysts=10
                )
            ],
        )
        g, label = consensus_growth(snap, cons)
        assert g == pytest.approx(0.10, abs=0.002) and "FY2027" in label

    def test_consensus_too_close_is_not_annualised(self):
        from advisor.valuation.consensus import Consensus, RevenueEstimate

        snap = snapshot(period_end=date(2026, 12, 1))
        cons = Consensus(
            symbol="X",
            years=[RevenueEstimate(label="FY2026", fiscal_year_end=date(2026, 12, 31), avg=110.0)],
        )
        assert consensus_growth(snap, cons) == (None, "")
