"""Trailing figures from the SEC's reported periods, and the figures a
valuation runs on. Rows are the SEC ``companyconcept`` shape; the values are
the ones the API returned on 2026-09-27 unless marked otherwise."""

from __future__ import annotations

from datetime import date

import pytest
from advisor.valuation.figures import (
    Flows,
    build_figures,
    fiscal_years,
    flows_from_rows,
    periodic,
    ttm_at,
)
from advisor.valuation.models import Fundamentals


def row(start, end, val, form="10-Q", filed="2026-07-31"):
    return {"start": start, "end": end, "val": val, "form": form, "filed": filed}


# AMZN operating cash flow. Its 10-Q states a trailing twelve months outright.
AMZN_OCF = [
    row("2024-07-01", "2025-06-30", 121_137e6),
    row("2025-01-01", "2025-06-30", 49_530e6),
    row("2025-04-01", "2025-06-30", 32_515e6),
    row("2025-01-01", "2025-12-31", 139_514e6, form="10-K", filed="2026-02-06"),
    row("2025-07-01", "2026-06-30", 161_403e6),
    row("2026-01-01", "2026-06-30", 71_419e6),
    row("2026-04-01", "2026-06-30", 45_387e6),
]

# WOLF revenue across its emergence from bankruptcy (fresh-start accounting).
# FY2025 is illustrative; the rest are the filed values.
WOLF_REVENUE = [
    row("2024-07-01", "2025-06-29", 757_600e3, form="10-K", filed="2025-08-20"),
    row("2025-06-30", "2025-09-29", 196_800e3, form="10-K", filed="2026-08-20"),
    row("2025-09-30", "2025-12-28", 168_500e3, filed="2026-02-06"),
    row("2025-09-30", "2026-03-29", 318_700e3, filed="2026-05-07"),
    row("2025-12-29", "2026-03-29", 150_200e3, filed="2026-05-07"),
    row("2025-09-30", "2026-06-28", 468_300e3, form="10-K", filed="2026-08-20"),
]

# SPCX: public for one quarter; two 10-Q columns and nothing annual.
SPCX_REVENUE = [
    row("2025-01-01", "2025-06-30", 8_138e6, filed="2026-08-04"),
    row("2025-04-01", "2025-06-30", 4_071e6, filed="2026-08-04"),
    row("2026-01-01", "2026-06-30", 12_508e6, filed="2026-08-04"),
    row("2026-04-01", "2026-06-30", 7_814e6, filed="2026-08-04"),
]
SPCX_OCF = [row("2026-01-01", "2026-06-30", 2_000e6, filed="2026-08-04")]  # illustrative
SPCX_CAPEX = [row("2026-01-01", "2026-06-30", 27_000e6, filed="2026-08-04")]  # illustrative
SPCX_OPINC = [row("2026-01-01", "2026-06-30", -2_090e6, filed="2026-08-04")]  # illustrative


class TestPeriodic:
    def test_the_latest_statement_of_a_period_stands(self):
        rows = [
            row("2025-01-01", "2025-12-31", 100.0, form="10-K", filed="2026-02-01"),
            row("2025-01-01", "2025-12-31", 104.0, form="10-K", filed="2027-02-01"),
        ]
        assert periodic(rows) == {(date(2025, 1, 1), date(2025, 12, 31)): 104.0}

    def test_an_8k_press_release_is_not_a_periodic_report(self):
        rows = [row("2025-07-01", "2025-09-30", 1.0, form="8-K")]
        assert periodic(rows) == {}

    @pytest.mark.parametrize(
        "bad",
        [
            {"start": "2025-01-01", "end": "2025-06-30", "val": "n/a", "form": "10-Q"},
            {"end": "2025-06-30", "val": 1.0, "form": "10-Q"},
            {"start": "2025-06-30", "end": "2025-01-01", "val": 1.0, "form": "10-Q"},
        ],
    )
    def test_a_malformed_row_is_skipped(self, bad):
        assert periodic([bad]) == {}


class TestTrailingTwelveMonths:
    def test_a_twelve_month_period_stated_outright(self):
        assert ttm_at(periodic(AMZN_OCF), date(2026, 6, 30)) == 161_403e6

    def test_the_bridge_reproduces_what_amzn_states(self):
        """FY2025 + H1 2026 − H1 2025 = the TTM AMZN's own 10-Q reports."""
        bridged = [r for r in AMZN_OCF if r["start"] != "2025-07-01"]
        assert ttm_at(periodic(bridged), date(2026, 6, 30)) == pytest.approx(161_403e6)

    def test_a_missing_piece_is_a_gap_not_a_zero(self):
        no_prior = [r for r in AMZN_OCF if r["start"] not in ("2025-07-01", "2025-01-01")]
        assert ttm_at(periodic(no_prior), date(2026, 6, 30)) is None

    def test_fresh_start_periods_are_chained_into_a_year(self):
        """WOLF: nine months as successor + one quarter as predecessor."""
        assert ttm_at(periodic(WOLF_REVENUE), date(2026, 6, 28)) == pytest.approx(665_100e3)

    def test_a_chain_that_cannot_reach_a_year_is_none(self):
        assert ttm_at(periodic(SPCX_REVENUE), date(2026, 6, 30)) is None

    def test_a_date_with_nothing_reported_is_none(self):
        assert ttm_at(periodic(AMZN_OCF), date(2026, 5, 31)) is None

    def test_a_53_week_year_is_still_a_year(self):
        rows = [row("2024-06-30", "2025-07-05", 371.0, form="10-K")]
        assert ttm_at(periodic(rows), date(2025, 7, 5)) == 371.0


class TestFiscalYears:
    def test_a_10q_trailing_twelve_months_is_not_a_fiscal_year(self):
        """AMZN's July–June TTM would otherwise enter the three-year median."""
        assert fiscal_years(AMZN_OCF) == {date(2025, 12, 31): 139_514e6}


def _years(concept_values: dict[int, float]) -> list[dict]:
    return [
        row(f"{y}-01-01", f"{y}-12-31", v, form="10-K", filed=f"{y + 1}-02-01")
        for y, v in concept_values.items()
    ]


class TestFlows:
    def test_amzn_like_figures_at_one_period_end(self):
        revenue = [
            row("2024-07-01", "2025-06-30", 670e9),
            row("2025-07-01", "2026-06-30", 775.68e9),
        ]
        capex = [row("2025-07-01", "2026-06-30", 173.03e9)]
        income = [row("2025-07-01", "2026-06-30", 93.1e9)]
        flows = flows_from_rows([revenue], [AMZN_OCF], [capex], [income])
        assert flows.period_end == date(2026, 6, 30)
        assert flows.revenue_ttm == pytest.approx(775.68e9)
        assert flows.growth == pytest.approx(775.68 / 670 - 1)
        assert flows.fcf_margin == pytest.approx((161.403 - 173.03) / 775.68)
        assert flows.nopat_margin == pytest.approx(93.1 * 0.79 / 775.68)
        assert "trailing 12 months" in flows.fcf_label

    def test_a_capex_payment_reported_negative_is_not_credited(self):
        revenue = [row("2025-07-01", "2026-06-30", 100.0)]
        ocf = [row("2025-07-01", "2026-06-30", 30.0)]
        capex = [row("2025-07-01", "2026-06-30", -10.0)]
        flows = flows_from_rows([revenue], [ocf], [capex], [[]])
        assert flows.fcf_margin == pytest.approx(0.20)

    def test_a_company_months_public_is_read_over_its_longest_span(self):
        """SPCX: no fiscal year on file, so margins are over six months, labelled."""
        flows = flows_from_rows([SPCX_REVENUE], [SPCX_OCF], [SPCX_CAPEX], [SPCX_OPINC])
        assert flows.revenue_ttm is None
        assert flows.growth == pytest.approx(12_508 / 8_138 - 1)
        assert flows.growth_label.startswith("6 months to 2026-06-30")
        assert flows.fcf_margin == pytest.approx((2_000 - 27_000) / 12_508)
        assert flows.fcf_label == "FCF, 6 months to 2026-06-30"
        assert flows.nopat_margin == pytest.approx(-2_090 * 0.79 / 12_508)

    def test_missing_capex_leaves_fcf_absent_but_not_the_operating_margin(self):
        """AAOI on 2026-09-27: no capex fact at the period end."""
        revenue = [row("2025-07-01", "2026-06-30", 596e6)]
        income = [row("2025-07-01", "2026-06-30", -67e6)]
        flows = flows_from_rows([revenue], [[]], [[]], [income])
        assert flows.fcf_margin is None and flows.fcf_label == ""
        assert flows.nopat_margin == pytest.approx(-67 * 0.79 / 596)

    def test_the_median_is_over_the_last_three_fiscal_years(self):
        revenue = _years({2021: 100.0, 2022: 100.0, 2023: 100.0, 2024: 100.0})
        ocf = _years({2021: 90.0, 2022: 30.0, 2023: 20.0, 2024: 25.0})
        capex = _years({2021: 0.0, 2022: 10.0, 2023: 10.0, 2024: 10.0})
        flows = flows_from_rows([revenue], [ocf], [capex], [[]])
        assert flows.fcf_median == pytest.approx(0.15)  # 20%, 10%, 15% → 15%
        assert flows.fcf_median_label == "FY2022–FY2024 median FCF"

    def test_one_fiscal_year_is_no_median(self):
        one = _years({2024: 100.0})
        flows = flows_from_rows([one], [_years({2024: 20.0})], [_years({2024: 5.0})], [[]])
        assert flows.fcf_median is None

    def test_no_revenue_at_all_is_no_flows(self):
        assert flows_from_rows([[]], [AMZN_OCF], [[]], [[]]) is None

    def test_a_preferred_concept_with_nothing_falls_through_to_the_next(self):
        revenue = [row("2025-07-01", "2026-06-30", 100.0)]
        flows = flows_from_rows([[], revenue], [[]], [[]], [[]])
        assert flows.revenue_ttm == 100.0


FUND = dict(
    symbol="SPCX",
    source_accession="0001628280-26-052535",
    period_end=date(2026, 6, 30),
    period_start=date(2026, 4, 1),
    fiscal_period="Q",
    revenue=7_814e6,
    prior_revenue=4_071e6,
    cash=93_522e6,
    marketable_securities=6_487e6,
    total_debt=39_364e6,
    shares_outstanding=13_181_779_945.0,
)


class TestBuildFigures:
    def test_trailing_twelve_months_is_the_base_when_on_file(self):
        flows = Flows(period_end=date(2026, 6, 30), revenue_ttm=25e9, growth=0.5, growth_label="g")
        fig = build_figures("spcx", 150.0, Fundamentals(**FUND), flows)
        assert fig.symbol == "SPCX"
        assert fig.revenue_base == 25e9
        assert fig.revenue_base_label.startswith("trailing 12 months to 2026-06-30")
        assert fig.revenue_growth == 0.5
        assert fig.net_cash == pytest.approx(93_522e6 + 6_487e6 - 39_364e6)

    def test_without_twelve_months_the_latest_period_is_annualised_and_says_so(self):
        flows = flows_from_rows([SPCX_REVENUE], [SPCX_OCF], [SPCX_CAPEX], [SPCX_OPINC])
        fig = build_figures("SPCX", 150.0, Fundamentals(**FUND), flows)
        assert fig.revenue_base == pytest.approx(4 * 7_814e6)
        assert "quarter to 2026-06-30 × 4" in fig.revenue_base_label
        assert any("annualised" in n for n in fig.notes)
        assert fig.revenue_growth == pytest.approx(12_508 / 8_138 - 1)  # the flows' span

    def test_no_flows_falls_back_to_the_filings_own_comparison(self):
        fig = build_figures("SPCX", 150.0, Fundamentals(**FUND), None)
        assert fig.revenue_growth == pytest.approx(7_814 / 4_071 - 1)
        assert fig.margins == []

    def test_a_yahoo_margin_is_used_only_when_the_sec_has_none(self):
        own = Flows(
            period_end=date(2026, 6, 30), revenue_ttm=1.0, fcf_margin=0.1, fcf_label="FCF, sec"
        )
        with_sec = build_figures("X", 1.0, None, own, fallback_margin=(0.3, "FCF (Yahoo)"))
        without = build_figures("X", 1.0, None, None, fallback_margin=(0.3, "FCF (Yahoo)"))
        assert [m.kind for m in with_sec.margins] == ["fcf"]
        assert [m.kind for m in without.margins] == ["yahoo"]
        assert without.start_margin == 0.3

    def test_flows_and_balance_sheet_months_apart_are_noted(self):
        flows = Flows(period_end=date(2025, 12, 31), revenue_ttm=1e9)
        fig = build_figures("X", 1.0, Fundamentals(**FUND), flows)
        assert any("balance sheet to 2026-06-30" in n for n in fig.notes)

    def test_nothing_at_all_is_all_absent(self):
        fig = build_figures("X", None, None, None)
        assert fig.enterprise_value is None and fig.market_cap is None
        assert fig.revenue_base is None and fig.margins == []
