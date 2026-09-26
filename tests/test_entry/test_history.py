"""SEC revenue/share series: missing quarters, fiscal years, look-ahead, splits, breaks."""

from __future__ import annotations

from datetime import date

import pytest
from advisor.valuation.history import (
    Point,
    adjust_for_splits,
    as_of,
    break_after,
    build_series,
    merge_concepts,
    quarter_key,
    quarterly_points,
    ttm,
    yahoo_rows,
)


def q(frame, start, end, val):
    return {"frame": frame, "start": start, "end": end, "val": val}


CALENDAR_YEAR = [
    q("CY2025Q1", "2025-01-01", "2025-03-31", 100),
    q("CY2025Q2", "2025-04-01", "2025-06-30", 110),
    q("CY2025Q3", "2025-07-01", "2025-09-30", 120),
    q("CY2025", "2025-01-01", "2025-12-31", 460),  # Q4 only inside the annual
]

# MSFT-like: fiscal year July-June; the unreported quarter is April-June.
FISCAL_JUNE = [
    q("CY2024Q3", "2024-07-01", "2024-09-30", 65),
    q("CY2024Q4", "2024-10-01", "2024-12-31", 70),
    q("CY2025Q1", "2025-01-01", "2025-03-31", 70),
    q("CY2024", "2024-07-01", "2025-06-30", 281),  # FY ending June 2025
]


class TestQuarters:
    def test_calendar_year_q4_derived_from_the_annual(self):
        pts = quarterly_points(CALENDAR_YEAR)
        assert pts[(2025, 4)].value == 130
        assert pts[(2025, 4)].end == date(2025, 12, 31)

    def test_fiscal_june_derives_april_june_not_q4(self):
        """Assuming Oct-Dec here gave MSFT a P/S of 24x against a real 11x."""
        pts = quarterly_points(FISCAL_JUNE)
        assert pts[(2025, 2)].value == 76
        assert (2025, 4) not in pts

    def test_two_missing_quarters_are_not_guessed(self):
        rows = [r for r in CALENDAR_YEAR if r["frame"] != "CY2025Q2"]
        assert (2025, 4) not in quarterly_points(rows)

    def test_negative_derived_quarter_is_a_gap(self):
        rows = CALENDAR_YEAR[:3] + [q("CY2025", "2025-01-01", "2025-12-31", 300)]
        assert (2025, 4) not in quarterly_points(rows)

    def test_share_average_uses_the_annual_not_subtraction(self):
        rows = [
            q("CY2025Q1", "2025-01-01", "2025-03-31", 10.0),
            q("CY2025Q2", "2025-04-01", "2025-06-30", 10.1),
            q("CY2025Q3", "2025-07-01", "2025-09-30", 10.2),
            q("CY2025", "2025-01-01", "2025-12-31", 10.15),
        ]
        assert quarterly_points(rows, flow=False)[(2025, 4)].value == 10.15

    def test_malformed_rows_are_skipped(self):
        rows = CALENDAR_YEAR + [{"frame": "CY2026Q1", "end": "bad", "val": 1}, {"val": 3}]
        assert (2026, 1) not in quarterly_points(rows)

    @pytest.mark.parametrize(
        "end,key",
        [
            (date(2025, 9, 27), (2025, 3)),  # INTC 52/53-week quarter
            (date(2024, 12, 28), (2024, 4)),
            (date(2025, 1, 3), (2024, 4)),  # a year that ends in early January
            (date(2025, 4, 5), (2025, 1)),
            (date(2025, 6, 30), (2025, 2)),
        ],
    )
    def test_quarter_key_near_boundaries(self, end, key):
        assert quarter_key(end) == key


class TestTTM:
    def test_needs_four_consecutive_quarters(self):
        pts = quarterly_points(CALENDAR_YEAR)
        gappy = {k: v for k, v in pts.items() if k != (2025, 2)}
        assert ttm(gappy) == []  # a hole means no trailing year anywhere
        pts[(2024, 4)] = Point(end=date(2024, 12, 31), value=90, known=date(2025, 2, 14))
        earlier, latest = ttm(pts)  # windows ending 2025Q3 and 2025Q4
        assert earlier.value == 90 + 100 + 110 + 120 and earlier.end == date(2025, 9, 30)
        assert latest.value == 100 + 110 + 120 + 130 and latest.end == date(2025, 12, 31)

    def test_known_date_is_the_latest_part(self):
        pts = quarterly_points(CALENDAR_YEAR)
        pts[(2024, 4)] = Point(end=date(2024, 12, 31), value=90, known=date(2025, 2, 14))
        earlier, latest = ttm(pts)
        assert earlier.known == date(2025, 11, 14)  # 2025-09-30 + 45 days
        assert latest.known == date(2026, 3, 16)  # 2025-12-31 + 75 days (derived Q4)

    def test_concepts_merge_by_preference(self):
        a = {(2025, 1): Point(end=date(2025, 3, 31), value=1, known=date(2025, 5, 15))}
        b = {
            (2025, 1): Point(end=date(2025, 3, 31), value=99, known=date(2025, 5, 15)),
            (2025, 2): Point(end=date(2025, 6, 30), value=2, known=date(2025, 8, 14)),
        }
        merged = merge_concepts([a, b])
        assert merged[(2025, 1)].value == 1 and merged[(2025, 2)].value == 2


class TestAsOf:
    pts = [
        Point(end=date(2025, 3, 31), value=1, known=date(2025, 5, 15)),
        Point(end=date(2025, 6, 30), value=2, known=date(2025, 8, 14)),
    ]

    def test_no_look_ahead(self):
        """The quarter that ended 06-30 is not known on 08-01."""
        assert as_of(self.pts, date(2025, 8, 1)).value == 1
        assert as_of(self.pts, date(2025, 8, 14)).value == 2

    def test_nothing_known_yet(self):
        assert as_of(self.pts, date(2025, 1, 1)) is None


def sp(end, value):
    return Point(end=end, value=value, known=end)


class TestSplits:
    def test_unadjusted_counts_before_a_split_are_scaled(self):
        shares = [sp(date(2024, 3, 31), 2.5), sp(date(2024, 9, 30), 24.8)]
        out = adjust_for_splits(shares, [(date(2024, 6, 10), 10.0)])
        assert out[0].value == pytest.approx(25.0)

    def test_already_restated_counts_are_left_alone(self):
        """NVDA: the SEC series already carries post-split counts before the split."""
        shares = [sp(date(2024, 3, 31), 24.9), sp(date(2024, 9, 30), 24.8)]
        out = adjust_for_splits(shares, [(date(2024, 6, 10), 10.0)])
        assert out[0].value == pytest.approx(24.9)

    def test_split_with_no_later_count(self):
        shares = [sp(date(2024, 3, 31), 2.5)]
        assert adjust_for_splits(shares, [(date(2024, 6, 10), 10.0)])[0].value == 2.5

    @pytest.mark.parametrize("ratio", [0.0, 1.0, -2.0])
    def test_degenerate_ratios_ignored(self, ratio):
        shares = [sp(date(2024, 3, 31), 2.5), sp(date(2024, 9, 30), 24.8)]
        assert adjust_for_splits(shares, [(date(2024, 6, 10), ratio)])[0].value == 2.5


class TestBreaks:
    def test_reorganisation_is_a_break(self):
        """WOLF: 0.19bn shares to 0.03bn on emergence from bankruptcy."""
        shares = [
            sp(date(2025, 9, 29), 0.19),
            sp(date(2025, 12, 28), 0.03),
            sp(date(2026, 3, 29), 0.04),
        ]
        assert break_after(shares) == date(2025, 12, 28)

    def test_an_acquisition_in_stock_is_not(self):
        """AVGO: 4.27bn to 4.67bn for VMware, a real 9% change."""
        shares = [sp(date(2023, 10, 29), 4.27), sp(date(2024, 2, 4), 4.67)]
        assert break_after(shares) is None

    def test_build_series_carries_the_break(self):
        rev = [
            q(f"CY{y}Q{n}", None, e, 100)
            for y, n, e in [
                (2025, 1, "2025-03-31"),
                (2025, 2, "2025-06-30"),
                (2025, 3, "2025-09-30"),
                (2025, 4, "2025-12-31"),
            ]
        ]
        shares = [q("CY2025Q1", None, "2025-03-31", 10), q("CY2025Q4", None, "2025-12-31", 1)]
        s = build_series("x", [rev], shares, [])
        assert s.broken_after == date(2025, 12, 31) and s.symbol == "X"

    def test_no_revenue_no_series(self):
        assert build_series("x", [[]], [q("CY2025Q1", None, "2025-03-31", 10)], []) is None


class TestYahooRows:
    """NBIS files IFRS; the SEC series is empty and Yahoo is the fallback."""

    # Yahoo's NBIS figures on 2026-09-25, $M.
    QUARTERS = {
        date(2025, 6, 30): 105.1,
        date(2025, 9, 30): 146.1,
        date(2025, 12, 31): 227.7,
        date(2026, 3, 31): 399.0,
        date(2026, 6, 30): 582.3,
    }
    ANNUAL = {date(2025, 12, 31): 529.8, date(2024, 12, 31): 91.5}

    def test_the_quarter_off_the_left_edge_is_derived(self):
        rows = yahoo_rows(self.QUARTERS, self.ANNUAL)
        q1 = [r for r in rows if r["frame"] == "CY2025Q1"]
        assert len(q1) == 1
        assert q1[0]["val"] == pytest.approx(529.8 - 105.1 - 146.1 - 227.7)
        assert q1[0]["end"] == "2025-03-31"

    def test_a_year_with_two_missing_quarters_is_not_guessed(self):
        rows = yahoo_rows(self.QUARTERS, self.ANNUAL)
        assert not any(r["frame"].startswith("CY2024") for r in rows)

    def test_ttm_from_yahoo(self):
        rows = yahoo_rows(self.QUARTERS, self.ANNUAL)
        points = ttm(quarterly_points(rows))
        assert [p.end for p in points] == [
            date(2025, 12, 31),
            date(2026, 3, 31),
            date(2026, 6, 30),
        ]
        assert points[0].value == pytest.approx(529.8)
        assert points[-1].value == pytest.approx(146.1 + 227.7 + 399.0 + 582.3)

    def test_a_negative_derived_quarter_is_a_gap(self):
        rows = yahoo_rows(self.QUARTERS, {date(2025, 12, 31): 400.0})
        assert not any(r["frame"] == "CY2025Q1" for r in rows)

    def test_nothing_in_nothing_out(self):
        assert yahoo_rows({}, {}) == []

    def test_series_names_its_source(self):
        rows = yahoo_rows(self.QUARTERS, self.ANNUAL)
        shares = yahoo_rows({d: 250.0 for d in self.QUARTERS}, {})
        s = build_series("nbis", [rows], shares, [], source="yfinance statements")
        assert s.source == "yfinance statements" and s.symbol == "NBIS"
