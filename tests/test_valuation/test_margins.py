"""Own FCF margins and the required-growth range across margins."""

from __future__ import annotations

from datetime import date, timedelta

import pytest
from advisor.valuation.margins import annual_values, median_margin, required_range
from advisor.valuation.models import ImpliedExpectations, ValuationSnapshot


def fy(y, v):
    return {"frame": f"CY{y}", "end": f"{y}-12-31", "val": v}


def series(pairs):
    return annual_values([fy(y, v) for y, v in pairs])


class TestMedianMargin:
    def test_meta_like_median(self):
        """2023 32.7%, 2024 32.9%, 2025 22.9% -> 32.7%."""
        rev = series([(2023, 100), (2024, 100), (2025, 100)])
        ocf = series([(2023, 50), (2024, 50.9), (2025, 60)])
        capex = series([(2023, 17.3), (2024, 18.0), (2025, 37.1)])
        m = median_margin([ocf], [capex], [rev])
        assert m.margin == pytest.approx(0.327) and m.label == "FY2023–FY2025 median"

    def test_last_three_years_only(self):
        rev = series([(y, 100) for y in range(2019, 2026)])
        ocf = series([(y, 90 if y < 2023 else 30) for y in range(2019, 2026)])
        capex = series([(y, 10) for y in range(2019, 2026)])
        assert median_margin([ocf], [capex], [rev]).margin == pytest.approx(0.20)

    def test_negative_capex_sign_is_normalized(self):
        rev, ocf = series([(2024, 100), (2025, 100)]), series([(2024, 30), (2025, 30)])
        capex = series([(2024, -10), (2025, -10)])
        assert median_margin([ocf], [capex], [rev]).margin == pytest.approx(0.20)

    def test_negative_median_is_kept_as_a_number(self):
        rev, ocf = series([(2024, 100), (2025, 100)]), series([(2024, 5), (2025, 5)])
        capex = series([(2024, 40), (2025, 40)])
        assert median_margin([ocf], [capex], [rev]).margin == pytest.approx(-0.35)

    def test_one_year_is_not_enough(self):
        rev, ocf, capex = series([(2025, 100)]), series([(2025, 30)]), series([(2025, 10)])
        m = median_margin([ocf], [capex], [rev])
        assert m.margin is None and "fewer than two" in m.label

    def test_zero_revenue_year_skipped(self):
        rev = series([(2023, 0), (2024, 100), (2025, 100)])
        ocf, capex = (
            series([(2023, 5), (2024, 30), (2025, 30)]),
            series([(y, 10) for y in (2023, 2024, 2025)]),
        )
        assert median_margin([ocf], [capex], [rev]).label == "FY2024–FY2025 median"

    def test_quarter_frames_ignored(self):
        rows = [{"frame": "CY2025Q1", "end": "2025-03-31", "val": 1}, fy(2025, 9)]
        assert annual_values(rows) == {date(2025, 12, 31): 9}


def snap(**kw):
    base = dict(
        symbol="META",
        asof=date(2026, 9, 25),
        price=752.22,
        shares_outstanding=2.5475e9,
        market_cap=1.916e12,
        net_cash=6.596e9,
        enterprise_value=1.9097e12,
        revenue_runrate=243.204e9,
        ev_to_revenue=7.85,
        source_accession="x",
        period_end=date(2026, 9, 25) - timedelta(days=87),
        scenarios=[
            ImpliedExpectations(
                terminal_multiple=m,
                fcf_margin=f,
                years=10,
                required_fcf=1,
                required_revenue=1,
                implied_cagr=c,
            )
            for m, f, c in ((30, 0.30, 0.0), (25, 0.25, 0.023), (20, 0.20, 0.05))
        ],
    )
    base.update(kw)
    return ValuationSnapshot(**base)


class TestRange:
    def test_meta_on_2026_09_25(self):
        """The numbers that made the case: -0.4% / +2.3% / +5.7%."""
        s = snap(
            margin_trailing=0.18,
            margin_trailing_label="trailing 4 quarters",
            margin_median=0.327,
            margin_median_label="FY2023–FY2025 median",
        )
        readings, notes = required_range(s)
        got = {r.label: round(r.required * 100, 1) for r in readings}
        assert got == {
            "generic": 2.3,
            "own trailing 4 quarters": 5.7,
            "own FY2023–FY2025 median": -0.4,
        }
        assert notes == []

    def test_burning_cash_is_left_out_with_the_reason(self):
        readings, notes = required_range(snap(margin_trailing=-0.003, margin_median=0.052))
        assert len(readings) == 2 and "burning cash" in notes[0]

    def test_old_snapshot_without_margins(self):
        readings, notes = required_range(snap())
        assert [r.label for r in readings] == ["generic"] and len(notes) == 2

    def test_negative_ev_gives_no_readings(self):
        readings, _ = required_range(snap(net_cash=1e13))
        assert readings == []

    def test_no_snapshot(self):
        assert required_range(None) == ([], [])
