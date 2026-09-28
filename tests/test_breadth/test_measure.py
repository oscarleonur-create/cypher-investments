"""Outcomes against matched controls, filing dates, industries, and the verdict."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd
import pytest
from advisor.breadth import companies as C
from advisor.breadth import filings as FL
from advisor.breadth import measure as M
from advisor.breadth.panel import Panel
from advisor.breadth.store import BreadthStore
from advisor.daemon.market_calendar import MARKET_TZ
from advisor.valuation.history import Point

N = 300
DAYS = pd.bdate_range("2025-01-01", periods=N)
NOW = datetime(2026, 9, 27, 21, 0, tzinfo=MARKET_TZ)


def panel_of(n_symbols=40, drift=None):
    symbols = [f"S{i:02d}" for i in range(n_symbols)]
    rows = np.arange(N)[:, None]
    d = np.zeros(n_symbols) if drift is None else np.asarray(drift)
    close = pd.DataFrame(100 * (1 + d) ** rows, index=DAYS, columns=symbols)
    volume = pd.DataFrame(
        np.tile(1e6 * (1 + np.arange(n_symbols)), (N, 1)), index=DAYS, columns=symbols
    )
    return Panel(close, close.copy(), volume)


def elig(p):
    return pd.DataFrame(True, index=p.close.index, columns=p.close.columns)


class TestOutcomes:
    def test_forward_return_beyond_controls(self):
        drift = [0.0] * 40
        drift[0] = 0.001  # S00 gains 0.1% a session; everyone else is flat
        p = panel_of(drift=drift)
        out = M.Outcomes(p, elig(p), {}, {}).of("S00", 100)
        assert out["d20"]["ret"] == pytest.approx(1.001**20 - 1)
        assert out["d20"]["control"] == pytest.approx(0.0)
        assert out["d20"]["excess"] == pytest.approx(1.001**20 - 1)
        assert "S00" not in out["controls"] and len(out["controls"]) == M.CONTROLS

    def test_horizons_not_reached_are_empty_not_zero(self):
        p = panel_of()
        out = M.Outcomes(p, elig(p), {}, {}).of("S00", N - 30)
        assert out["d20"] is not None and out["d60"] is None and out["d120"] is None
        assert out["mae20"] is not None and out["mae60"] is None

    def test_a_name_that_stops_trading_has_no_outcome(self):
        p = panel_of()
        p.close.iloc[110:, 0] = np.nan  # delisted: beyond FILL_LIMIT sessions of nothing
        out = M.Outcomes(p, elig(p), {}, {}).of("S00", 100)
        assert out["d20"] is None

    def test_a_short_halt_is_bridged(self):
        p = panel_of()
        p.close.iloc[120, 0] = np.nan  # the 20th session after row 100
        out = M.Outcomes(p, elig(p), {}, {}).of("S00", 100)
        assert out["d20"] is not None

    def test_mae_is_the_worst_close_ahead(self):
        p = panel_of()
        p.close.iloc[105, 0] = 90.0
        out = M.Outcomes(p, elig(p), {}, {}).of("S00", 100)
        assert out["mae20"] == pytest.approx(-0.10)

    def test_controls_are_deterministic(self):
        p = panel_of()
        a = M.Outcomes(p, elig(p), {}, {}).of("S03", 100)["controls"]
        b = M.Outcomes(p, elig(p), {}, {}).of("S03", 100)["controls"]
        assert a == b

    def test_industry_match_first_then_widened(self):
        p = panel_of(n_symbols=60)
        cik_of = {s: i + 1 for i, s in enumerate(p.close.columns)}
        # 30 semiconductor makers spread across every size, the rest software.
        sic_of = {i + 1: (3674 if i % 2 == 0 else 7372) for i in range(60)}
        o = M.Outcomes(p, elig(p), cik_of, sic_of)
        controls, level = o.controls("S00", 100)
        assert level == "industry"  # 5 other semis in its size bucket: too few; 29 in all
        assert all(sic_of[cik_of[o.symbols[i]]] == 3674 for i in controls)
        sic_of = {i + 1: 3674 for i in range(60)}
        _, level = M.Outcomes(p, elig(p), cik_of, sic_of).controls("S00", 100)
        assert level == "industry+size"

    def test_no_eligible_peers(self):
        p = panel_of(n_symbols=3)
        e = elig(p)
        e.iloc[:, 1:] = False
        out = M.Outcomes(p, e, {}, {}).of("S00", 100)
        assert out["match"] == "none" and out["d20"] is None


class TestEvaluate:
    def _recs(self, excess, sessions):
        out = []
        for i in range(sessions):
            day = date(2025, 1, 1) + timedelta(days=i)
            out.append(
                {
                    "grp": "F+P",
                    "day": day,
                    "outcomes": {
                        "match": "industry+size",
                        "d20": {"ret": excess + 0.01, "control": 0.01, "excess": excess
                                + (0.001 if i % 2 else -0.001)},
                        "mae20": -0.05,
                    },
                }
            )  # fmt: skip
        return out

    def test_undetermined_with_few_windows(self):
        cells = M.evaluate(self._recs(0.05, 60))  # 60 sessions = 3 windows of 20
        c = next(c for c in cells if c["group"] == "F+P" and c["horizon"] == "d20")
        assert c["verdict"] == "UNDETERMINED" and c["windows"] == 3 and c["ci"] is None

    def test_edge_needs_an_interval_clear_of_zero(self):
        cells = M.evaluate(self._recs(0.05, 400))
        c = next(c for c in cells if c["group"] == "F+P" and c["horizon"] == "d20")
        assert c["verdict"] == "EDGE" and c["ci"][0] > 0

    def test_empty_groups_are_reported_not_skipped(self):
        cells = M.evaluate([])
        from advisor.breadth.signals import GROUPS

        assert len(cells) == len(GROUPS) * len(M.HORIZONS)
        assert all(c["verdict"] == "UNDETERMINED" for c in cells)


class TestFilings:
    MASTER = """\
Description:           Master Index of EDGAR Dissemination Feed
Last Data Received:    June 30, 2026

CIK|Company Name|Form Type|Date Filed|Filename
--------------------------------------------------------------------------------
320193|Apple Inc.|10-Q|2026-05-01|edgar/data/320193/0000320193-26-000010.txt
320193|Apple Inc.|4|2026-05-02|edgar/data/320193/0000320193-26-000011.txt
320193|Apple Inc.|10-Q/A|2026-05-09|edgar/data/320193/0000320193-26-000012.txt
99|Late Co|10-K|2026-04-02|edgar/data/99/0000000099-26-000001.txt
99|Late Co|10-Q|2026-05-12|edgar/data/99/0000000099-26-000002.txt
bad|row|10-Q|2026-05-01|x
"""

    def test_parse_keeps_periodic_originals(self):
        rows = FL.parse_master(self.MASTER)
        assert [(r[1], r[2]) for r in rows] == [(320193, "10-Q"), (99, "10-K"), (99, "10-Q")]
        assert rows[0][0] == "0000320193-26-000010"

    def test_a_late_annual_report_does_not_date_the_quarter(self, tmp_path):
        with BreadthStore(tmp_path / "b.db") as store:
            FL.sync_filings(store, NOW, fetch=lambda y, q: self.MASTER)
            dates = FL.FilingDates(store)
            # Q1 ended 2026-03-31; the 10-K on 04-02 is last year's, the 10-Q on 05-12 is Q1's.
            assert dates.first_after(99, date(2026, 3, 31), annual=False) == date(2026, 5, 12)
            assert dates.first_after(99, date(2025, 12, 31), annual=True) == date(2026, 4, 2)
            assert dates.first_after(12345, date(2026, 3, 31), annual=False) is None

    def test_a_filing_too_late_is_not_the_disclosure(self, tmp_path):
        with BreadthStore(tmp_path / "b.db") as store:
            FL.sync_filings(store, NOW, fetch=lambda y, q: self.MASTER)
            dates = FL.FilingDates(store)
            end = date(2026, 5, 12) - timedelta(days=FL.MAX_FILING_LAG_DAYS + 1)
            assert dates.first_after(99, end, annual=False) is None

    def test_final_quarters_are_read_once_and_failures_reported(self, tmp_path):
        calls = []

        def fetch(y, q):
            calls.append((y, q))
            if (y, q) == (2026, 3):
                raise TimeoutError("sec.gov")
            return ""

        with BreadthStore(tmp_path / "b.db") as store:
            r = FL.sync_filings(store, NOW, fetch=fetch)
            assert r["errors"] == ["2026Q3: sec.gov"]
            n = len(calls)
            FL.sync_filings(store, NOW + timedelta(days=1), fetch=fetch)
            # Only the still-open quarter (which also failed) is asked again.
            assert calls[n:] == [(2026, 3)]

    def test_dating_prefers_the_filing_and_counts_the_fallbacks(self, tmp_path):
        with BreadthStore(tmp_path / "b.db") as store:
            FL.sync_filings(store, NOW, fetch=lambda y, q: self.MASTER)
            dates = FL.FilingDates(store)
            qs = {
                (2026, 1): Point(end=date(2026, 3, 31), value=1.0, known=date(2026, 5, 15)),
                (2026, 2): Point(end=date(2026, 6, 30), value=1.0, known=date(2026, 8, 14)),
            }
            tally: dict = {}
            out = M.dated(qs, 99, dates, tally)
            assert out[(2026, 1)].known == date(2026, 5, 12)
            assert out[(2026, 2)].known == date(2026, 8, 14)  # no 10-Q on file yet: the lag
            assert tally == {"filing": 1, "lag": 1}


class TestCompanies:
    def test_groups_and_divisions(self):
        assert C.major_group(3674) == 36 and C.division(3674) == "manufacturing"
        assert C.division(7372) == "services"
        assert C.major_group(None) is None and C.division(0) is None

    def test_sync_reads_only_missing_and_survives_errors(self, tmp_path):
        seen = []

        def fetch(cik):
            seen.append(cik)
            if cik == 3:
                raise TimeoutError("x")
            return None if cik == 2 else {"sic": 3674, "sic_desc": "Semis", "name": "A"}

        with BreadthStore(tmp_path / "b.db") as store:
            r = C.sync_companies(store, [1, 2, 3, 1], NOW, fetch=fetch)
            assert r["read"] == 2 and r["error_count"] == 1
            assert C.sic_map(store) == {1: 3674, 2: None}  # no SIC is stored, not retried daily
            C.sync_companies(store, [1, 2, 3], NOW, fetch=fetch)
            assert seen == [1, 2, 3, 3]
