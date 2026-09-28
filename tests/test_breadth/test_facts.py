"""SEC frames: which periods are read, what is re-read, and filling fiscal-year gaps."""

from __future__ import annotations

from datetime import date, datetime

import pytest
from advisor.breadth import facts as F
from advisor.breadth.store import BreadthStore
from advisor.daemon.market_calendar import MARKET_TZ

NOW = datetime(2026, 9, 27, 18, 30, tzinfo=MARKET_TZ)
REV = "RevenueFromContractWithCustomerExcludingAssessedTax"


@pytest.fixture
def store(tmp_path):
    s = BreadthStore(tmp_path / "b.db")
    yield s
    s.close()


def test_frames_for_only_ended_periods():
    frames = F.frames_for(date(2026, 9, 27))
    assert "CY2021Q1" in frames and "CY2020Q4" not in frames
    assert "CY2026Q2" in frames and "CY2026Q3" not in frames
    assert "CY2025" in frames and "CY2026" not in frames
    # On the last day of a quarter the quarter has not ended yet.
    assert "CY2026Q3" not in F.frames_for(date(2026, 9, 30))
    assert "CY2026Q3" in F.frames_for(date(2026, 10, 1))


def test_frame_end():
    assert F.frame_end("CY2026Q2") == date(2026, 6, 30)
    assert F.frame_end("CY2025Q4") == date(2025, 12, 31)
    assert F.frame_end("CY2025") == date(2025, 12, 31)


def q(frame, val, start, end, cik=1):
    return {"cik": cik, "val": val, "start": start, "end": end, "accn": "x", "frame": frame}


class Frames:
    """A fake frames API: CIK 1 files calendar quarters, 404 for the old SalesRevenueNet."""

    def __init__(self):
        self.calls = []

    def __call__(self, concept, unit, frame):
        self.calls.append((concept, frame))
        if concept == "SalesRevenueNet":
            return 404, []
        if concept == REV and "Q" in frame:
            end = F.frame_end(frame)
            return 200, [
                {"cik": 1, "val": 100.0, "start": None, "end": end.isoformat(), "accn": "a"}
            ]
        return 200, []


def test_settled_frames_are_not_reread_but_recent_ones_are(store):
    src = Frames()
    r1 = F.sync_frames(store, NOW, fetch=src)
    n1 = len(src.calls)
    assert r1.frames_fetched == n1 and r1.frames_missing > 0
    r2 = F.sync_frames(store, NOW, fetch=src)
    recent = [f for f in F.frames_for(NOW.date()) if not F.settled(f, NOW.date())]
    assert r2.frames_fetched == len(recent) * len(F.CONCEPTS)
    assert r2.frames_settled == n1 - r2.frames_fetched
    # Re-reading replaced rows, never duplicated them.
    assert store.conn.execute("SELECT COUNT(*) FROM breadth_facts").fetchone()[0] == r1.rows


def test_a_failing_frame_is_reported_and_retried(store):
    def flaky(concept, unit, frame):
        raise TimeoutError("sec.gov timed out")

    r = F.sync_frames(store, NOW, fetch=flaky)
    assert r.frames_fetched == 0 and r.errors
    assert store.conn.execute("SELECT COUNT(*) FROM breadth_frame_state").fetchone()[0] == 0


def test_calendar_filer_has_a_ttm_with_no_gap(store):
    F.sync_frames(store, NOW, fetch=Frames())
    assert F.revenue_gap(store, 1, NOW.date()) is None
    ttm = F.revenue_ttm(store, 1)
    assert ttm[-1].value == 400.0 and ttm[-1].end == date(2026, 6, 30)
    assert ttm[-1].known == date(2026, 8, 14)  # 45 days after the quarter


def _put(store, cik, rows):
    with store.conn:
        F._upsert(store, REV, [{**r, "cik": cik} for r in rows], "frames", NOW)


def test_fiscal_june_filer_shows_holes_then_is_filled(store):
    # MSFT-like: fiscal year to June. The fiscal Q4 (Apr-Jun) is never a
    # quarter, and its annual never lines up with a calendar-year frame.
    quarters = []
    for y in (2024, 2025):
        quarters += [
            q(f"CY{y - 1}Q3", 60.0, f"{y - 1}-07-01", f"{y - 1}-09-30"),
            q(f"CY{y - 1}Q4", 70.0, f"{y - 1}-10-01", f"{y - 1}-12-31"),
            q(f"CY{y}Q1", 70.0, f"{y}-01-01", f"{y}-03-31"),
        ]
    _put(store, 1, quarters)
    assert F.revenue_gap(store, 1, NOW.date()) == "none"

    annuals = [
        q("CY2023", 260.0, "2023-07-01", "2024-06-30"),
        q("CY2024", 280.0, "2024-07-01", "2025-06-30"),
    ]
    served = []

    def concept(cik, c):
        served.append((cik, c))
        return quarters + annuals if c == REV else []

    r = F.fill_gaps(store, [1], NOW, fetch=concept)
    assert r.gaps_filled == 1
    assert F.revenue_ttm(store, 1)  # the derived fiscal Q4 closes the chain
    # Filled this month: not read again tomorrow, even if a gap remains.
    F.fill_gaps(store, [1], NOW, fetch=concept)
    assert len(served) == len(F.CONCEPTS)


def test_no_revenue_is_not_a_gap(store):
    assert F.revenue_gap(store, 999, NOW.date()) is None
    assert F.coverage(store, [999], NOW.date()) == {
        "ok": 0,
        "no revenue": 1,
        "none": 0,
        "stale": 0,
        "holes": 0,
    }


def test_stale_ttm(store):
    rows = [
        q(f"CY2024Q{i}", 10.0, None, F.frame_end(f"CY2024Q{i}").isoformat()) for i in (1, 2, 3, 4)
    ]
    _put(store, 1, rows)
    assert F.revenue_gap(store, 1, NOW.date()) == "stale"


def _calendar_quarters(first_year, last_frame, skip=()):
    out, y, qn = [], first_year, 1
    while True:
        frame = f"CY{y}Q{qn}"
        if frame not in skip:
            out.append(q(frame, 10.0, None, F.frame_end(frame).isoformat()))
        if frame == last_frame:
            return out
        y, qn = (y + 1, 1) if qn == 4 else (y, qn + 1)


def test_a_hole_before_first_year_is_not_a_gap(store):
    # AAPL's own series reaches 2008 and lacks one quarter there.
    _put(store, 1, _calendar_quarters(2008, "CY2026Q2", skip={"CY2008Q3"}))
    assert F.revenue_gap(store, 1, NOW.date()) is None


def test_a_recent_hole_is_a_gap(store):
    _put(store, 1, _calendar_quarters(2019, "CY2026Q2", skip={"CY2024Q4"}))
    assert F.revenue_gap(store, 1, NOW.date()) == "holes"


def test_a_company_listed_after_first_year_has_no_hole_before_its_first_ttm(store):
    _put(store, 1, _calendar_quarters(2023, "CY2026Q2"))
    assert F.revenue_gap(store, 1, NOW.date()) is None


def test_malformed_rows_are_skipped(store):
    with store.conn:
        n = F._upsert(
            store, REV, [{"cik": "x", "val": 1}, {"cik": 1, "val": None, "end": "2026-06-30",
                                                   "frame": "CY2026Q2"}], "frames", NOW
        )  # fmt: skip
    assert n == 0
