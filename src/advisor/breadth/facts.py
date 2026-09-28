"""Quarterly revenue and diluted shares for every SEC filer, from the ``frames`` API.

One ``frames`` call returns one concept for one calendar period across every
filer: measured on 2026-09-27, CY2026Q2 revenue for ~4,000 companies in two
calls under a second. The per-company ``companyconcept`` API the depth layer
uses would need a call per company per concept.

The rows are the same shape ``valuation.history`` already reads — framed,
undimensioned facts — so the quarter logic, the fiscal-Q4 derivation, the
concept merge and the 45/75-day "known" lag are reused, not rewritten.

**What frames cannot do, measured.** 48 calls (four revenue concepts,
CY2024Q1–CY2026Q2 plus two annual frames) gave 5,765 filers some revenue
and 4,101 a trailing-twelve-month figure. The gap is structural: a company
whose fiscal year does not end in December never reports its fiscal fourth
quarter on its own, and its fiscal year never matches a calendar annual
frame, so the quarter cannot be derived. MSFT's latest TTM from frames alone
ended 2026-03-31; COST had none. For names that pass E0 and show that gap,
the company's own ``companyconcept`` series fills it (``fill_gaps``).

**Point in time, with one known leak.** A frame returns the latest filed
value for each period, so a later restatement replaces the original. Values
are treated as known 45 days after their quarter (75 for a derived fourth),
as in the depth layer; the restatement itself cannot be undone from this
source, and a replay over these facts says so.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from advisor.breadth.store import BreadthStore
from advisor.valuation.history import (
    REVENUE_CONCEPTS,
    SHARES_CONCEPT,
    Point,
    merge_concepts,
    quarter_key,
    quarterly_points,
    ttm,
)

logger = logging.getLogger(__name__)

FIRST_YEAR = 2021  # quarters from here cover a four-year replay plus a year of TTM
# A frame for a period this long ago is final enough not to be re-read; a
# younger one is refetched on every sync as late filers and amendments land.
SETTLED_AFTER_DAYS = 400
# The latest TTM of a company that reports every quarter is at most this old:
# a 92-day quarter plus the 75-day lag of a derived fourth quarter, plus slack.
STALE_TTM_DAYS = 200
# How often one company's own series is re-read to fill a gap.
CONCEPT_REFRESH_DAYS = 30

_UNITS = {SHARES_CONCEPT: "shares"}
CONCEPTS: tuple[str, ...] = (*REVENUE_CONCEPTS, SHARES_CONCEPT)


def frames_for(today: date) -> list[str]:
    """Every quarterly and annual frame from ``FIRST_YEAR`` whose period has ended. Pure."""
    out = []
    for year in range(FIRST_YEAR, today.year + 1):
        if date(year, 12, 31) < today:
            out.append(f"CY{year}")
        for q in range(1, 5):
            end_month = 3 * q
            end = date(year + (end_month == 12), end_month % 12 + 1, 1) - timedelta(days=1)
            if end < today:
                out.append(f"CY{year}Q{q}")
    return out


def frame_end(frame: str) -> date:
    """The last day of a frame's calendar period. Pure."""
    year = int(frame[2:6])
    if len(frame) == 6:
        return date(year, 12, 31)
    q = int(frame[7])
    month = 3 * q
    return date(year + (month == 12), month % 12 + 1, 1) - timedelta(days=1)


def settled(frame: str, today: date) -> bool:
    return (today - frame_end(frame)).days > SETTLED_AFTER_DAYS


# ── fetching ──────────────────────────────────────────────────────────────

FetchFrame = Callable[[str, str, str], tuple[int, list[dict]]]  # concept, unit, frame
FetchConcept = Callable[[int, str], list[dict]]  # cik, concept


def sec_frame(concept: str, unit: str, frame: str) -> tuple[int, list[dict]]:
    """(HTTP status, rows). 404 means no filer used the concept for that period."""
    import httpx

    from advisor.news.edgar import _client_ready
    from advisor.research.config import get_settings

    _client_ready()
    url = f"https://data.sec.gov/api/xbrl/frames/us-gaap/{concept}/{unit}/{frame}.json"
    r = httpx.get(url, headers={"User-Agent": get_settings().edgar_user_agent}, timeout=60)
    if r.status_code == 404:
        return 404, []
    r.raise_for_status()
    return r.status_code, list(r.json().get("data", []))


def sec_concept(cik: int, concept: str) -> list[dict]:
    """One company's framed rows for ``concept``; [] when it never used it."""
    from advisor.valuation.history import _concept_rows

    return [r for r in _concept_rows(cik, concept) if r.get("frame")]


# ── syncing ───────────────────────────────────────────────────────────────


@dataclass
class FactsReport:
    frames_fetched: int = 0
    frames_settled: int = 0  # skipped: already read and final
    frames_missing: int = 0  # 404: nobody filed that concept for that period
    rows: int = 0
    gaps_filled: int = 0
    gap_rows: int = 0
    errors: list[str] = field(default_factory=list)

    def summary(self) -> str:
        s = (
            f"{self.frames_fetched} frames read ({self.frames_settled} settled, "
            f"{self.frames_missing} absent), {self.rows} rows; "
            f"{self.gaps_filled} companies gap-filled ({self.gap_rows} rows)"
        )
        if self.errors:
            s += f"; {len(self.errors)} errors"
        return s

    def as_dict(self) -> dict:
        return {
            "frames_fetched": self.frames_fetched,
            "frames_settled": self.frames_settled,
            "frames_missing": self.frames_missing,
            "rows": self.rows,
            "gaps_filled": self.gaps_filled,
            "gap_rows": self.gap_rows,
            "errors": self.errors[:20],
        }


def _upsert(store: BreadthStore, concept: str, rows: list[dict], source: str, now: datetime):
    out = []
    for r in rows:
        try:
            out.append(
                (
                    int(r["cik"]),
                    concept,
                    str(r["frame"]),
                    float(r["val"]),
                    r.get("start"),
                    str(r["end"]),
                    r.get("accn"),
                    source,
                    now.isoformat(),
                )
            )
        except (KeyError, TypeError, ValueError):
            continue
    store.conn.executemany(
        "INSERT OR REPLACE INTO breadth_facts "
        "(cik, concept, frame, val, start, end, accn, source, fetched_at) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        out,
    )
    return len(out)


def sync_frames(
    store: BreadthStore, now: datetime, *, fetch: FetchFrame = sec_frame
) -> FactsReport:
    """Read every frame not yet settled. A settled frame already on file is skipped."""
    report = FactsReport()
    today = now.date()
    done = {
        (r[0], r[1]) for r in store.conn.execute("SELECT concept, frame FROM breadth_frame_state")
    }
    for concept in CONCEPTS:
        unit = _UNITS.get(concept, "USD")
        for frame in frames_for(today):
            if (concept, frame) in done and settled(frame, today):
                report.frames_settled += 1
                continue
            try:
                status, rows = fetch(concept, unit, frame)
            except Exception as exc:  # noqa: BLE001
                report.errors.append(f"{concept} {frame}: {exc}")
                continue
            with store.conn:
                n = _upsert(store, concept, [{**r, "frame": frame} for r in rows], "frames", now)
                store.conn.execute(
                    "INSERT OR REPLACE INTO breadth_frame_state "
                    "(concept, frame, status, rows, fetched_at) VALUES (?, ?, ?, ?, ?)",
                    (concept, frame, status, n, now.isoformat()),
                )
            report.frames_fetched += 1
            report.rows += n
            if status == 404:
                report.frames_missing += 1
    return report


# ── reading ───────────────────────────────────────────────────────────────


def _rows(store: BreadthStore, cik: int, concept: str) -> list[dict]:
    return [
        {"frame": r[0], "val": r[1], "start": r[2], "end": r[3]}
        for r in store.conn.execute(
            "SELECT frame, val, start, end FROM breadth_facts WHERE cik = ? AND concept = ?",
            (cik, concept),
        )
    ]


def revenue_ttm(store: BreadthStore, cik: int) -> list[Point]:
    """TTM revenue through time, each point with the day it became known."""
    return ttm(merge_concepts([quarterly_points(_rows(store, cik, c)) for c in REVENUE_CONCEPTS]))


def diluted_shares(store: BreadthStore, cik: int) -> list[Point]:
    return sorted(
        quarterly_points(_rows(store, cik, SHARES_CONCEPT), flow=False).values(),
        key=lambda p: p.end,
    )


def revenue_gap(store: BreadthStore, cik: int, today: date) -> str | None:
    """Why this company's TTM revenue from frames is not usable, or None. Pure over the store.

    - ``none``: quarters on file but no four consecutive ones.
    - ``stale``: the latest TTM ends too long ago for a quarterly filer.
    - ``holes``: a quarter since ``FIRST_YEAR`` with no TTM ending on it —
      the signature of a non-calendar fiscal year. Older holes do not count:
      a company's own series reaches back to 2008 (AAPL's lacks 2008Q3),
      and nothing here reads that far.
    A company with no revenue at all is not a gap: it may have none to report.
    """
    quarters = merge_concepts([quarterly_points(_rows(store, cik, c)) for c in REVENUE_CONCEPTS])
    if not quarters:
        return None
    points = ttm(quarters)
    if not points:
        return "none"
    if (today - max(p.end for p in points)).days > STALE_TTM_DAYS:
        return "stale"
    covered = {quarter_key(p.end) for p in points}
    first = max(min(quarters), (FIRST_YEAR, 1))
    key = first
    for _ in range(3):  # the first TTM can only end on the fourth quarter
        key = _next(key)
    last = max(covered)
    while key <= last:
        if key not in covered:
            return "holes"
        key = _next(key)
    return None


def _next(key: tuple[int, int]) -> tuple[int, int]:
    year, q = key
    return (year + 1, 1) if q == 4 else (year, q + 1)


def fill_gaps(
    store: BreadthStore,
    ciks: list[int],
    now: datetime,
    *,
    fetch: FetchConcept = sec_concept,
) -> FactsReport:
    """For each company whose frames TTM has a gap, read its own series once a month."""
    report = FactsReport()
    today = now.date()
    cutoff = (now - timedelta(days=CONCEPT_REFRESH_DAYS)).isoformat()
    recent = {
        r[0]
        for r in store.conn.execute(
            "SELECT DISTINCT cik FROM breadth_concept_state WHERE fetched_at >= ?", (cutoff,)
        )
    }
    for cik in ciks:
        if cik in recent or revenue_gap(store, cik, today) is None:
            continue
        added = 0
        for concept in CONCEPTS:
            try:
                rows = fetch(cik, concept)
            except Exception as exc:  # noqa: BLE001
                report.errors.append(f"CIK {cik} {concept}: {exc}")
                continue
            with store.conn:
                n = _upsert(
                    store, concept, [{**r, "cik": cik} for r in rows], "companyconcept", now
                )
                store.conn.execute(
                    "INSERT OR REPLACE INTO breadth_concept_state (cik, concept, rows, fetched_at) "
                    "VALUES (?, ?, ?, ?)",
                    (cik, concept, n, now.isoformat()),
                )
            added += n
        report.gaps_filled += 1
        report.gap_rows += added
    return report


def coverage(store: BreadthStore, ciks: list[int], today: date) -> dict[str, int]:
    """How many of ``ciks`` have a usable TTM revenue, and why the rest do not."""
    out = {"ok": 0, "no revenue": 0, "none": 0, "stale": 0, "holes": 0}
    for cik in ciks:
        gap = revenue_gap(store, cik, today)
        if gap is not None:
            out[gap] += 1
        elif revenue_ttm(store, cik):
            out["ok"] += 1
        else:
            out["no revenue"] += 1
    return out
