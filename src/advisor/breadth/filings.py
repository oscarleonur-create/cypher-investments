"""When each company actually filed: EDGAR's quarterly index of every filing.

A quarter's revenue is public the day the company files the 10-Q (or 10-K)
that carries it — not a fixed 45 days after the quarter. Dating it at +45
days put every calendar-year filer's revenue on the same session (May 15,
August 14, ...): two years of family F fell on about eight dates, which is
both a false timing and too few independent windows for any verdict. The
first periodic report filed after the period ended is the real date.

The earnings release (an 8-K) usually comes days before the 10-Q; the
figure in it is not in XBRL, so this date is still conservative — late,
never early.

Source: ``full-index/<year>/QTR<n>/master.gz``, one compressed file per
quarter (4.3 MB for 2026 Q2, measured 2026-09-27) listing CIK, form and date
filed for everything. A quarter's file is final once the quarter is over and
it was read after that; the current quarter's is re-read on every sync.
"""

from __future__ import annotations

import bisect
import gzip
import logging
from collections.abc import Callable
from datetime import date, datetime, timedelta

from advisor.breadth.store import BreadthStore

logger = logging.getLogger(__name__)

INDEX_URL = "https://www.sec.gov/Archives/edgar/full-index/{year}/QTR{q}/master.gz"
FIRST_YEAR = 2021
# Periodic reports that carry a quarter's or a year's financial statements.
PERIODIC_FORMS = frozenset({"10-Q", "10-K", "10-QT", "10-KT", "20-F", "40-F"})
# A report filed this long after a period is not the one that disclosed it
# (a late filer's catch-up, or a different period's report).
MAX_FILING_LAG_DAYS = 200

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS breadth_filings (
    accession TEXT NOT NULL PRIMARY KEY,
    cik       INTEGER NOT NULL,
    form      TEXT NOT NULL,
    filed     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_breadth_filings_cik ON breadth_filings(cik, filed);
CREATE TABLE IF NOT EXISTS breadth_index_state (
    quarter    TEXT NOT NULL PRIMARY KEY,   -- 2026Q2
    rows       INTEGER NOT NULL,
    fetched_at TEXT NOT NULL
);
"""


def quarters_for(today: date) -> list[tuple[int, int]]:
    """Every calendar quarter from ``FIRST_YEAR`` through today's. Pure."""
    out = []
    for year in range(FIRST_YEAR, today.year + 1):
        for q in range(1, 5):
            if date(year, 3 * q - 2, 1) <= today:
                out.append((year, q))
    return out


def _quarter_end(year: int, q: int) -> date:
    m = 3 * q
    return date(year + (m == 12), m % 12 + 1, 1) - timedelta(days=1)


def parse_master(text: str, forms: frozenset[str] = PERIODIC_FORMS) -> list[tuple]:
    """``(accession, cik, form, filed)`` for the kept forms. Pure.

    Lines look like ``320193|Apple Inc.|10-Q|2026-08-01|edgar/data/320193/0000320193-26-000071.txt``
    after a free-text header that ends in a line of dashes.
    """
    out = []
    for line in text.splitlines():
        parts = line.split("|")
        if len(parts) != 5:
            continue
        cik, _, form, filed, path = parts
        if form.strip() not in forms:
            continue
        try:
            cik_i = int(cik)
            date.fromisoformat(filed.strip())
        except ValueError:
            continue  # the header row "CIK|Company Name|..."
        accession = path.strip().rsplit("/", 1)[-1].removesuffix(".txt")
        out.append((accession, cik_i, form.strip(), filed.strip()))
    return out


FetchIndex = Callable[[int, int], str]


def sec_index(year: int, q: int) -> str:
    import httpx

    from advisor.news.edgar import _client_ready
    from advisor.research.config import get_settings

    _client_ready()
    r = httpx.get(
        INDEX_URL.format(year=year, q=q),
        headers={"User-Agent": get_settings().edgar_user_agent},
        timeout=120,
    )
    r.raise_for_status()
    return gzip.decompress(r.content).decode("latin-1")


def sync_filings(store: BreadthStore, now: datetime, *, fetch: FetchIndex = sec_index) -> dict:
    """Read each quarter's index not yet final. Never raises."""
    store.conn.executescript(_SCHEMA)
    today = now.date()
    state = {
        r[0]: date.fromisoformat(r[1][:10])
        for r in store.conn.execute("SELECT quarter, fetched_at FROM breadth_index_state")
    }
    read, rows, errors = 0, 0, []
    for year, q in quarters_for(today):
        key = f"{year}Q{q}"
        fetched = state.get(key)
        if fetched is not None and fetched > _quarter_end(year, q):
            continue  # read after the quarter closed: final
        try:
            parsed = parse_master(fetch(year, q))
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{key}: {exc}")
            continue
        with store.conn:
            store.conn.executemany(
                "INSERT OR IGNORE INTO breadth_filings (accession, cik, form, filed) "
                "VALUES (?, ?, ?, ?)",
                parsed,
            )
            store.conn.execute(
                "INSERT OR REPLACE INTO breadth_index_state (quarter, rows, fetched_at) "
                "VALUES (?, ?, ?)",
                (key, len(parsed), now.isoformat()),
            )
        read += 1
        rows += len(parsed)
    return {"quarters_read": read, "filings": rows, "errors": errors}


QUARTERLY_FORMS = frozenset({"10-Q", "10-QT"})
ANNUAL_FORMS = frozenset({"10-K", "10-KT", "20-F", "40-F"})


class FilingDates:
    """The first report of the right kind each company filed after a period ended.

    A quarter is dated by a 10-Q, a fiscal year (and the fourth quarter
    derived from it) by an annual report. "Any periodic report" would be
    wrong in the dangerous direction: a 10-K for last year filed late, on
    April 2, would date the March quarter weeks before its 10-Q.
    """

    def __init__(self, store: BreadthStore) -> None:
        store.conn.executescript(_SCHEMA)
        self._by_cik: dict[int, list[tuple[date, str]]] = {}
        for cik, filed, form in store.conn.execute(
            "SELECT cik, filed, form FROM breadth_filings ORDER BY cik, filed"
        ):
            self._by_cik.setdefault(cik, []).append((date.fromisoformat(filed), form))

    def __bool__(self) -> bool:
        return bool(self._by_cik)

    def first_after(self, cik: int, end: date, *, annual: bool) -> date | None:
        """The first 10-Q (or annual report) filed after ``end``, within the lag limit."""
        rows = self._by_cik.get(cik)
        if not rows:
            return None
        forms = ANNUAL_FORMS if annual else QUARTERLY_FORMS
        i = bisect.bisect_right(rows, (end, "￿"))
        for filed, form in rows[i:]:
            if (filed - end).days > MAX_FILING_LAG_DAYS:
                return None
            if form in forms:
                return filed
        return None
