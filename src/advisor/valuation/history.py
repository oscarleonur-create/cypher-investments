"""Quarterly revenue and share counts through time, from the SEC's own series.

The entry zone compares a name with itself: is today's price-to-sales below
its two-year median? That needs revenue and share count *as they were known*
on every past day, not today's figures applied backwards — INTC's diluted
shares went from 4.5bn to 5.1bn after its $20bn raise, so today's count on a
2025 price would overstate that day's market cap by 13%.

Source: the SEC ``companyconcept`` API. It carries only undimensioned facts —
the consolidated figure this repo insists on — and one call returns a
concept's whole history. Three things it does not do for you:

1. **The fiscal fourth quarter is not reported as a quarter.** It exists
   only inside the annual figure: annual minus the three reported quarters of
   that fiscal year, and only when all three are present. For a calendar-year
   company that is October–December; for MSFT (year to June) April–June.
2. **``filed`` is the last filing that repeated a value, not the first.**
   AMZN's Q1 2025 revenue shows ``filed`` 2026-04-30 — the next year's 10-Q,
   which restated it as a comparative. Using it would hide a year of history;
   using the period end would leak the future. A value is taken as known
   45 days after its quarter ends (75 for a derived Q4, which waits on the
   10-K).
3. **Companies change revenue concepts.** META reported ``Revenues`` until
   2018 and ``RevenueFromContract…`` after. Concepts are merged by
   preference, one value per period.
"""

from __future__ import annotations

import logging
import re
from datetime import date, timedelta

from pydantic import BaseModel

logger = logging.getLogger(__name__)

REVENUE_CONCEPTS: tuple[str, ...] = (
    "RevenueFromContractWithCustomerExcludingAssessedTax",
    "Revenues",
    "RevenueFromContractWithCustomerIncludingAssessedTax",
    "SalesRevenueNet",
)
SHARES_CONCEPT = "WeightedAverageNumberOfDilutedSharesOutstanding"
QUARTER_LAG = timedelta(days=45)
ANNUAL_LAG = timedelta(days=75)

_QUARTER = re.compile(r"CY(\d{4})Q([1-4])$")
_YEAR = re.compile(r"CY(\d{4})$")


class Point(BaseModel):
    end: date  # period end
    value: float
    known: date  # first day the value is treated as public


def quarter_key(end: date) -> tuple[int, int]:
    """The calendar quarter a period ending on ``end`` belongs to.

    52/53-week years end a few days either side of a quarter boundary: INTC's
    quarters end 2025-09-27, some retailers' on 2025-01-03. A period ending in
    the first week of a quarter's first month is counted in the quarter before.
    """
    year, month = end.year, end.month
    if month in (1, 4, 7, 10) and end.day <= 7:
        month -= 1
        if month == 0:
            year, month = year - 1, 12
    return year, (month - 1) // 3 + 1


def quarterly_points(rows: list[dict], *, flow: bool = True) -> dict[tuple[int, int], Point]:
    """One point per calendar quarter from a concept's framed rows. Pure.

    The quarter a company never reports on its own is its *fiscal* fourth —
    the one ending with its fiscal year. For a calendar-year company that is
    October–December; for MSFT (fiscal year to June) it is April–June, and
    assuming October–December there produced a P/S of 24x against a real 11x.
    So the missing quarter is found from the annual figure's own end date:
    the annual period's quarters are the reported ones ending within the year
    before it, and when exactly one is absent it is derived.

    ``flow`` quantities (revenue) derive it by subtraction; a ``flow=False``
    series (a weighted-average share count) is an average, not a sum, so the
    annual average stands in for the missing quarter.
    """
    quarters: dict[tuple[int, int], Point] = {}
    annual: list[tuple[date, date | None, float]] = []
    for row in rows:
        frame = str(row.get("frame") or "")
        try:
            value = float(row["val"])
            end = date.fromisoformat(row["end"])
            start = date.fromisoformat(row["start"]) if row.get("start") else None
        except (KeyError, TypeError, ValueError):
            continue
        if _QUARTER.fullmatch(frame):
            quarters[quarter_key(end)] = Point(end=end, value=value, known=end + QUARTER_LAG)
        elif _YEAR.fullmatch(frame):
            annual.append((end, start, value))

    for end, start, value in annual:
        key = quarter_key(end)
        if key in quarters:
            continue
        begin = start or (end - timedelta(days=366))
        inside = [p for k, p in quarters.items() if begin < p.end < end]
        if flow:
            if len(inside) != 3:
                continue  # can't tell which quarter is missing, or more than one is
            missing = value - sum(p.value for p in inside)
            if missing <= 0:
                continue  # a restatement mismatch; better a gap than a negative quarter
        else:
            missing = value
        quarters[key] = Point(end=end, value=missing, known=end + ANNUAL_LAG)
    return quarters


def merge_concepts(series: list[dict[tuple[int, int], Point]]) -> dict[tuple[int, int], Point]:
    """First series wins per quarter; later ones only fill gaps."""
    merged: dict[tuple[int, int], Point] = {}
    for s in series:
        for key, point in s.items():
            merged.setdefault(key, point)
    return merged


def ttm(quarters: dict[tuple[int, int], Point]) -> list[Point]:
    """Trailing-twelve-month sums wherever four consecutive quarters exist."""
    out = []
    for year, q in sorted(quarters):
        keys = []
        y, qq = year, q
        for _ in range(4):
            keys.append((y, qq))
            qq -= 1
            if qq == 0:
                y, qq = y - 1, 4
        if all(k in quarters for k in keys):
            pts = [quarters[k] for k in keys]
            out.append(
                Point(
                    end=pts[0].end,
                    value=sum(p.value for p in pts),
                    known=max(p.known for p in pts),
                )
            )
    return out


def adjust_for_splits(shares: list[Point], splits: list[tuple[date, float]]) -> list[Point]:
    """Put pre-split share counts on today's basis, where the SEC has not.

    Yahoo's prices are split-adjusted; SEC share counts are restated only in
    filings made after the split, so a history mixes both bases. A point
    before a split is multiplied by the ratio only if it *looks* unadjusted —
    within 30% of 1/ratio of the first count after the split. Already-restated
    points are left alone.
    """
    out = sorted(shares, key=lambda p: p.end)
    for when, ratio in sorted(splits):
        if not ratio or ratio <= 0 or abs(ratio - 1) < 1e-9:
            continue
        after = [p for p in out if p.end >= when]
        if not after:
            continue
        reference = after[0].value
        adjusted = []
        for p in out:
            if p.end < when and reference > 0 and abs(p.value * ratio / reference - 1) < 0.3:
                p = p.model_copy(update={"value": p.value * ratio})
            adjusted.append(p)
        out = adjusted
    return out


def break_after(shares: list[Point], max_jump: float = 2.5) -> date | None:
    """The period end after which the share count is continuous, if it ever broke.

    A quarter-on-quarter change beyond ``max_jump`` (either way) that no split
    explains is a different capital structure — WOLF's emergence from
    bankruptcy in 2025 is the live case. Price history before it describes a
    different set of shares, so it cannot anchor a median.
    """
    ordered = sorted(shares, key=lambda p: p.end)
    last_break = None
    for prev, cur in zip(ordered, ordered[1:]):
        if prev.value <= 0 or cur.value <= 0:
            continue
        ratio = cur.value / prev.value
        if ratio > max_jump or ratio < 1 / max_jump:
            last_break = cur.end
    return last_break


def as_of(points: list[Point], day: date) -> Point | None:
    """The latest point known on ``day``."""
    known = [p for p in points if p.known <= day]
    return max(known, key=lambda p: p.end) if known else None


# ── Live source ───────────────────────────────────────────────────────────


def _concept_rows(cik: int, concept: str, taxonomy: str = "us-gaap") -> list[dict]:
    import httpx

    from advisor.news.edgar import _client_ready
    from advisor.research.config import get_settings

    _client_ready()
    url = f"https://data.sec.gov/api/xbrl/companyconcept/CIK{cik:010d}/{taxonomy}/{concept}.json"
    r = httpx.get(url, headers={"User-Agent": get_settings().edgar_user_agent}, timeout=20)
    if r.status_code == 404:
        return []
    r.raise_for_status()
    units = r.json().get("units", {})
    return units.get("USD") or units.get("shares") or next(iter(units.values()), [])


class Series(BaseModel):
    symbol: str
    revenue_ttm: list[Point]
    shares: list[Point]
    # Share count broke (reorganisation, unexplained jump) at this period end;
    # history before it must not be compared with today.
    broken_after: date | None = None


def build_series(
    symbol: str,
    revenue_rows: list[list[dict]],
    share_rows: list[dict],
    splits: list[tuple[date, float]],
) -> Series | None:
    """Pure assembly of a Series from raw concept rows and known splits."""
    revenue_ttm = ttm(merge_concepts([quarterly_points(rows) for rows in revenue_rows]))
    shares = adjust_for_splits(list(quarterly_points(share_rows, flow=False).values()), splits)
    if not revenue_ttm or not shares:
        return None
    return Series(
        symbol=symbol.upper(),
        revenue_ttm=revenue_ttm,
        shares=shares,
        broken_after=break_after(shares),
    )


def _splits(symbol: str) -> list[tuple[date, float]]:
    try:
        import yfinance as yf

        s = yf.Ticker(symbol).splits
        return [(ts.date(), float(r)) for ts, r in s.items()] if s is not None else []
    except Exception as exc:  # noqa: BLE001
        logger.info("history: no split history for %s: %s", symbol, exc)
        return []


def load_series(symbol: str) -> Series | None:
    """TTM revenue and diluted shares through time. None when either is missing."""
    from advisor.news.edgar import company_for

    company = company_for(symbol)
    cik = getattr(company, "cik", None)
    if not cik:
        return None
    try:
        revenue_rows = [_concept_rows(int(cik), c) for c in REVENUE_CONCEPTS]
        share_rows = _concept_rows(int(cik), SHARES_CONCEPT)
    except Exception as exc:  # noqa: BLE001
        logger.warning("history: SEC series unavailable for %s: %s", symbol, exc)
        return None
    return build_series(symbol, revenue_rows, share_rows, _splits(symbol))
