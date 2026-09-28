"""Each filer's industry, from its SEC submissions record: the SIC code.

The null a signal is judged against is "other names of the same kind, the
same day". Without an industry, a momentum signal in an AI-led market would
look like edge when it is only a sector tilt: semiconductors beat utilities
whatever the rule. SIC is the SEC's own classification, free and on every
filer's record. It is coarse and dated (software sits in 7372, semiconductors
in 3674) but it is what the SEC holds; the two-digit major group is fine
enough to match on and coarse enough to have company.

One call per company, once: a record is re-read only when missing.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from datetime import datetime

from advisor.breadth.store import BreadthStore

logger = logging.getLogger(__name__)

# SIC divisions by two-digit major group: the fallback when a major group has
# too few names that day to draw controls from.
_DIVISIONS: tuple[tuple[int, int, str], ...] = (
    (1, 9, "agriculture"),
    (10, 14, "mining"),
    (15, 17, "construction"),
    (20, 39, "manufacturing"),
    (40, 49, "transport and utilities"),
    (50, 51, "wholesale"),
    (52, 59, "retail"),
    (60, 67, "finance"),
    (70, 89, "services"),
    (90, 99, "public administration"),
)


def major_group(sic: int | None) -> int | None:
    return sic // 100 if sic and sic > 0 else None


def division(sic: int | None) -> str | None:
    g = major_group(sic)
    if g is None:
        return None
    for lo, hi, name in _DIVISIONS:
        if lo <= g <= hi:
            return name
    return None


FetchSubmission = Callable[[int], dict | None]


def sec_submission(cik: int) -> dict | None:
    """The fields kept from a filer's submissions record; None if the SEC has none."""
    import httpx

    from advisor.news.edgar import _client_ready
    from advisor.research.config import get_settings

    _client_ready()
    url = f"https://data.sec.gov/submissions/CIK{cik:010d}.json"
    r = httpx.get(url, headers={"User-Agent": get_settings().edgar_user_agent}, timeout=30)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    j = r.json()
    try:
        sic = int(j.get("sic")) if j.get("sic") not in (None, "") else None
    except (TypeError, ValueError):
        sic = None
    return {"sic": sic, "sic_desc": j.get("sicDescription"), "name": j.get("name")}


def sync_companies(
    store: BreadthStore,
    ciks: list[int],
    now: datetime,
    *,
    fetch: FetchSubmission = sec_submission,
) -> dict:
    """Read the SIC of every CIK not yet on file. Returns counts; never raises."""
    have = {r[0] for r in store.conn.execute("SELECT cik FROM breadth_companies")}
    todo = [c for c in dict.fromkeys(ciks) if c not in have]
    done, errors = 0, []
    for cik in todo:
        try:
            rec = fetch(cik)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"CIK {cik}: {exc}")
            continue
        rec = rec or {}
        with store.conn:
            store.conn.execute(
                "INSERT OR REPLACE INTO breadth_companies (cik, sic, sic_desc, name, fetched_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (cik, rec.get("sic"), rec.get("sic_desc"), rec.get("name"), now.isoformat()),
            )
        done += 1
    return {"asked": len(todo), "read": done, "errors": errors[:20], "error_count": len(errors)}


def sic_map(store: BreadthStore) -> dict[int, int | None]:
    return {r[0]: r[1] for r in store.conn.execute("SELECT cik, sic FROM breadth_companies")}
