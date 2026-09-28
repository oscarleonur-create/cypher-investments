"""Open-market insider purchases for every filer: family I's raw material (B2).

A director or officer buying the company's stock with their own money on the
open market is the one insider trade with a single motive. Sales have many
(taxes, diversification, a house); purchases have one. Clusters of them —
several insiders buying within weeks — are the documented signal
(Lakonishok–Lee 2001; Cohen–Malloy–Pomorski 2012, who also show that a
*routine* buyer, one who buys in the same month every year, carries none).

Two sources, one record:

- **History** from the SEC's quarterly *Insider Transactions Data Sets*
  (every Form 4, transaction codes included; about 11 MB zipped per quarter).
  The files are linked from one index page and their paths are not uniform —
  2026 Q2 moved from ``structureddata`` to ``datastandardsinnovation``
  (found 2026-09-28) — so links are read from the page, never built.
- **The current quarter**, not yet in a data set, from EDGAR's daily index:
  each Form 4 filed on an E0 issuer, read from its own XML.

Both are cut the same way (``Txn``): non-derivative rows with code P or S,
amendments dropped (a ``4/A`` restates an original already counted), and the
10b5-1 flag normalised — the data sets write it four ways ("", 0/1,
false/true). A joint filing (a fund and its general partner) is one buyer.
"""

from __future__ import annotations

import csv
import io
import logging
import re
import xml.etree.ElementTree as ET
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta

from advisor.breadth.store import BreadthStore

logger = logging.getLogger(__name__)

DATASETS_PAGE = "https://www.sec.gov/data-research/sec-markets-data/insider-transactions-data-sets"
DAILY_INDEX_URL = "https://www.sec.gov/Archives/edgar/daily-index/{year}/QTR{q}/master.{ymd}.idx"
FILING_URL = "https://www.sec.gov/Archives/{path}"
FIRST_YEAR = 2020  # three years before the replay window, for the routine-buyer test
KEPT_CODES = frozenset({"P", "S"})

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS breadth_insider_txns (
    accession  TEXT NOT NULL,
    issuer_cik INTEGER NOT NULL,
    owner_cik  INTEGER NOT NULL,     -- the filing's first owner: a joint filing is one buyer
    officer    INTEGER NOT NULL,
    director   INTEGER NOT NULL,
    ten_pct    INTEGER NOT NULL,
    title      TEXT,
    code       TEXT NOT NULL,        -- P open-market purchase, S sale
    acq_disp   TEXT,
    shares     REAL,
    price      REAL,
    trans_date TEXT NOT NULL,
    filed      TEXT NOT NULL,
    direct     TEXT,
    plan       INTEGER,              -- 1 under a 10b5-1 plan, 0 not, NULL not stated
    source     TEXT NOT NULL,        -- 'dataset' or 'daily'
    UNIQUE (accession, owner_cik, trans_date, code, shares, price)
);
CREATE INDEX IF NOT EXISTS idx_breadth_insider_issuer
    ON breadth_insider_txns(issuer_cik, code, filed);
CREATE TABLE IF NOT EXISTS breadth_insider_state (
    key        TEXT NOT NULL PRIMARY KEY,   -- '2026q2' (a data set) or '2026-09-25' (a day)
    rows       INTEGER NOT NULL,
    fetched_at TEXT NOT NULL
);
"""


@dataclass(frozen=True)
class Txn:
    accession: str
    issuer_cik: int
    owner_cik: int
    officer: bool
    director: bool
    ten_pct: bool
    title: str | None
    code: str
    acq_disp: str | None
    shares: float | None
    price: float | None
    trans_date: date
    filed: date
    direct: str | None
    plan: bool | None
    source: str

    @property
    def value(self) -> float:
        return (self.shares or 0.0) * (self.price or 0.0)


def plan_flag(raw: str | None) -> bool | None:
    """The 10b5-1 box as the data sets write it: '', 0/1, false/true. Pure."""
    s = (raw or "").strip().lower()
    if s in ("1", "true"):
        return True
    if s in ("0", "false"):
        return False
    return None


def _num(raw) -> float | None:
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _day(raw: str) -> date | None:
    raw = (raw or "").strip()
    for fmt in ("%d-%b-%Y", "%Y-%m-%d", "%Y%m%d"):
        try:
            return datetime.strptime(raw, fmt).date()
        except ValueError:
            continue
    return None


# ── the quarterly data sets ───────────────────────────────────────────────


def dataset_links(html: str) -> dict[str, str]:
    """``{'2026q2': absolute url}`` from the index page. Pure."""
    out = {}
    for path in re.findall(r'href="([^"]*?(\d{4}q[1-4])_form345\.zip)"', html):
        href, key = path
        out[key] = href if href.startswith("http") else f"https://www.sec.gov{href}"
    return out


def _tsv(z: zipfile.ZipFile, name: str) -> list[dict[str, str]]:
    with z.open(name) as f:
        text = io.TextIOWrapper(f, encoding="utf-8", errors="replace")
        return list(csv.DictReader(text, delimiter="\t", quoting=csv.QUOTE_NONE))


def parse_dataset(blob: bytes) -> list[Txn]:
    """Every kept transaction in one quarter's zip. Pure."""
    z = zipfile.ZipFile(io.BytesIO(blob))
    subs = {}
    for r in _tsv(z, "SUBMISSION.tsv"):
        if (r.get("DOCUMENT_TYPE") or "").strip() != "4":
            continue  # Forms 3 and 5, and every amendment
        filed = _day(r.get("FILING_DATE", ""))
        try:
            issuer = int(r["ISSUERCIK"])
        except (KeyError, TypeError, ValueError):
            continue
        if filed:
            subs[r["ACCESSION_NUMBER"]] = (issuer, filed, plan_flag(r.get("AFF10B5ONE")))
    owners: dict[str, dict] = {}
    for r in _tsv(z, "REPORTINGOWNER.tsv"):
        acc = r.get("ACCESSION_NUMBER")
        if acc not in subs:
            continue
        try:
            cik = int(r["RPTOWNERCIK"])
        except (KeyError, TypeError, ValueError):
            continue
        rel = {x.strip() for x in (r.get("RPTOWNER_RELATIONSHIP") or "").split(",")}
        title = (r.get("RPTOWNER_TITLE") or "").strip() or None
        cur = owners.setdefault(
            acc, {"cik": cik, "officer": False, "director": False, "ten_pct": False, "title": None}
        )
        # A joint filing is one buyer, named by its lowest CIK; every owner's
        # role counts for it, whatever order the rows come in.
        cur["cik"] = min(cur["cik"], cik)
        cur["officer"] |= "Officer" in rel
        cur["director"] |= "Director" in rel
        cur["ten_pct"] |= "TenPercentOwner" in rel
        cur["title"] = cur["title"] or title
    out = []
    for r in _tsv(z, "NONDERIV_TRANS.tsv"):
        acc = r.get("ACCESSION_NUMBER")
        code = (r.get("TRANS_CODE") or "").strip()
        if acc not in subs or acc not in owners or code not in KEPT_CODES:
            continue
        when = _day(r.get("TRANS_DATE", ""))
        if when is None:
            continue
        issuer, filed, plan = subs[acc]
        o = owners[acc]
        out.append(
            Txn(
                accession=acc,
                issuer_cik=issuer,
                owner_cik=o["cik"],
                officer=o["officer"],
                director=o["director"],
                ten_pct=o["ten_pct"],
                title=o["title"],
                code=code,
                acq_disp=(r.get("TRANS_ACQUIRED_DISP_CD") or "").strip() or None,
                shares=_num(r.get("TRANS_SHARES")),
                price=_num(r.get("TRANS_PRICEPERSHARE")),
                trans_date=when,
                filed=filed,
                direct=(r.get("DIRECT_INDIRECT_OWNERSHIP") or "").strip() or None,
                plan=plan,
                source="dataset",
            )
        )
    return out


# ── one Form 4, from the daily index ──────────────────────────────────────


def _text(node, path: str) -> str | None:
    el = node.find(path)
    if el is None:
        return None
    return (el.text or "").strip() or None


def _bool(raw: str | None) -> bool:
    return (raw or "").strip().lower() in ("1", "true")


def parse_form4(text: str, accession: str, filed: date) -> list[Txn]:
    """The kept transactions of one Form 4's full submission text. Pure."""
    m = re.search(r"<ownershipDocument>.*?</ownershipDocument>", text, re.S)
    if not m:
        return []
    try:
        doc = ET.fromstring(m.group(0))
    except ET.ParseError:
        return []
    if (_text(doc, "documentType") or "") != "4":
        return []
    try:
        issuer = int(_text(doc, "issuer/issuerCik") or "")
    except ValueError:
        return []
    owners = []
    for ro in doc.findall("reportingOwner"):
        try:
            cik = int(_text(ro, "reportingOwnerId/rptOwnerCik") or "")
        except ValueError:
            continue
        rel = ro.find("reportingOwnerRelationship")
        owners.append(
            {
                "cik": cik,
                "officer": _bool(_text(rel, "isOfficer")) if rel is not None else False,
                "director": _bool(_text(rel, "isDirector")) if rel is not None else False,
                "ten_pct": _bool(_text(rel, "isTenPercentOwner")) if rel is not None else False,
                "title": _text(rel, "officerTitle") if rel is not None else None,
            }
        )
    if not owners:
        return []
    first = min(owners, key=lambda o: o["cik"])
    role = {k: any(o[k] for o in owners) for k in ("officer", "director", "ten_pct")}
    plan = plan_flag(_text(doc, "aff10b5One"))
    out = []
    for t in doc.findall("nonDerivativeTable/nonDerivativeTransaction"):
        code = _text(t, "transactionCoding/transactionCode") or ""
        when = _day(_text(t, "transactionDate/value") or "")
        if code not in KEPT_CODES or when is None:
            continue
        out.append(
            Txn(
                accession=accession,
                issuer_cik=issuer,
                owner_cik=first["cik"],
                title=first["title"],
                code=code,
                acq_disp=_text(t, "transactionAmounts/transactionAcquiredDisposedCode/value"),
                shares=_num(_text(t, "transactionAmounts/transactionShares/value")),
                price=_num(_text(t, "transactionAmounts/transactionPricePerShare/value")),
                trans_date=when,
                filed=filed,
                direct=_text(t, "ownershipNature/directOrIndirectOwnership/value"),
                plan=plan,
                source="daily",
                **role,
            )
        )
    return out


def form4_rows(index_text: str, issuers: set[int]) -> list[tuple[str, str]]:
    """``(accession, archive path)`` of each Form 4 filed on one of ``issuers``. Pure.

    A Form 4 is listed once under each party — the issuer and each owner —
    so only the issuer's line is kept, and each accession once.
    """
    out, seen = [], set()
    for line in index_text.splitlines():
        parts = line.split("|")
        if len(parts) != 5 or parts[2].strip() != "4":
            continue
        try:
            cik = int(parts[0])
        except ValueError:
            continue
        path = parts[4].strip()
        accession = path.rsplit("/", 1)[-1].removesuffix(".txt")
        if cik in issuers and accession not in seen:
            seen.add(accession)
            out.append((accession, path))
    return out


# ── syncing ───────────────────────────────────────────────────────────────

GetText = Callable[[str], str]
GetBytes = Callable[[str], bytes | None]  # None when the file does not exist (404)


def _sec_get(url: str):
    import httpx

    from advisor.news.edgar import _client_ready
    from advisor.research.config import get_settings

    _client_ready()
    return httpx.get(url, headers={"User-Agent": get_settings().edgar_user_agent}, timeout=120)


def sec_text(url: str) -> str:
    r = _sec_get(url)
    r.raise_for_status()
    return r.text


def sec_bytes(url: str) -> bytes | None:
    """The file, or None when it does not exist (404: a holiday has no index).

    A 403 is *not* absence: the SEC answers 403 as well as 429 when it is
    refusing a client, and a refused day read as empty would be marked done
    and never asked for again.
    """
    r = _sec_get(url)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return r.content


def refused(exc: Exception) -> bool:
    """True when the SEC is refusing us (429, 403), as opposed to a one-off failure."""
    response = getattr(exc, "response", None)
    return getattr(response, "status_code", None) in (403, 429)


def _insert(store: BreadthStore, txns: list[Txn]) -> int:
    before = store.conn.total_changes
    store.conn.executemany(
        "INSERT OR IGNORE INTO breadth_insider_txns (accession, issuer_cik, owner_cik, officer, "
        "director, ten_pct, title, code, acq_disp, shares, price, trans_date, filed, direct, "
        "plan, source) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
        [
            (t.accession, t.issuer_cik, t.owner_cik, int(t.officer), int(t.director),
             int(t.ten_pct), t.title, t.code, t.acq_disp, t.shares, t.price,
             t.trans_date.isoformat(), t.filed.isoformat(), t.direct,
             None if t.plan is None else int(t.plan), t.source)
            for t in txns
        ],
    )  # fmt: skip
    return store.conn.total_changes - before


def _quarter_of(d: date) -> tuple[int, int]:
    return d.year, (d.month - 1) // 3 + 1


def _quarter_start(year: int, q: int) -> date:
    return date(year, 3 * q - 2, 1)


def sync_insiders(
    store: BreadthStore,
    now: datetime,
    issuers: set[int],
    *,
    get_text: GetText = sec_text,
    get_bytes: GetBytes = sec_bytes,
) -> dict:
    """Data sets not yet loaded, then each day since the last one from the daily index.

    Never raises. A data set is loaded once; a day is read once it is over.
    The daily path reads only Form 4s filed on ``issuers`` (the E0 names).
    """
    store.conn.executescript(_SCHEMA)
    done = {r[0] for r in store.conn.execute("SELECT key FROM breadth_insider_state")}
    report = {"datasets": 0, "days": 0, "filings": 0, "rows": 0, "errors": []}

    try:
        links = dataset_links(get_text(DATASETS_PAGE))
    except Exception as exc:  # noqa: BLE001
        links = {}
        report["errors"].append(f"data set index: {exc}")
    published = sorted(k for k in links if int(k[:4]) >= FIRST_YEAR)
    for key in published:
        if key in done:
            continue
        try:
            blob = get_bytes(links[key])
            if blob is None:
                raise FileNotFoundError("linked from the index page but not found")
            txns = parse_dataset(blob)
        except Exception as exc:  # noqa: BLE001
            report["errors"].append(f"{key}: {exc}")
            if refused(exc):
                report["refused"] = True
                return report
            continue  # not marked done: retried next run
        with store.conn:
            n = _insert(store, txns)
            store.conn.execute(
                "INSERT OR REPLACE INTO breadth_insider_state VALUES (?, ?, ?)",
                (key, n, now.isoformat()),
            )
        report["datasets"] += 1
        report["rows"] += n

    # Days after the last published quarter, up to yesterday (today's index is not final).
    if published:
        y, q = int(published[-1][:4]), int(published[-1][-1])
        y, q = (y + 1, 1) if q == 4 else (y, q + 1)
        day = _quarter_start(y, q)
    else:
        day = now.date() - timedelta(days=90)
    last = now.date() - timedelta(days=1)
    while day <= last:
        key = day.isoformat()
        if key in done or day.weekday() >= 5:
            day += timedelta(days=1)
            continue
        yy, qq = _quarter_of(day)
        try:
            index = get_bytes(DAILY_INDEX_URL.format(year=yy, q=qq, ymd=day.strftime("%Y%m%d")))
            rows = form4_rows(index.decode("latin-1"), issuers) if index else []
            txns: list[Txn] = []
            for accession, path in rows:
                txns += parse_form4(get_text(FILING_URL.format(path=path)), accession, day)
        except Exception as exc:  # noqa: BLE001
            report["errors"].append(f"{key}: {exc}")
            if refused(exc):
                # Asking again tomorrow's worth of days would only extend the
                # block. Stop; everything from this day on is read next run.
                report["refused"] = True
                break
            day += timedelta(days=1)
            continue  # not marked done: retried next run
        with store.conn:
            n = _insert(store, txns)
            store.conn.execute(
                "INSERT OR REPLACE INTO breadth_insider_state VALUES (?, ?, ?)",
                (key, n, now.isoformat()),
            )
        report["days"] += 1
        report["filings"] += len(rows)
        report["rows"] += n
        day += timedelta(days=1)
    report["errors"] = report["errors"][:20]
    return report


def load_purchases(store: BreadthStore) -> list[Txn]:
    """Every open-market purchase on file, oldest first."""
    store.conn.executescript(_SCHEMA)
    out = []
    for r in store.conn.execute(
        "SELECT accession, issuer_cik, owner_cik, officer, director, ten_pct, title, code, "
        "acq_disp, shares, price, trans_date, filed, direct, plan, source "
        "FROM breadth_insider_txns WHERE code = 'P' ORDER BY filed"
    ):
        out.append(
            Txn(
                accession=r[0], issuer_cik=r[1], owner_cik=r[2], officer=bool(r[3]),
                director=bool(r[4]), ten_pct=bool(r[5]), title=r[6], code=r[7],
                acq_disp=r[8], shares=r[9], price=r[10],
                trans_date=date.fromisoformat(r[11]), filed=date.fromisoformat(r[12]),
                direct=r[13], plan=None if r[14] is None else bool(r[14]), source=r[15],
            )
        )  # fmt: skip
    return out
