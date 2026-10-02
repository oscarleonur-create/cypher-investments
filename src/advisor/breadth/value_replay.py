"""Does value add to a pick? The pick records, valued as of their own day.

User, 2026-09-30: picks carry no conclusion — *"posible entry, por esta tesis,
el valor de la empresa está por debajo de lo que esperamos, debería estar en X
valor"*. The value range exists (``valuation.rationale.price_range``: bear /
base / bull from the company's own filed margins), but it was never tested on
picks. Before a pick says "worth $X", this measures whether the picks priced
below their value did any better than those priced above it.

**Each record is valued with only what was public on its day.** The SEC's
``companyfacts`` gives every fact with the date it was *filed*; a record on
day *d* sees only facts filed on or before *d*:

- flows (revenue, operating cash flow, capex, operating income) through the
  same pure code the live card uses (``figures.flows_from_rows``);
- the balance at the latest instant filed by *d*: cash plus short-term
  securities, less debt (the first debt concept reported at that instant; a
  filing with no debt figure at all is debt-free, one with debt figures only
  at other instants is left unvalued — never assumed zero);
- diluted weighted-average shares of the latest period, put on the basis of
  the stored split-adjusted closes: multiplied by every split dated after the
  filing that reported them.

Then the live engine unchanged: ``implied.build_snapshot`` → ``price_range``.

**Buckets** (``bucket``): below its base value (``below_bear`` or
``above_bear``), above it (``above_base`` or ``above_bull``), no range but the
price asks for a margin the company has never filed (``no_support``), no
range but it asks for less (``one_reading``), or not valued (``no_data``).

The outcomes are the signal replay's own (``breadth_records``: excess over
matched peers at 20, 60 and 120 sessions), judged the same way.

Not seen: restatements are taken as first filed (the filter is by filing
date, so a later restatement cannot leak — but a first filing's error stays);
a stock with no split history from Yahoo is taken as never split.
"""

from __future__ import annotations

import gzip
import json
import logging
import sqlite3
import statistics
from datetime import date, datetime
from pathlib import Path

logger = logging.getLogger(__name__)

GROUPS = ("F+P", "2+", "F")
HORIZONS = {"d20": 20, "d60": 60, "d120": 120}
# A valuation from a period that ended longer ago than this describes another
# business (the live card calls it stale at a similar age).
MAX_PERIOD_AGE_DAYS = 200

FACTS_URL = "https://data.sec.gov/api/xbrl/companyfacts/CIK{cik:010d}.json"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS value_facts (
    cik        INTEGER PRIMARY KEY,
    fetched_at TEXT NOT NULL,
    status     INTEGER NOT NULL,
    blob       BLOB
);
CREATE TABLE IF NOT EXISTS value_splits (
    symbol     TEXT PRIMARY KEY,
    fetched_at TEXT NOT NULL,
    splits     TEXT NOT NULL
);
"""


def _concepts() -> dict[str, tuple[str, ...]]:
    from advisor.valuation.figures import (
        CAPEX_CONCEPTS,
        OCF_CONCEPTS,
        OPERATING_CONCEPTS,
        REVENUE_CONCEPTS,
    )
    from advisor.valuation.fundamentals import (
        CASH_CONCEPTS,
        DEBT_CONCEPTS,
        SECURITIES_CONCEPTS,
    )
    from advisor.valuation.history import SHARES_CONCEPT

    return {
        "revenue": REVENUE_CONCEPTS,
        "ocf": OCF_CONCEPTS,
        "capex": CAPEX_CONCEPTS,
        "operating": OPERATING_CONCEPTS,
        "cash": CASH_CONCEPTS,
        "securities": SECURITIES_CONCEPTS,
        "debt": DEBT_CONCEPTS,
        "shares": (SHARES_CONCEPT,),
    }


# ── pure: facts as of a day ─────────────────────────────────────────────────


def slim(facts: dict) -> dict[str, list[dict]]:
    """Keep only the us-gaap concepts a valuation reads, one row list each. Pure."""
    gaap = (facts.get("facts") or {}).get("us-gaap") or {}
    out: dict[str, list[dict]] = {}
    for names in _concepts().values():
        for name in names:
            units = (gaap.get(name) or {}).get("units") or {}
            rows = units.get("USD") or units.get("shares") or next(iter(units.values()), [])
            if rows:
                out[name] = [
                    {
                        k: r[k]
                        for k in ("start", "end", "val", "form", "filed", "frame", "accn")
                        if k in r
                    }
                    for r in rows
                ]
    return out


def as_of(rows: list[dict], day: date) -> list[dict]:
    """The rows public by ``day``: filed on or before it. Pure."""
    d = day.isoformat()
    return [r for r in rows if str(r.get("filed", "9999")) <= d]


def _instants(rows: list[dict]) -> dict[str, tuple[float, str]]:
    """One (value, filing accession) per instant end: the latest filing's. Pure."""
    best: dict[str, tuple[str, float, str]] = {}
    for r in rows:
        if r.get("start") or not r.get("end"):
            continue
        try:
            v = float(r["val"])
        except (KeyError, TypeError, ValueError):
            continue
        filed = str(r.get("filed", ""))
        if r["end"] not in best or filed >= best[r["end"]][0]:
            best[r["end"]] = (filed, v, str(r.get("accn", "")))
    return {e: (v, a) for e, (_, v, a) in best.items()}


def balance(facts: dict[str, list[dict]], day: date) -> tuple[float | None, date | None, str]:
    """Net cash at the latest instant public by ``day``. Pure.

    Returns ``(net_cash, instant, note)``. Debt is judged like the live
    reader judges one filing (``fundamentals._total_debt``): the filing that
    reported this cash either carries a debt figure at this instant (used), no
    debt figure at all (debt-free: zero), or debt figures only at other
    instants (unparseable here: None, never zero).
    """
    names = _concepts()
    cash_at: dict[str, tuple[float, str]] = {}
    for c in names["cash"]:
        for end, va in _instants(as_of(facts.get(c, []), day)).items():
            cash_at.setdefault(end, va)
    if not cash_at:
        return None, None, "no cash reported"
    end = max(cash_at)
    cash, accn = cash_at[end]
    for c in names["securities"]:
        got = _instants(as_of(facts.get(c, []), day)).get(end)
        if got is not None:
            cash += got[0]
            break
    debt, mentioned = None, False
    for c in names["debt"]:
        rows = as_of(facts.get(c, []), day)
        mentioned = mentioned or any(str(r.get("accn", "")) == accn for r in rows)
        got = _instants(rows).get(end)
        if got is not None:
            debt = got[0]
            break
    if debt is None:
        if mentioned:
            return None, date.fromisoformat(end), "debt in the filing but not at this instant"
        debt = 0.0
    return cash - debt, date.fromisoformat(end), ""


def shares(
    facts: dict[str, list[dict]], day: date, splits: list[tuple[date, float]]
) -> tuple[float | None, date | None]:
    """Diluted shares of the latest period public by ``day``, on today's split basis. Pure."""
    from advisor.valuation.history import SHARES_CONCEPT

    best = None
    for r in as_of(facts.get(SHARES_CONCEPT, []), day):
        if not r.get("end") or not r.get("start"):
            continue
        key = (r["end"], str(r.get("filed", "")))
        if best is None or key > best[0]:
            best = (key, r)
    if best is None:
        return None, None
    r = best[1]
    try:
        value = float(r["val"])
    except (TypeError, ValueError):
        return None, None
    filed = date.fromisoformat(str(r["filed"]))
    for when, ratio in splits:
        if when > filed and ratio and ratio > 0:
            value *= ratio
    return value, date.fromisoformat(r["end"])


def value_on(
    symbol: str,
    day: date,
    price: float,
    facts: dict[str, list[dict]],
    splits: list[tuple[date, float]],
) -> dict:
    """The live price-range card, computed as of ``day`` from what was public. Pure."""
    from advisor.valuation.figures import Figures, build_figures, flows_from_rows
    from advisor.valuation.implied import build_snapshot
    from advisor.valuation.rationale import price_range

    names = _concepts()

    def rows(kind):
        return [as_of(facts.get(c, []), day) for c in names[kind]]

    flows = flows_from_rows(rows("revenue"), rows("ocf"), rows("capex"), rows("operating"))
    if flows is None or flows.revenue_ttm is None:
        return {"bucket": "no_data", "why": "no revenue filed"}
    if (day - flows.period_end).days > MAX_PERIOD_AGE_DAYS:
        return {"bucket": "no_data", "why": f"latest period {flows.period_end} is stale"}
    net_cash, asof, note = balance(facts, day)
    if net_cash is None:
        return {"bucket": "no_data", "why": note}
    n, _ = shares(facts, day, splits)
    if not n:
        return {"bucket": "no_data", "why": "no diluted share count"}
    fig: Figures = build_figures(symbol, price, None, flows, price_source="breadth close")
    fig = fig.model_copy(update={"shares": n, "net_cash": net_cash, "balance_asof": asof})
    snap = build_snapshot(fig, price, asof=day)
    if snap is None:
        return {"bucket": "no_data", "why": "the engine refused the inputs"}
    card = price_range(snap, today=day)
    cases = {c.name: c for c in card.cases}
    own = [m.value for m in card.own_margins if m.value is not None]
    market = cases.get("market")
    out = {
        "verdict": card.verdict,
        "refused": card.refused,
        "base": cases["base"].value_per_share if "base" in cases else None,
        "bear": cases["bear"].value_per_share if "bear" in cases else None,
        "bull": cases["bull"].value_per_share if "bull" in cases else None,
        "market_margin": market.margin if market else None,
        "best_margin": max(own) if own else None,
        "growth": card.growth,
        "period_end": flows.period_end.isoformat(),
    }
    out["bucket"] = bucket(out)
    return out


def bucket(v: dict) -> str:
    """Where the price sits against the value, in five plain groups. Pure."""
    verdict = v.get("verdict")
    if verdict in ("below_bear", "above_bear"):
        return "below_base"
    if verdict in ("above_base", "above_bull"):
        return "above_base"
    need, best = v.get("market_margin"), v.get("best_margin")
    if need is None:
        return "no_data"
    if best is None or best <= 0 or need > best:
        return "no_support"
    return "one_reading"


# ── evaluation ──────────────────────────────────────────────────────────────


BUCKETS = ("below_base", "above_base", "one_reading", "no_support", "no_data")


def evaluate(rows: list[dict]) -> list[dict]:
    """One cell per (group, bucket, horizon): excess over peers, judged like the replay."""
    from advisor.learning.evaluate import MIN_SESSIONS, blocks_of, cluster_ci, verdict

    cells = []
    for grp in GROUPS:
        for b in BUCKETS:
            members = [r for r in rows if r["grp"] == grp and r["value"]["bucket"] == b]
            for h, k in HORIZONS.items():
                have = [r for r in members if (r.get("outcomes") or {}).get(h)]
                cell = {"group": grp, "bucket": b, "horizon": h, "n": len(have)}
                if not have:
                    cells.append({**cell, "verdict": "UNDETERMINED"})
                    continue
                values = [(r["day"], r["outcomes"][h]["excess"]) for r in have]
                windows = len(blocks_of(values, k))
                ci = cluster_ci(values, block=k) if windows >= MIN_SESSIONS else None
                v, why = verdict(len(values), windows, ci)
                cells.append(
                    {
                        **cell,
                        "mean": statistics.fmean(r["outcomes"][h]["ret"] for r in have),
                        "excess": statistics.fmean(x for _, x in values),
                        "beat": sum(x > 0 for _, x in values) / len(values),
                        "ci": list(ci) if ci else None,
                        "windows": windows,
                        "verdict": v.value,
                        "reason": why,
                    }
                )
    return cells


# ── live: fetch, cache, run ─────────────────────────────────────────────────


def _cache(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.executescript(_SCHEMA)
    return conn


def cached_facts(conn: sqlite3.Connection, cik: int) -> dict[str, list[dict]] | None:
    """The slimmed companyfacts for ``cik``: from the cache, else the SEC (then cached)."""
    import httpx

    from advisor.news.edgar import _client_ready
    from advisor.research.config import get_settings

    row = conn.execute("SELECT status, blob FROM value_facts WHERE cik = ?", (cik,)).fetchone()
    if row is not None:
        return json.loads(gzip.decompress(row[1])) if row[0] == 200 and row[1] else None
    _client_ready()
    try:
        r = httpx.get(
            FACTS_URL.format(cik=cik),
            headers={"User-Agent": get_settings().edgar_user_agent},
            timeout=60,
        )
    except httpx.HTTPError as exc:
        logger.warning("value_replay: companyfacts %s failed: %s", cik, exc)
        return None  # not cached: tried again next run
    blob = gzip.compress(json.dumps(slim(r.json())).encode()) if r.status_code == 200 else None
    with conn:
        conn.execute(
            "INSERT OR REPLACE INTO value_facts VALUES (?, ?, ?, ?)",
            (cik, datetime.now().isoformat(), r.status_code, blob),
        )
    return json.loads(gzip.decompress(blob)) if blob else None


def cached_splits(conn: sqlite3.Connection, symbol: str) -> list[tuple[date, float]]:
    from advisor.valuation.history import _splits

    row = conn.execute("SELECT splits FROM value_splits WHERE symbol = ?", (symbol,)).fetchone()
    if row is None:
        got = _splits(symbol)
        with conn:
            conn.execute(
                "INSERT OR REPLACE INTO value_splits VALUES (?, ?, ?)",
                (symbol, datetime.now().isoformat(),
                 json.dumps([(d.isoformat(), r) for d, r in got])),
            )  # fmt: skip
        return got
    return [(date.fromisoformat(d), float(r)) for d, r in json.loads(row[0])]


def run(breadth_db: Path, run_id: str, cache_path: Path, *, progress=None) -> dict:
    """Value every record of ``run_id`` in ``GROUPS`` as of its day; judge by bucket."""
    bconn = sqlite3.connect(str(breadth_db))
    q = ",".join("?" * len(GROUPS))
    recs = [
        {"grp": g, "symbol": s, "cik": c, "day": date.fromisoformat(d),
         "outcomes": json.loads(o) if o else None}
        for g, s, c, d, o in bconn.execute(
            f"SELECT grp, symbol, cik, day, outcomes_json FROM breadth_records "
            f"WHERE run_id = ? AND grp IN ({q})",
            (run_id, *GROUPS),
        )
    ]  # fmt: skip
    for r in recs:
        row = bconn.execute(
            "SELECT close FROM breadth_bars WHERE symbol = ? AND day = ?",
            (r["symbol"], r["day"].isoformat()),
        ).fetchone()
        r["price"] = row[0] if row else None
    bconn.close()

    cache = _cache(cache_path)
    facts_by: dict[int, dict | None] = {}
    splits_by: dict[str, list] = {}
    for k, r in enumerate(recs):
        if progress and k % 200 == 0:
            progress(k, len(recs))
        if not r["cik"] or not r["price"]:
            r["value"] = {"bucket": "no_data", "why": "no CIK or price"}
            continue
        if r["cik"] not in facts_by:
            facts_by[r["cik"]] = cached_facts(cache, int(r["cik"]))
        facts = facts_by[r["cik"]]
        if not facts:
            r["value"] = {"bucket": "no_data", "why": "no companyfacts"}
            continue
        if r["symbol"] not in splits_by:
            splits_by[r["symbol"]] = cached_splits(cache, r["symbol"])
        try:
            r["value"] = value_on(r["symbol"], r["day"], r["price"], facts, splits_by[r["symbol"]])
        except Exception as exc:  # noqa: BLE001 — one bad company must not stop the study
            r["value"] = {"bucket": "no_data", "why": f"error: {exc}"}
    cache.close()
    counts: dict[str, dict[str, int]] = {}
    for r in recs:
        counts.setdefault(r["grp"], {}).setdefault(r["value"]["bucket"], 0)
        counts[r["grp"]][r["value"]["bucket"]] += 1
    return {"run_id": run_id, "records": len(recs), "counts": counts,
            "cells": evaluate(recs), "rows": recs}  # fmt: skip


# ── stored runs: what a pick's verdict quotes ───────────────────────────────

_RUNS_SCHEMA = """
CREATE TABLE IF NOT EXISTS breadth_value_runs (
    id           TEXT PRIMARY KEY,
    started_at   TEXT NOT NULL,
    rules        TEXT NOT NULL,
    years        INTEGER NOT NULL,
    signal_run   TEXT NOT NULL,
    summary_json TEXT NOT NULL
);
"""


def replay_value(breadth_db: Path, now: datetime, *, years: int = 3, progress=None) -> dict:
    """Value the latest signal replay of ``years`` (current rules); store the cells.

    The companyfacts cache lives beside the run, in the breadth store: the
    first run downloads ~1,500 companies (about 25 minutes), later ones reuse it.
    """
    import uuid

    from advisor.breadth.ruleset import signal_rules, value_rules

    conn = sqlite3.connect(str(breadth_db))
    signal = None
    for rid, params in conn.execute(
        "SELECT id, params_json FROM breadth_replay_runs WHERE rules = ? ORDER BY started_at DESC",
        (signal_rules().version,),
    ):
        if int(json.loads(params).get("years", 0)) == years:
            signal = rid
            break
    conn.close()
    if signal is None:
        return {"ok": False, "error": f"no {years}-year signal replay under the current rules"}
    out = run(breadth_db, signal, breadth_db, progress=progress)
    out.pop("rows")
    run_id = f"{now:%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    summary = {"ok": True, "id": run_id, "years": years, "signal_run": signal,
               "rules": value_rules().version, **out}  # fmt: skip
    conn = sqlite3.connect(str(breadth_db))
    with conn:
        conn.executescript(_RUNS_SCHEMA)
        conn.execute(
            "INSERT INTO breadth_value_runs VALUES (?, ?, ?, ?, ?, ?)",
            (run_id, now.isoformat(), summary["rules"], years, signal,
             json.dumps(summary, default=str)),
        )  # fmt: skip
    conn.close()
    return summary


def latest_value_runs(store) -> list[dict]:
    """The latest value test of each window under the current rules, longest first."""
    from advisor.breadth.ruleset import value_rules

    store.conn.executescript(_RUNS_SCHEMA)
    current = value_rules().version
    latest: dict[int, dict] = {}
    for summary, rules, years in store.conn.execute(
        "SELECT summary_json, rules, years FROM breadth_value_runs ORDER BY started_at DESC"
    ):
        if rules == current and years not in latest:
            latest[years] = json.loads(summary)
    return [latest[y] for y in sorted(latest, reverse=True)]
