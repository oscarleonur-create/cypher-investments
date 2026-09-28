"""What each group's records went on to do, against names like them the same day.

**The null is matched controls, not the name's own drift.** The learning
loop scores a record against its symbol's average return; for momentum that
is circular — a name in the top decile for a year has a high average return
*because* it went up. Here each record is compared with ``CONTROLS`` names
drawn at random from the same day's eligible universe, matched on industry
(SIC two-digit major group) and size (quintile of median dollar volume).
When a match leaves fewer than ``MIN_POOL`` names, the match is widened —
industry alone, then SIC division with size, then size, then everything —
and the level used is recorded on the outcome, so a report can show how
much of the comparison was like for like.

**Statistics are the learning loop's.** Records are grouped by session and
resampled in blocks as long as the horizon (``learning.evaluate``): two years
hold about eight independent 60-session windows and the verdict says so.
UNDETERMINED is the default; EDGE needs an interval that clears zero.

**What a replay cannot see.** The universe is today's symbol directory, so
names delisted since are missing (survivorship flatters every long rule).
Revenue is the latest filed value, so restatements leak. Returns leave out
dividends. News, the model's reading and the user's judgement are absent.
Every report repeats these.
"""

from __future__ import annotations

import json
import logging
import statistics
import uuid
import zlib
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from advisor.breadth import signals as S
from advisor.breadth.companies import division, major_group, sic_map
from advisor.breadth.filings import FilingDates
from advisor.breadth.insiders import load_purchases
from advisor.breadth.panel import Panel, load_panel
from advisor.breadth.store import BreadthStore
from advisor.breadth.universe import eligibility_panel
from advisor.learning.evaluate import MIN_SESSIONS, blocks_of, cluster_ci, verdict
from advisor.valuation.history import (
    ANNUAL_LAG,
    REVENUE_CONCEPTS,
    Point,
    merge_concepts,
    quarterly_points,
)

logger = logging.getLogger(__name__)

HORIZONS: dict[str, int] = {"d20": 20, "d60": 60, "d120": 120}
TAILS: dict[str, int] = {"mae20": 20, "mae60": 60}
CONTROLS = 10
MIN_POOL = 10
SIZE_BUCKETS = 5
# The second comparison matches the past this many sessions: insiders buy
# after falls (median -12.5% over the 60 sessions before a cluster, measured
# 2026-09-28), and "did it beat peers" then mostly asks whether falls
# continued. "Did it beat peers that fell as much" asks what the signal adds.
TREND_SESSIONS = 60
# A future close may be missing (a halt); the last close within this many
# sessions stands in. Beyond it the outcome is left empty, never zero.
FILL_LIMIT = 5
# Live records need this much history behind the session they are made on:
# a year for momentum and E0, plus the convergence and cooldown windows.
LIVE_HISTORY_DAYS = 600

CAVEATS = (
    "universe is today's listings: names delisted since are missing (survivorship)",
    "revenue is the latest filed value: restatements leak into the past",
    "returns are split-adjusted closes without dividends",
    "F is dated by the 10-Q/10-K that carried it (the earnings release is usually days "
    "earlier); where no filing is found, 45 days after the quarter (75 for a fiscal fourth)",
    "no news, no model reading, no user judgement: signals only",
)


# ── inputs ────────────────────────────────────────────────────────────────


def cik_map(store: BreadthStore) -> dict[str, int]:
    """Symbol -> CIK from the latest universe snapshot."""
    day = store.latest_universe_day()
    if day is None:
        return {}
    return {
        r[0]: r[1]
        for r in store.conn.execute(
            "SELECT symbol, cik FROM breadth_universe WHERE day = ? AND cik IS NOT NULL",
            (day.isoformat(),),
        )
    }


def dated(
    quarters: dict[tuple[int, int], Point], cik: int, dates: FilingDates | None, tally: dict
) -> dict[tuple[int, int], Point]:
    """Each point dated by the report that disclosed it; the fixed lag only when none is found.

    A point whose lag is ``ANNUAL_LAG`` was derived from the annual figure,
    so it waits for the annual report; any other waits for a 10-Q. Pure.
    """
    out = {}
    for key, p in quarters.items():
        annual = (p.known - p.end) == ANNUAL_LAG
        filed = dates.first_after(cik, p.end, annual=annual) if dates else None
        if filed is not None:
            out[key] = p.model_copy(update={"known": filed})
            tally["filing"] = tally.get("filing", 0) + 1
        else:
            out[key] = p
            tally["lag"] = tally.get("lag", 0) + 1
    return out


def revenue_quarters(
    store: BreadthStore,
    cik_of: dict[str, int],
    dates: FilingDates | None = None,
    tally: dict | None = None,
) -> dict[str, dict[tuple[int, int], Point]]:
    """Each symbol's quarterly revenue, concepts merged, dated by its filing."""
    tally = tally if tally is not None else {}
    by_cik: dict[int, dict[str, list[dict]]] = {}
    marks = ",".join("?" * len(REVENUE_CONCEPTS))
    for cik, concept, frame, val, start, end in store.conn.execute(
        "SELECT cik, concept, frame, val, start, end FROM breadth_facts "
        f"WHERE concept IN ({marks})",
        REVENUE_CONCEPTS,
    ):
        by_cik.setdefault(cik, {}).setdefault(concept, []).append(
            {"frame": frame, "val": val, "start": start, "end": end}
        )
    wanted = set(cik_of.values())
    quarters_of_cik = {
        cik: dated(
            merge_concepts([quarterly_points(rows.get(c, [])) for c in REVENUE_CONCEPTS]),
            cik,
            dates,
            tally,
        )
        for cik, rows in by_cik.items()
        if cik in wanted
    }
    return {s: quarters_of_cik[c] for s, c in cik_of.items() if quarters_of_cik.get(c)}


# ── outcomes ──────────────────────────────────────────────────────────────


class Outcomes:
    """Forward returns and matched controls over one panel."""

    def __init__(
        self,
        panel: Panel,
        eligible: pd.DataFrame,
        cik_of: dict[str, int],
        sic_of: dict[int, int | None],
    ) -> None:
        self.panel = panel
        self.symbols = list(panel.close.columns)
        self.col = {s: i for i, s in enumerate(self.symbols)}
        self.eligible = eligible.reindex(columns=self.symbols, fill_value=False).to_numpy()
        close = panel.close
        ahead = close.ffill(limit=FILL_LIMIT)
        self.fwd = {h: (ahead.shift(-k) / close - 1).to_numpy() for h, k in HORIZONS.items()}
        rev = ahead.iloc[::-1]
        self.mae = {
            h: (rev.rolling(k, min_periods=1).min().iloc[::-1].shift(-1) / close - 1).to_numpy()
            for h, k in TAILS.items()
        }
        for h, k in TAILS.items():  # a window not yet complete is no tail yet
            self.mae[h][len(close) - k :, :] = np.nan
        self.dv = (close * panel.volume).rolling(20, min_periods=1).median().to_numpy()
        sic = [sic_of.get(cik_of.get(s)) if cik_of.get(s) else None for s in self.symbols]
        self.group = np.array([major_group(x) or -1 for x in sic])
        self.division = np.array([division(x) or "" for x in sic])
        # The past TREND_SESSIONS' return, for the second comparison.
        self.trend = (close / close.shift(TREND_SESSIONS) - 1).to_numpy()
        self._cache: dict[tuple[str, int], np.ndarray] = {}

    def _buckets(self, kind: str, row: int) -> np.ndarray:
        """Quintile of ``kind`` ('size' or 'trend') among the session's eligible; -1 unranked."""
        if (kind, row) not in self._cache:
            v = (self.dv if kind == "size" else self.trend)[row]
            ok = self.eligible[row] & np.isfinite(v)
            bucket = np.full(len(v), -1)
            if ok.sum() >= SIZE_BUCKETS:
                ranks = pd.Series(v[ok]).rank(pct=True).to_numpy()
                bucket[ok] = np.minimum((ranks * SIZE_BUCKETS).astype(int), SIZE_BUCKETS - 1)
            self._cache[(kind, row)] = bucket
        return self._cache[(kind, row)]

    def controls(self, symbol: str, row: int, *, trend: bool = False) -> tuple[list[int], str]:
        """Up to ``CONTROLS`` like names that session, and how like them they are.

        ``trend=True`` is the second comparison: peers whose last
        ``TREND_SESSIONS`` went like this name's (same quintile), so a record
        on a name that fell 12% is judged against others that fell too.
        """
        j = self.col[symbol]
        pool = self.eligible[row].copy()
        pool[j] = False
        size = self._buckets("size", row)
        same_group = (self.group == self.group[j]) & (self.group[j] != -1)
        same_div = (self.division == self.division[j]) & (self.division[j] != "")
        same_size = (size == size[j]) & (size[j] != -1)
        everyone = np.ones_like(pool)
        if trend:
            tr = self._buckets("trend", row)
            if tr[j] == -1:
                return [], "none"
            same_trend = tr == tr[j]
            levels = (
                ("industry+trend", same_group & same_trend),
                ("division+trend", same_div & same_trend),
                ("trend", same_trend),
            )
        else:
            levels = (
                ("industry+size", same_group & same_size),
                ("industry", same_group),
                ("division+size", same_div & same_size),
                ("size", same_size),
                ("all", everyone),
            )
        idx = np.array([], dtype=int)
        for level, mask in levels:
            idx = np.flatnonzero(pool & mask)
            if len(idx) >= MIN_POOL:
                break
        if len(idx) == 0:
            return [], "none"
        salt = ":trend" if trend else ""
        seed = zlib.crc32(f"{symbol}:{self.panel.sessions[row].date()}{salt}".encode())
        rng = np.random.default_rng(seed)
        pick = rng.choice(idx, size=min(CONTROLS, len(idx)), replace=False)
        return sorted(int(i) for i in pick), level

    def of(self, symbol: str, row: int) -> dict:
        """Outcomes of a record made at ``row``'s close. Horizons not yet reached are None."""
        j = self.col[symbol]
        picks, level = self.controls(symbol, row)
        tpicks, tlevel = self.controls(symbol, row, trend=True)
        out: dict = {
            "match": level,
            "controls": [self.symbols[i] for i in picks],
            "match_trend": tlevel,
        }
        for h in HORIZONS:
            ret = self.fwd[h][row, j]
            ctrl = self.fwd[h][row, picks] if picks else np.array([])
            ctrl = ctrl[np.isfinite(ctrl)]
            if not np.isfinite(ret) or len(ctrl) == 0:
                out[h] = None
                continue
            tctrl = self.fwd[h][row, tpicks] if tpicks else np.array([])
            tctrl = tctrl[np.isfinite(tctrl)]
            out[h] = {
                "ret": float(ret),
                "control": float(ctrl.mean()),
                "excess": float(ret - ctrl.mean()),
                "n_controls": int(len(ctrl)),
                "excess_trend": float(ret - tctrl.mean()) if len(tctrl) else None,
            }
        for h in TAILS:
            v = self.mae[h][row, j]
            out[h] = float(v) if np.isfinite(v) else None
        return out


# ── evaluation ────────────────────────────────────────────────────────────


def evaluate(recs: list[dict]) -> list[dict]:
    """One cell per (group, horizon): the record's return beyond its matched controls."""
    cells = []
    for grp in S.GROUPS:
        members = [r for r in recs if r["grp"] == grp]
        for h, k in HORIZONS.items():
            have = [r for r in members if (r.get("outcomes") or {}).get(h)]
            if not have:
                cells.append({"group": grp, "horizon": h, "n": 0, "verdict": "UNDETERMINED",
                              "reason": "no outcomes yet"})  # fmt: skip
                continue
            values = [(r["day"], r["outcomes"][h]["excess"]) for r in have]
            windows = len(blocks_of(values, k))
            ci = cluster_ci(values, block=k) if windows >= MIN_SESSIONS else None
            v, why = verdict(len(values), windows, ci)
            tail_key = "mae20" if k <= 20 else "mae60"
            tails = [
                r["outcomes"][tail_key] for r in have if r["outcomes"].get(tail_key) is not None
            ]
            levels: dict[str, int] = {}
            for r in have:
                levels[r["outcomes"]["match"]] = levels.get(r["outcomes"]["match"], 0) + 1
            # The second comparison: against peers whose past trend was alike.
            # Shown with its own interval; the verdict stays on the first.
            tvals = [
                (r["day"], r["outcomes"][h]["excess_trend"])
                for r in have
                if r["outcomes"][h].get("excess_trend") is not None
            ]
            twin = len(blocks_of(tvals, k)) if tvals else 0
            tci = cluster_ci(tvals, block=k) if twin >= MIN_SESSIONS else None
            cells.append(
                {
                    "group": grp,
                    "horizon": h,
                    "n": len(have),
                    "sessions": len({r["day"] for r in have}),
                    "windows": windows,
                    "mean": statistics.fmean(r["outcomes"][h]["ret"] for r in have),
                    "control": statistics.fmean(r["outcomes"][h]["control"] for r in have),
                    "excess": statistics.fmean(x for _, x in values),
                    "beat": sum(x > 0 for _, x in values) / len(values),
                    "ci": list(ci) if ci else None,
                    "tail": statistics.fmean(tails) if tails else None,
                    "tail_horizon": tail_key,
                    "match": levels,
                    "n_trend": len(tvals),
                    "excess_trend": statistics.fmean(x for _, x in tvals) if tvals else None,
                    "ci_trend": list(tci) if tci else None,
                    "verdict": v.value,
                    "reason": why,
                }
            )
    return cells


# ── runs ──────────────────────────────────────────────────────────────────


def _inputs(store: BreadthStore, panel: Panel, tally: dict | None = None, t=S.DEFAULT):
    cik_of = cik_map(store)
    eligible = eligibility_panel(panel.close, panel.volume)
    eligible = S.dedupe_share_classes(eligible, panel, cik_of)
    dates = FilingDates(store)
    quarters = revenue_quarters(store, cik_of, dates or None, tally)
    events = S.fundamental_events(quarters, panel.sessions, t)
    symbols_of: dict[int, list[str]] = {}
    for s, c in cik_of.items():
        symbols_of.setdefault(c, []).append(s)
    ievents = S.insider_events(load_purchases(store), symbols_of, panel.sessions, t)
    return cik_of, eligible, events, ievents


def _write(store, recs, origin: str, run_id: str, rules: str, cik_of, now: datetime) -> None:
    with store.conn:
        store.conn.executemany(
            "INSERT OR REPLACE INTO breadth_records (id, origin, run_id, grp, symbol, cik, day, "
            "rules, detail_json, outcomes_json, updated_at) VALUES (?,?,?,?,?,?,?,?,?,?,?)",
            [
                (
                    f"{origin}:{run_id}:{r['grp']}:{r['symbol']}:{r['day']}",
                    origin,
                    run_id,
                    r["grp"],
                    r["symbol"],
                    cik_of.get(r["symbol"]),
                    r["day"].isoformat(),
                    rules,
                    json.dumps(r["detail"]),
                    json.dumps(r.get("outcomes")),
                    now.isoformat(),
                )
                for r in recs
            ],
        )


def replay(
    store: BreadthStore,
    now: datetime,
    *,
    years: int = 2,
    t: S.Thresholds = S.DEFAULT,
    panel: Panel | None = None,
) -> dict:
    """Run the families day by day over the last ``years``, record, and judge."""
    from advisor.breadth.ruleset import signal_rules
    from advisor.learning.rules import code_rev

    rules = signal_rules(t)
    panel = panel if panel is not None else load_panel(store)
    if panel.close.empty:
        return {"ok": False, "error": "no bars on file: run `advisor breadth sync` first"}
    dating: dict[str, int] = {}
    cik_of, eligible, events, ievents = _inputs(store, panel, dating, t)
    last_row = len(panel.sessions) - 1
    first_row = panel.row_of(panel.sessions[-1].date() - timedelta(days=365 * years))
    if first_row is None or first_row < S.MOMENTUM_LOOKBACK:
        first_row = min(S.MOMENTUM_LOOKBACK, last_row)
    recs = S.records(panel, eligible, events, first_row, last_row, t, ievents)
    out = Outcomes(panel, eligible, cik_of, sic_map(store))
    for r in recs:
        r["outcomes"] = out.of(r["symbol"], r["row"])

    run_id = f"{now:%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    cells = evaluate(recs)
    eligible_days = eligible.iloc[first_row : last_row + 1].sum(axis=1)
    summary = {
        "ok": True,
        "run_id": run_id,
        "rules": rules.version,
        "code": code_rev(),
        "from": panel.sessions[first_row].date().isoformat(),
        "to": panel.sessions[last_row].date().isoformat(),
        "sessions": last_row - first_row + 1,
        "eligible_per_session": {
            "median": int(eligible_days.median()),
            "min": int(eligible_days.min()),
            "max": int(eligible_days.max()),
        },
        "records": {g: sum(1 for r in recs if r["grp"] == g) for g in S.GROUPS},
        "per_session": {
            g: round(sum(1 for r in recs if r["grp"] == g) / max(1, last_row - first_row + 1), 2)
            for g in S.GROUPS
        },
        "cells": cells,
        "cells_tested": len([c for c in cells if c["n"]]),
        "revenue_quarters_dated_by": dating,
        "caveats": list(CAVEATS),
    }
    with store.conn:
        store.conn.execute(
            "INSERT INTO breadth_replay_runs (id, started_at, finished_at, code, rules, "
            "params_json, summary_json) VALUES (?,?,?,?,?,?,?)",
            (
                run_id,
                now.isoformat(),
                datetime.now(now.tzinfo).isoformat(),
                code_rev(),
                rules.version,
                json.dumps({"years": years, "thresholds": t.__dict__}),
                json.dumps(summary, default=str),
            ),
        )
    _write(store, recs, "replay", run_id, rules.version, cik_of, now)
    return summary


def record_live(store: BreadthStore, day: date, now: datetime, t: S.Thresholds = S.DEFAULT) -> dict:
    """Record the groups on ``day`` (the last closed session) and fill every live outcome due.

    Same functions as the replay, over enough history for every window. A
    session already recorded is rewritten, not duplicated.
    """
    from advisor.breadth.ruleset import signal_rules

    rules = signal_rules(t)
    pending = store.conn.execute(
        "SELECT MIN(day) FROM breadth_records WHERE origin = 'live' AND day < ?",
        (day.isoformat(),),
    ).fetchone()[0]
    start = min(
        day - timedelta(days=LIVE_HISTORY_DAYS),
        date.fromisoformat(pending) - timedelta(days=LIVE_HISTORY_DAYS) if pending else day,
    )
    panel = load_panel(store, start=start, end=day)
    row = panel.row_of(day)
    if row is None or panel.sessions[row].date() != day:
        return {"ok": False, "error": f"no bars on file for {day}"}
    cik_of, eligible, events, ievents = _inputs(store, panel, t=t)
    recs = S.records(panel, eligible, events, row, row, t, ievents)
    out = Outcomes(panel, eligible, cik_of, sic_map(store))
    for r in recs:
        r["outcomes"] = out.of(r["symbol"], r["row"])
    with store.conn:
        # The day is rewritten whole: a record the current rules no longer
        # make must not survive from an earlier run of the same day.
        store.conn.execute(
            "DELETE FROM breadth_records WHERE origin = 'live' AND day = ?", (day.isoformat(),)
        )
    _write(store, recs, "live", "live", rules.version, cik_of, now)

    filled = 0
    for rid, symbol, d in store.conn.execute(
        "SELECT id, symbol, day FROM breadth_records WHERE origin = 'live' AND day < ?",
        (day.isoformat(),),
    ).fetchall():
        r0 = panel.row_of(date.fromisoformat(d))
        if r0 is None or symbol not in out.col:
            continue
        with store.conn:
            store.conn.execute(
                "UPDATE breadth_records SET outcomes_json = ?, updated_at = ? WHERE id = ?",
                (json.dumps(out.of(symbol, r0)), now.isoformat(), rid),
            )
        filled += 1
    return {
        "ok": True,
        "day": day.isoformat(),
        "rules": rules.version,
        "recorded": {g: sum(1 for r in recs if r["grp"] == g) for g in S.GROUPS},
        "outcomes_refreshed": filled,
    }


def load_records(store: BreadthStore, run_id: str) -> list[dict]:
    return [
        {
            "grp": g,
            "symbol": s,
            "day": date.fromisoformat(d),
            "outcomes": json.loads(o) if o else None,
        }
        for g, s, d, o in store.conn.execute(
            "SELECT grp, symbol, day, outcomes_json FROM breadth_records WHERE run_id = ?",
            (run_id,),
        )
    ]
