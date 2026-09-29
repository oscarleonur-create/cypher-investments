"""The pick's entry plan, replayed: what the plan as shown would have done.

The breadth replay (``measure``) judges the *signal*: buy at the close of the
first session two families agree, hold exactly 20 sessions, no stop. The
entry plan on a pick (``picks.entry_plan``) adds two things that replay never
tested (user, 2026-09-29: *"si esto no estaba, ¿cómo hacíamos backtesting?"*):

- **a stop** — the entry engine's position stop, 2·σ·√10 of the name's own
  daily moves kept in 8–25%, which the learning sweep only tested on the
  engine's in-zone entries, never on momentum picks;
- **a later entry** — most picks are shown sessions after the first day
  (9 of the 10 on 2026-09-28), and entering later was not measured.

So each F+P and 2+ record is replayed as the plan would trade it, entered at
``OFFSETS`` sessions after the first day **only if the name was still a pick
that session** (two families still agreeing, still eligible — ``picks.active_at``,
the code that builds the list), held ``HOLD`` sessions:

- with the stop: out at the stop when a session's low reaches it, or at the
  open when it gaps below (daily bars cannot say more);
- without it: the close ``HOLD`` sessions later, as the signal replay does.

Both are compared with the same matched peers the signal replay uses, held
over the same sessions without a stop, and judged with the same block
bootstrap and verdict. The difference between the two is the stop's effect.

What it cannot see is the signal replay's own list (``measure.CAVEATS``), and:
a stop filled at the stop price on a session that traded through it (a fill
at the open is taken only for a gap), and no costs.
"""

from __future__ import annotations

import json
import logging
import statistics
import uuid
from datetime import datetime, timedelta

import numpy as np

from advisor.breadth import signals as S
from advisor.breadth.panel import Panel, load_panel
from advisor.breadth.store import BreadthStore

logger = logging.getLogger(__name__)

OFFSETS = (0, 5, 10, 15)  # sessions after the first day of agreement
HOLD = 20  # the plan's horizon: its review date
GROUPS = ("F+P", "2+")  # the groups a pick's plan quotes (``picks.plan_group``)

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS breadth_plan_runs (
    id           TEXT PRIMARY KEY,
    started_at   TEXT NOT NULL,
    rules        TEXT NOT NULL,
    params_json  TEXT NOT NULL,
    summary_json TEXT NOT NULL
);
"""


def trade(
    close: np.ndarray, low: np.ndarray, open_: np.ndarray, e: int, stop: float | None, hold: int
) -> dict | None:
    """One plan trade entered at ``close[e]``: with and without its stop. Pure.

    ``close``, ``low``, ``open_`` are one name's series. A session with no bar
    is skipped (it cannot trigger a stop); the exit without a stop is the last
    close on or before session ``e + hold``, as long as one exists after entry.
    Returns None when the window is not complete or the entry has no price.
    """
    entry = close[e]
    if not np.isfinite(entry) or entry <= 0 or e + hold >= len(close):
        return None
    window = range(e + 1, e + hold + 1)
    closes = [close[i] for i in window if np.isfinite(close[i])]
    if not closes:
        return None
    plain = closes[-1] / entry - 1
    out = {"plain": plain, "stopped": False, "held": hold, "with_stop": plain}
    if stop is None:
        return out
    for n, i in enumerate(window, start=1):
        o = open_[i] if np.isfinite(open_[i]) else close[i]
        lo = low[i] if np.isfinite(low[i]) else close[i]
        if not np.isfinite(lo):
            continue
        if np.isfinite(o) and o <= stop:
            return {**out, "with_stop": o / entry - 1, "stopped": True, "held": n}
        if lo <= stop:
            return {**out, "with_stop": stop / entry - 1, "stopped": True, "held": n}
    return out


def _lows_opens(store: BreadthStore, panel: Panel, symbols: list[str]):
    """Low and open arrays aligned to the panel, for ``symbols`` only (NaN where absent)."""
    idx = {d: i for i, d in enumerate(panel.sessions.strftime("%Y-%m-%d"))}
    low, open_ = {}, {}
    for s in symbols:
        lo = np.full(len(idx), np.nan)
        op = np.full(len(idx), np.nan)
        for day, o, lw in store.conn.execute(
            "SELECT day, open, low FROM breadth_bars WHERE symbol = ?", (s,)
        ):
            i = idx.get(day)
            if i is not None:
                op[i] = o if o is not None else np.nan
                lo[i] = lw if lw is not None else np.nan
        low[s], open_[s] = lo, op
    return low, open_


def summarize(rows: list[dict]) -> list[dict]:
    """One cell per (group, offset): the plan with and without its stop, beyond peers."""
    from advisor.learning.evaluate import MIN_SESSIONS, blocks_of, cluster_ci, verdict

    cells = []
    for grp in GROUPS:
        for k in OFFSETS:
            have = [r for r in rows if r["grp"] == grp and r["offset"] == k]
            cell: dict = {"group": grp, "offset": k, "n": len(have)}
            if not have:
                cells.append({**cell, "verdict": "UNDETERMINED", "reason": "no trades"})
                continue
            for name in ("with_stop", "plain"):
                values = [(r["day"], r[name] - r["peers"]) for r in have]
                windows = len(blocks_of(values, HOLD))
                ci = cluster_ci(values, block=HOLD) if windows >= MIN_SESSIONS else None
                v, why = verdict(len(values), windows, ci)
                cell[name] = {
                    "mean": statistics.fmean(r[name] for r in have),
                    "excess": statistics.fmean(x for _, x in values),
                    "beat": sum(x > 0 for _, x in values) / len(values),
                    "ci": list(ci) if ci else None,
                    "windows": windows,
                    "verdict": v.value,
                    "reason": why,
                }
            cell["peers"] = statistics.fmean(r["peers"] for r in have)
            cell["stopped"] = sum(r["stopped"] for r in have) / len(have)
            cell["stop_pct"] = statistics.fmean(r["stop_pct"] for r in have)
            cell["stop_effect"] = cell["with_stop"]["mean"] - cell["plain"]["mean"]
            cells.append(cell)
    return cells


def replay_plan(store: BreadthStore, now: datetime, *, years: int = 3) -> dict:
    """Replay the entry plan over the last ``years``; store and return the cells."""
    from advisor.breadth.measure import Outcomes, _inputs, sic_map
    from advisor.breadth.picks import active_at, sigma_before
    from advisor.breadth.ruleset import signal_rules
    from advisor.entry.proposal import position_stop_pct

    store.conn.executescript(_SCHEMA)
    panel = load_panel(store)
    if panel.close.empty:
        return {"ok": False, "error": "no bars on file: run `advisor breadth sync` first"}
    cik_of, eligible_df, events, ievents = _inputs(store, panel)
    last = len(panel.sessions) - 1
    first = panel.row_of(panel.sessions[-1].date() - timedelta(days=365 * years))
    if first is None or first < S.MOMENTUM_LOOKBACK:
        first = min(S.MOMENTUM_LOOKBACK, last)
    recs = [
        r
        for r in S.records(panel, eligible_df, events, first, last, S.DEFAULT, ievents)
        if r["grp"] in GROUPS
    ]
    states = S.price_states(panel, eligible_df)
    p_on = states["momentum"] | states["high"] | states["breakout"]
    p_recent = p_on.rolling(S.CONVERGE_SESSIONS, min_periods=1).max().astype(bool).to_numpy()
    eligible = eligible_df.to_numpy()
    cols = {s: k for k, s in enumerate(eligible_df.columns)}
    f_by: dict[str, list] = {}
    for ev in events:
        f_by.setdefault(ev.symbol, []).append(ev)
    i_by: dict[str, list] = {}
    for ev in ievents:
        i_by.setdefault(ev.symbol, []).append(ev)
    out = Outcomes(panel, eligible_df, cik_of, sic_map(store))
    close = panel.close.to_numpy()
    fwd = out.fwd["d20"]
    symbols = sorted({r["symbol"] for r in recs})
    low, open_ = _lows_opens(store, panel, symbols)

    rows, skipped = [], {"not_a_pick": 0, "incomplete": 0, "no_peers": 0}
    for r in recs:
        j = cols[r["symbol"]]
        for k in OFFSETS:
            e = r["row"] + k
            if e > last:
                skipped["incomplete"] += 1
                continue
            if k and active_at(r["symbol"], e, j, f_by, i_by, p_recent, eligible) is None:
                skipped["not_a_pick"] += 1  # the list would not have shown it that session
                continue
            sigma = sigma_before(close[: e + 1, j])
            stop_pct = position_stop_pct(sigma)
            stop = close[e, j] * (1 - stop_pct) if stop_pct is not None else None
            t = trade(close[:, j], low[r["symbol"]], open_[r["symbol"]], e, stop, HOLD)
            if t is None or stop is None:
                skipped["incomplete"] += 1
                continue
            peers, _ = out.controls(r["symbol"], e)
            pr = fwd[e, peers] if peers else np.array([])
            pr = pr[np.isfinite(pr)]
            if len(pr) == 0:
                skipped["no_peers"] += 1
                continue
            rows.append({"grp": r["grp"], "symbol": r["symbol"], "offset": k,
                         "day": panel.sessions[e].date(), "peers": float(pr.mean()),
                         "stop_pct": stop_pct, **t})  # fmt: skip

    run_id = f"{now:%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    rules = signal_rules().version
    summary = {
        "ok": True,
        "run_id": run_id,
        "rules": rules,
        "years": years,
        "from": panel.sessions[first].date().isoformat(),
        "to": panel.sessions[last].date().isoformat(),
        "records": len(recs),
        "trades": len(rows),
        "skipped": skipped,
        "hold": HOLD,
        "cells": summarize(rows),
    }
    with store.conn:
        store.conn.execute(
            "INSERT INTO breadth_plan_runs VALUES (?, ?, ?, ?, ?)",
            (run_id, now.isoformat(), rules,
             json.dumps({"years": years, "offsets": list(OFFSETS), "hold": HOLD}),
             json.dumps(summary, default=str)),
        )  # fmt: skip
    return summary


def latest_plan_runs(store: BreadthStore) -> list[dict]:
    """The latest plan replay of each window under the current rules, longest first."""
    from advisor.breadth.ruleset import signal_rules

    store.conn.executescript(_SCHEMA)
    current = signal_rules().version
    latest: dict[int, dict] = {}
    for summary, rules in store.conn.execute(
        "SELECT summary_json, rules FROM breadth_plan_runs ORDER BY started_at DESC"
    ):
        if rules != current:
            continue
        s = json.loads(summary)
        latest.setdefault(int(s["years"]), s)
    return [latest[y] for y in sorted(latest, reverse=True)]
