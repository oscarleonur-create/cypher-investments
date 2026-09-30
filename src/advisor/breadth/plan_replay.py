"""The pick's entry plan, replayed: what the plan as shown would have done.

The breadth replay (``measure``) judges the *signal*: buy at the close of the
first session two families agree, hold exactly 20 sessions. A pick is traded
differently (user decision, 2026-09-30: *"los picks son diarios, el backtest
tiene que tener una salida clara, +5% -2.5% del entry price"*):

- **Target** ``TAKE_PROFIT`` above the entry, **stop** ``STOP_LOSS`` below it,
  both fixed from the entry price;
- **time exit** at the close ``HOLD`` sessions later when neither is reached.

Each F+P and 2+ record is replayed that way, entered at ``OFFSETS`` sessions
after the first day **only if the name was still a pick that session** (two
families still agreeing, still eligible — ``picks.active_at``, the code that
builds the list). On daily bars:

- a session that opens beyond either level fills at the open (a gap);
- a session whose range reaches both is counted as the **stop**: daily bars
  cannot say which came first, so the replay takes the worse (and counts it);
- otherwise the level reached fills at its price.

Each trade is compared with the same matched peers the signal replay uses,
over the same sessions it was held, and judged with the same block bootstrap
and verdict. The signal's own 20-session hold is kept beside it (``plain``),
so what the exits change is visible.

Not seen: the signal replay's list (``measure.CAVEATS``), fills inside a
session's range at exactly the level (no slippage), and costs.
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

# User decision, 2026-09-30: a pick is a trade with a clear exit.
TAKE_PROFIT = 0.05
STOP_LOSS = 0.025
HOLD = 20  # time exit when neither level is reached: the plan's review date
OFFSETS = (0, 5, 10, 15)  # sessions after the first day of agreement
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


def levels(entry: float) -> tuple[float, float]:
    """The target and the stop for an entry price. Pure."""
    return entry * (1 + TAKE_PROFIT), entry * (1 - STOP_LOSS)


def trade(
    close: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    open_: np.ndarray,
    e: int,
    hold: int = HOLD,
) -> dict | None:
    """One pick traded from ``close[e]`` with the bracket, and held plain beside it. Pure.

    ``close``, ``high``, ``low``, ``open_`` are one name's series. A session
    with no bar is skipped (it cannot reach a level); a missing high, low or
    open falls back to the close. Returns None when the entry has no price or
    the ``hold`` window is not complete.
    """
    entry = close[e]
    if not np.isfinite(entry) or entry <= 0 or e + hold >= len(close):
        return None
    window = range(e + 1, e + hold + 1)
    closes = [close[i] for i in window if np.isfinite(close[i])]
    if not closes:
        return None
    target, stop = levels(entry)
    plain = closes[-1] / entry - 1

    def done(price: float, how: str, n: int, both: bool = False) -> dict:
        return {"ret": price / entry - 1, "exit": how, "held": n, "both": both, "plain": plain}

    last_n = hold
    for n, i in enumerate(window, start=1):
        c = close[i]
        if not np.isfinite(c):
            continue
        last_n = n
        o = open_[i] if np.isfinite(open_[i]) else c
        hi = high[i] if np.isfinite(high[i]) else c
        lo = low[i] if np.isfinite(low[i]) else c
        if o <= stop:
            return done(o, "stop", n)
        if o >= target:
            return done(o, "target", n)
        hit_stop, hit_target = lo <= stop, hi >= target
        if hit_stop:
            return done(stop, "stop", n, both=hit_target)  # both: the worse is assumed
        if hit_target:
            return done(target, "target", n)
    return done(closes[-1], "time", last_n)


def _bars(store: BreadthStore, panel: Panel, symbols: list[str]):
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


def _judge(values):
    from advisor.learning.evaluate import MIN_SESSIONS, blocks_of, cluster_ci, verdict

    windows = len(blocks_of(values, HOLD))
    ci = cluster_ci(values, block=HOLD) if windows >= MIN_SESSIONS else None
    v, why = verdict(len(values), windows, ci)
    return {
        "mean": statistics.fmean(x for _, x in values),
        "ci": list(ci) if ci else None,
        "windows": windows,
        "verdict": v.value,
        "reason": why,
    }


def summarize(rows: list[dict]) -> list[dict]:
    """One cell per (group, offset): the bracket's results, beyond peers, and the plain hold."""
    cells = []
    for grp in GROUPS:
        for k in OFFSETS:
            have = [r for r in rows if r["grp"] == grp and r["offset"] == k]
            cell: dict = {"group": grp, "offset": k, "n": len(have)}
            if not have:
                cells.append({**cell, "verdict": "UNDETERMINED", "reason": "no trades"})
                continue
            n = len(have)
            exits = {h: sum(r["exit"] == h for r in have) / n for h in ("target", "stop", "time")}
            cell.update(
                {
                    "exits": exits,
                    "both": sum(r["both"] for r in have) / n,
                    "held_median": statistics.median(r["held"] for r in have),
                    "ret": statistics.fmean(r["ret"] for r in have),
                    "peers": statistics.fmean(r["peers"] for r in have),
                    # Per trade, not per day: the bracket's expectancy.
                    "raw": _judge([(r["day"], r["ret"]) for r in have]),
                    "excess": _judge([(r["day"], r["ret"] - r["peers"]) for r in have]),
                    "plain": _judge([(r["day"], r["plain"] - r["peers_hold"]) for r in have]),
                    "plain_ret": statistics.fmean(r["plain"] for r in have),
                    "time_ret": (
                        statistics.fmean(r["ret"] for r in have if r["exit"] == "time")
                        if exits["time"]
                        else None
                    ),
                }
            )
            cell["verdict"] = cell["excess"]["verdict"]
            cells.append(cell)
    return cells


def replay_plan(store: BreadthStore, now: datetime, *, years: int = 3) -> dict:
    """Replay a pick's bracket over the last ``years``; store and return the cells."""
    from advisor.breadth.measure import Outcomes, _inputs, sic_map
    from advisor.breadth.picks import active_at
    from advisor.breadth.ruleset import plan_rules

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
    high = panel.high.to_numpy()
    ahead = panel.close.ffill(limit=5).to_numpy()  # a peer's close on the exit session
    symbols = sorted({r["symbol"] for r in recs})
    low, open_ = _bars(store, panel, symbols)

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
            t = trade(close[:, j], high[:, j], low[r["symbol"]], open_[r["symbol"]], e)
            if t is None:
                skipped["incomplete"] += 1
                continue
            peers, _ = out.controls(r["symbol"], e)
            if not peers:
                skipped["no_peers"] += 1
                continue
            base = close[e, peers]
            same = ahead[e + t["held"], peers] / base - 1  # over the sessions it was held
            full = ahead[e + HOLD, peers] / base - 1
            same, full = same[np.isfinite(same)], full[np.isfinite(full)]
            if len(same) == 0 or len(full) == 0:
                skipped["no_peers"] += 1
                continue
            rows.append({"grp": r["grp"], "symbol": r["symbol"], "offset": k,
                         "day": panel.sessions[e].date(), "peers": float(same.mean()),
                         "peers_hold": float(full.mean()), **t})  # fmt: skip

    run_id = f"{now:%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:6]}"
    rules = plan_rules().version
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
        "take_profit": TAKE_PROFIT,
        "stop_loss": STOP_LOSS,
        "hold": HOLD,
        "cells": summarize(rows),
    }
    with store.conn:
        store.conn.execute(
            "INSERT INTO breadth_plan_runs VALUES (?, ?, ?, ?, ?)",
            (run_id, now.isoformat(), rules,
             json.dumps({"years": years, "offsets": list(OFFSETS), "hold": HOLD,
                         "take_profit": TAKE_PROFIT, "stop_loss": STOP_LOSS}),
             json.dumps(summary, default=str)),
        )  # fmt: skip
    return summary


def latest_plan_runs(store: BreadthStore) -> list[dict]:
    """The latest plan replay of each window under the current rules, longest first.

    A run under other exits (the σ stop of 2026-09-29) or other signal rules
    judged a different plan and is not shown.
    """
    from advisor.breadth.ruleset import plan_rules

    store.conn.executescript(_SCHEMA)
    current = plan_rules().version
    latest: dict[int, dict] = {}
    for summary, rules in store.conn.execute(
        "SELECT summary_json, rules FROM breadth_plan_runs ORDER BY started_at DESC"
    ):
        if rules != current:
            continue
        s = json.loads(summary)
        latest.setdefault(int(s["years"]), s)
    return [latest[y] for y in sorted(latest, reverse=True)]
