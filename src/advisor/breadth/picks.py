"""Picks: the names where two or more families agree on the last close, with why.

User decision (2026-09-28): show top picks in the frontend with the rationale
of each pick. The measurement has not proven any family (``docs/breadth-plan.md``),
so a pick says exactly what was measured and no more:

- **What a pick is.** An E0-eligible name on which at least two of F, I and P
  are active on the session — the ``2+`` state the replay judged, not a new
  rule. F and I are active for ``CONVERGE_SESSIONS`` after their event; P
  when any of its states was on in that window.
- **The order is evidence, not a forecast.** More families first; then F
  present (the one family whose interval has cleared zero, in the three-year
  replay only); then the size of F's acceleration; then the most recent
  agreement. No score, no expected return, no price target, no fair value.
- **Every pick carries its group's track record** — the latest replay's cells
  and the live count — and the caveats, so the evidence behind the list
  travels with it.

Computed once a night by ``breadth_sync`` and stored; the API only reads.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta

import numpy as np
import pandas as pd

from advisor.breadth import signals as S
from advisor.breadth.panel import Panel, load_panel
from advisor.breadth.store import BreadthStore

logger = logging.getLogger(__name__)

TOP_N = 10
HISTORY_DAYS = 600  # a year for momentum and E0, plus the windows

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS breadth_picks (
    day          TEXT NOT NULL,
    rank         INTEGER NOT NULL,
    symbol       TEXT NOT NULL,
    rules        TEXT NOT NULL,
    payload_json TEXT NOT NULL,
    built_at     TEXT NOT NULL,
    PRIMARY KEY (day, symbol)
);
"""


@dataclass
class _Active:
    families: list[str]
    f: S.FEvent | None
    i: S.IEvent | None
    since_row: int


def order_key(p: dict) -> tuple:
    """How picks are ordered: evidence first, never a forecast. Pure.

    More families; F present; larger acceleration (points of YoY growth gained);
    the most recent agreement.
    """
    f = p.get("f") or {}
    accel = (f.get("growth", 0.0) - f.get("growth_before", 0.0)) if f else 0.0
    return (-len(p["families"]), "F" not in p["families"], -accel, -p["since_row"])


def _latest(events, row: int, window: int):
    """The newest of ``events`` active at ``row`` (within ``window`` sessions after it)."""
    best = None
    for e in events:
        if e.row <= row <= e.row + window and (best is None or e.row > best.row):
            best = e
    return best


def active_at(
    symbol: str,
    row: int,
    j: int,
    f_by: dict,
    i_by: dict,
    p_recent: np.ndarray,
    eligible: np.ndarray,
) -> _Active | None:
    """Which families agree on ``symbol`` at ``row``, and since when. Pure.

    ``since`` is the first session of the current run in which two or more
    families were active and the name eligible, looked for within the last
    three months.
    """
    C = S.CONVERGE_SESSIONS

    def fams(r: int):
        f = _latest(f_by.get(symbol, ()), r, C)
        i = _latest(i_by.get(symbol, ()), r, C)
        names = [n for n, on in (("F", f), ("I", i), ("P", p_recent[r, j])) if on]
        return names, f, i

    names, f, i = fams(row)
    if len(names) < 2 or not eligible[row, j]:
        return None
    since = row
    for r in range(row - 1, max(-1, row - 63), -1):
        if not eligible[r, j] or len(fams(r)[0]) < 2:
            break
        since = r
    return _Active(families=names, f=f, i=i, since_row=since)


def _pct(x: float | None, sign: bool = True) -> str:
    if x is None or x != x:  # NaN: a price the day's bars did not have
        return "n/a"
    return f"{x * 100:+.1f}%" if sign else f"{x * 100:.1f}%"


def _usd(x: float) -> str:
    for unit, div in (("bn", 1e9), ("M", 1e6), ("k", 1e3)):
        if abs(x) >= div:
            return f"${x / div:,.1f}{unit}"
    return f"${x:,.0f}"


def rationale(p: dict, eligible_count: int) -> dict:
    """The reasons for a pick, each with its source, and what would undo it. Pure.

    Only numbers already in the pick are used: nothing here estimates what the
    name is worth or where its price goes.
    """
    reasons: list[dict] = []
    undo: list[str] = []
    f, i, pp = p.get("f"), p.get("i"), p.get("p") or {}
    if f:
        end = f.get("quarter_end") or f["quarter"]
        reasons.append(
            {
                "family": "F",
                "text": (
                    f"Revenue for the quarter ended {end}: {_usd(f['revenue'])}, "
                    f"{_pct(f['growth'])} year on year, accelerating from "
                    f"{_pct(f['growth_before'])} the quarter before."
                ),
                "source": f"SEC XBRL revenue; public from {f['known']} (its 10-Q/10-K)",
            }
        )
        undo.append("the next quarter's year-on-year growth slows")
    if pp.get("momentum_on"):
        reasons.append(
            {
                "family": "P",
                "text": (
                    f"12-month return excluding the last month: {_pct(pp.get('momentum_12_1'))}, "
                    f"in the top decile of the {eligible_count:,} eligible names."
                ),
                "source": "daily closes (Yahoo), ranked the same session",
            }
        )
    if pp.get("high_on"):
        off = pp.get("from_52w_high")
        # P counts if a state was on at any session of the window, so the
        # price may have left the high since: say which (FEIM, 2026-09-28:
        # "within 2% ... (now -8.1%)" read as a contradiction).
        still = off is not None and off >= -0.02
        reasons.append(
            {
                "family": "P",
                "text": (
                    f"Within 2% of its 52-week high (now {_pct(off)})."
                    if still
                    else f"Came within 2% of its 52-week high in the last month; now "
                    f"{_pct(off)} below it."
                ),
                "source": "daily highs and closes (Yahoo)",
            }
        )
    if pp.get("breakout_on") and pp.get("breakout_day"):
        rvol = pp.get("breakout_rvol")
        reasons.append(
            {
                "family": "P",
                "text": (
                    f"Broke out on {pp['breakout_day']}: {_pct(pp.get('breakout_move'))} "
                    f"on {rvol:.1f}x its usual volume, above its 50-session high."
                    if rvol
                    else f"Broke out on {pp['breakout_day']} above its 50-session high."
                ),
                "source": "daily bars (Yahoo)",
            }
        )
    if pp:
        undo.append("price momentum fades: out of the top decile and more than 2% off its high")
    if i:
        reasons.append(
            {
                "family": "I",
                "text": (
                    f"{i['buyers']} officers or directors bought {_usd(i['value'])} of stock "
                    f"on the open market within 30 days (latest filing {i['filed']})."
                ),
                "source": "SEC Form 4 (code P, not under a 10b5-1 plan, not routine)",
            }
        )
        undo.append("the same insiders start selling")
    move = p.get("move_since")
    reasons.append(
        {
            "family": "",
            "text": (
                f"The families have agreed since {p['since']} ({p['sessions_since']} sessions); "
                f"the price has moved {_pct(move)} since then."
            ),
            "source": "breadth records",
        }
    )
    return {"reasons": reasons, "invalidates": undo}


def _replay_record(store: BreadthStore) -> list[dict]:
    """The latest replay of each window, longest first.

    Every window is shown, not the one that looks best: in the first runs the
    three-year window cleared zero at 20 sessions and the two-year did not.
    Only runs of the current signal rules count — a run under older rules
    judged different records.
    """
    from advisor.breadth.ruleset import signal_rules

    current = signal_rules().version
    latest: dict[int, dict] = {}
    for run_id, params, summary, rules in store.conn.execute(
        "SELECT id, params_json, summary_json, rules FROM breadth_replay_runs "
        "ORDER BY started_at DESC"
    ):
        if rules != current:
            continue
        years = int(json.loads(params).get("years", 0))
        if years in latest:
            continue
        s = json.loads(summary)
        latest[years] = {
            "years": years,
            "run_id": run_id,
            "from": s.get("from"),
            "to": s.get("to"),
            "cells": [c for c in s.get("cells", []) if c.get("group") in ("2+", "F+P", "F", "I")],
            "cells_tested": s.get("cells_tested"),
        }
    return [latest[y] for y in sorted(latest, reverse=True)]


def _held(db_path) -> set[str]:
    try:
        from advisor.daemon.store import DaemonStore

        book = DaemonStore(db_path).load_latest_book()
        return {p.underlying.upper() for p in book.positions} if book else set()
    except Exception as exc:  # noqa: BLE001
        logger.info("picks: no book: %s", exc)
        return set()


LiveFetch = Callable[[list[str], date], dict]  # symbols, day -> {symbol: Bar of that day}


def yahoo_today(symbols: list[str], day: date) -> dict:
    """``day``'s bar so far for each symbol (the session in progress). Missing names left out."""
    from advisor.breadth.bars import clean, yahoo_fetch

    out = {}
    for s, bars in yahoo_fetch(symbols, day).items():
        good, _ = clean([b for b in bars if b.day == day])
        if good:
            out[s] = good[-1]
    return out


def with_live_row(panel: Panel, day: date, bars: dict) -> Panel:
    """``panel`` with one more session, ``day``, holding the live bars. Pure.

    Names without a live bar are NaN on it, never yesterday's price: a pick
    must rest on today's price or not be made.
    """
    ts = pd.Timestamp(day)
    if ts in panel.close.index:
        return panel
    idx = panel.close.index.append(pd.DatetimeIndex([ts]))
    close, high, volume = (f.reindex(idx) for f in (panel.close, panel.high, panel.volume))
    for s, b in bars.items():
        if s in close.columns:
            close.loc[ts, s] = b.close
            high.loc[ts, s] = b.high if b.high is not None else b.close
            volume.loc[ts, s] = b.volume if b.volume is not None else np.nan
    return Panel(close, high, volume)


def build_picks(
    store: BreadthStore,
    day: date,
    now: datetime,
    db_path=None,
    n: int = TOP_N,
    *,
    live: LiveFetch | None = None,
):
    """Rank the names where families agree on ``day``; store and return the top ``n``.

    With ``live``, ``day`` is a session still in progress (or closed but not yet
    synced): the panel is the stored history plus one row of live bars, fetched
    only for names that *can* qualify — two families need F or I, so a name
    with neither active is never asked for. The result is marked provisional;
    F and I rest on filings through last night (EDGAR's indexes publish at the
    end of the day) and a breakout's volume is the session's so far.
    """
    from advisor.breadth.measure import CAVEATS, _inputs
    from advisor.breadth.ruleset import signal_rules

    store.conn.executescript(_SCHEMA)
    start = day - timedelta(days=HISTORY_DAYS)
    if live is None:
        panel = load_panel(store, start=start, end=day)
        row = panel.row_of(day)
        if row is None or panel.sessions[row].date() != day:
            return {"ok": False, "error": f"no bars on file for {day}"}
    else:
        base = load_panel(store, start=start, end=day - timedelta(days=1))
        if base.close.empty:
            return {"ok": False, "error": "no stored history"}
        _, _, ev0, iev0 = _inputs(store, base)
        reach = len(base.sessions) - S.CONVERGE_SESSIONS
        wanted = sorted({e.symbol for e in [*ev0, *iev0] if e.row >= reach})
        bars = live(wanted, day) if wanted else {}
        if not bars:
            return {"ok": False, "error": f"no live prices for {day}"}
        panel = with_live_row(base, day, bars)
        row = len(panel.sessions) - 1
    cik_of, eligible_df, events, ievents = _inputs(store, panel)
    states = S.price_states(panel, eligible_df)
    p_on = states["momentum"] | states["high"] | states["breakout"]
    p_recent = p_on.rolling(S.CONVERGE_SESSIONS, min_periods=1).max().astype(bool).to_numpy()
    eligible = eligible_df.to_numpy()
    cols = {s: k for k, s in enumerate(eligible_df.columns)}
    f_by: dict[str, list] = {}
    for e in events:
        f_by.setdefault(e.symbol, []).append(e)
    i_by: dict[str, list] = {}
    for e in ievents:
        i_by.setdefault(e.symbol, []).append(e)

    names = {
        r[0]: (r[1], r[2])
        for r in store.conn.execute(
            "SELECT u.symbol, u.name, c.sic_desc FROM breadth_universe u "
            "LEFT JOIN breadth_companies c ON c.cik = u.cik WHERE u.day = "
            "(SELECT MAX(day) FROM breadth_universe)"
        )
    }
    held = _held(db_path) if db_path else set()
    close = panel.close.to_numpy()
    high = panel.high.to_numpy()
    volume = panel.volume.to_numpy()
    sessions = panel.sessions

    picks = []
    for symbol, j in cols.items():
        if not np.isfinite(close[row, j]):
            continue  # no price on the session itself (a live bar that did not come back)
        a = active_at(symbol, row, j, f_by, i_by, p_recent, eligible)
        if a is None:
            continue
        lo = max(0, row - S.CONVERGE_SESSIONS + 1)
        c_now, c_since = close[row, j], close[a.since_row, j]
        ret20 = close[row, j] / close[row - 20, j] - 1 if row >= 20 else None
        p_detail = {}
        if "P" in a.families:
            mom = close[row - S.MOMENTUM_SKIP, j] / close[row - S.MOMENTUM_LOOKBACK, j] - 1
            top = np.nanmax(high[max(0, row - S.HIGH_WINDOW + 1) : row + 1, j])
            p_detail = {
                "momentum_on": bool(states["momentum"].iloc[lo : row + 1, j].any()),
                "momentum_12_1": float(mom) if np.isfinite(mom) else None,
                "high_on": bool(states["high"].iloc[lo : row + 1, j].any()),
                "from_52w_high": float(c_now / top - 1) if top > 0 else None,
                "breakout_on": bool(states["breakout"].iloc[lo : row + 1, j].any()),
            }
            hits = np.flatnonzero(states["breakout"].iloc[lo : row + 1, j].to_numpy())
            if len(hits):
                b = lo + int(hits[-1])
                usual = np.nanmedian(volume[max(0, b - S.VOLUME_WINDOW) : b, j])
                p_detail["breakout_day"] = sessions[b].date().isoformat()
                p_detail["breakout_move"] = float(close[b, j] / close[b - 1, j] - 1)
                p_detail["breakout_rvol"] = float(volume[b, j] / usual) if usual else None
        name, sector = names.get(symbol, (None, None))
        picks.append(
            {
                "symbol": symbol,
                "name": name,
                "sector": sector,
                "held": symbol in held,
                "families": a.families,
                "since": sessions[a.since_row].date().isoformat(),
                "since_row": a.since_row,
                "sessions_since": row - a.since_row,
                "price": float(c_now),
                "move_since": float(c_now / c_since - 1) if c_since > 0 else None,
                "return_20d": float(ret20) if ret20 is not None and np.isfinite(ret20) else None,
                "f": None
                if a.f is None
                else {
                    "quarter": a.f.quarter,
                    "quarter_end": a.f.end.isoformat() if a.f.end else None,
                    "revenue": a.f.revenue,
                    "growth": a.f.growth,
                    "growth_before": a.f.growth_before,
                    "known": sessions[a.f.row].date().isoformat(),
                },
                "i": None
                if a.i is None
                else {"buyers": a.i.buyers, "value": a.i.value, "filed": a.i.filed.isoformat()},
                "p": p_detail or None,
            }
        )
    picks.sort(key=order_key)
    top = picks[:n]
    n_eligible = int(eligible[row].sum())
    for p in top:
        p.update(rationale(p, n_eligible))
        p["provisional"] = live is not None
        p["asof"] = now.isoformat()
    rules = signal_rules().version
    track = _replay_record(store)
    live_records = {
        g: c
        for g, c in store.conn.execute(
            "SELECT grp, COUNT(*) FROM breadth_records WHERE origin = 'live' GROUP BY grp"
        )
    }
    with store.conn:
        store.conn.execute("DELETE FROM breadth_picks WHERE day = ?", (day.isoformat(),))
        store.conn.executemany(
            "INSERT INTO breadth_picks (day, rank, symbol, rules, payload_json, built_at) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            [
                (day.isoformat(), k + 1, p["symbol"], rules, json.dumps(p), now.isoformat())
                for k, p in enumerate(top)
            ],
        )
    return {
        "ok": True,
        "day": day.isoformat(),
        "built_at": now.isoformat(),
        "provisional": live is not None,
        "rules": rules,
        "candidates": len(picks),
        "picks": top,
        "track_record": track,
        "live_records": live_records,
        "caveats": list(CAVEATS),
    }


def refresh_picks(
    store: BreadthStore, now: datetime, db_path=None, *, live: LiveFetch = yahoo_today
):
    """Build the picks for the most recent session there is, final when its bars are stored.

    - Session in progress: provisional, on live prices.
    - Closed today but not yet synced (16:00 to the nightly sync): provisional too,
      on the day's final prices; the nightly build replaces it.
    - Otherwise (evening after the sync, weekend, before the open): final, from the store.
    """
    from advisor.breadth.bars import last_closed_session
    from advisor.daemon import market_calendar as mc

    et = mc.to_et(now)
    today = et.date()
    if mc.is_trading_day(today) and et.time() >= mc.REGULAR_OPEN:
        built = build_picks(store, today, now, db_path)  # stored bars: final
        if built.get("ok"):
            return built
        return build_picks(store, today, now, db_path, live=live)
    return build_picks(store, last_closed_session(now), now, db_path)


def _since_pick(store: BreadthStore, day: str, picks: list[dict]) -> None:
    """Each pick's move from its day's price to the latest stored close. In place."""
    for p in picks:
        r = store.conn.execute(
            "SELECT day, close FROM breadth_bars WHERE symbol = ? AND day > ? "
            "ORDER BY day DESC LIMIT 1",
            (p["symbol"], day),
        ).fetchone()
        if r and p.get("price"):
            p["since_pick"] = {"day": r[0], "close": r[1], "move": r[1] / p["price"] - 1}


def latest_picks(store: BreadthStore, day: str | None = None, history: int = 30) -> dict:
    """The picks of ``day`` (default: the newest), the days on file, and the record.

    A past day's picks carry how far each has moved since (``since_pick``):
    the history is how the list is checked, not a second list to act on.
    """
    from advisor.breadth.measure import CAVEATS

    store.conn.executescript(_SCHEMA)
    days = [
        {"day": d, "count": c, "provisional": bool(json.loads(p).get("provisional"))}
        for d, c, p in store.conn.execute(
            "SELECT day, COUNT(*), (SELECT payload_json FROM breadth_picks b2 "
            "WHERE b2.day = b.day ORDER BY rank LIMIT 1) FROM breadth_picks b "
            "GROUP BY day ORDER BY day DESC LIMIT ?",
            (history,),
        )
    ]
    if day is None:
        day = days[0]["day"] if days else None
    live = {
        g: c
        for g, c in store.conn.execute(
            "SELECT grp, COUNT(*) FROM breadth_records WHERE origin = 'live' GROUP BY grp"
        )
    }
    base = {"days": days, "track_record": _replay_record(store), "live_records": live,
            "caveats": list(CAVEATS)}  # fmt: skip
    if not day:
        return {"day": None, "picks": [], **base}
    rows = store.conn.execute(
        "SELECT rank, payload_json, rules, built_at FROM breadth_picks WHERE day = ? ORDER BY rank",
        (day,),
    ).fetchall()
    picks = [{"rank": r[0], **json.loads(r[1])} for r in rows]
    _since_pick(store, day, picks)
    return {
        "day": day,
        "built_at": rows[0][3] if rows else None,
        "rules": rows[0][2] if rows else None,
        "provisional": bool(picks and picks[0].get("provisional")),
        "picks": picks,
        **base,
    }
