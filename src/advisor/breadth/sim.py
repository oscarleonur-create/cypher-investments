"""Simulated picks: the ones you chose to follow, entered and exited by the system's own rules.

User, 2026-10-04: "cuando hablo de almacenar nuestros picks hablo de las
decisiones del sistema así yo no haya entrado… Dame la opción de add to sim,
así no tenemos un tracking demasiado amplio… el sistema lo toma como entrada
y dentro de sus análisis posteriores también tiene que darnos la salida."
The same day: the entry is the price when you add it, and the exit is the
plan already on main — its σ stop, else out at the close 20 sessions later.

Every day's picks are already stored (``breadth_picks``) and every pick
record is measured against its peers (``breadth_records``). A sim is a third,
smaller thing: one pick you chose, with a price you could have paid, and an
exit the system calls without you.

    entry     the live price when you add it (TastyTrade's mark, else its last)
    stop      entry × (1 − the pick plan's stop %): 2·σ·√10 kept in 8–25%
    window    the sessions after the one the entry belongs to, through the
              20th (``plan_replay.HOLD``)
    exit      the first session that opens at or below the stop: at its open;
              else the first whose low reaches it: at the stop; else the 20th
              session's close

Exits are read from the stored daily bars (``breadth_bars``), the same bars
the replays measured, after each night's sync. A session with no bar cannot
trigger anything. Nothing here can be edited or deleted once added: a record
you can prune is a record you can flatter.
"""

from __future__ import annotations

import json
import uuid
from datetime import date, datetime, timedelta

from advisor.breadth.plan_replay import HOLD
from advisor.daemon import market_calendar as mc

_SCHEMA = """\
CREATE TABLE IF NOT EXISTS breadth_sims (
    id          TEXT PRIMARY KEY,
    symbol      TEXT NOT NULL,
    pick_day    TEXT NOT NULL,        -- the picks list it was added from
    added_at    TEXT NOT NULL,
    session     TEXT NOT NULL,        -- the session the entry price belongs to
    entry       REAL NOT NULL,
    entry_source TEXT NOT NULL,
    stop        REAL,                 -- NULL: no volatility estimate, a time exit only
    stop_pct    REAL,
    exit_by     TEXT NOT NULL,        -- the 20th session after ``session``
    status      TEXT NOT NULL,        -- OPEN | STOP | TIME
    exit_day    TEXT,
    exit_price  REAL,
    ret         REAL,
    held        INTEGER,              -- sessions in the trade at exit
    pick_json   TEXT NOT NULL         -- the pick as it stood: families, verdict, plan
);
CREATE INDEX IF NOT EXISTS idx_breadth_sims_open ON breadth_sims(status, symbol);
"""


class SimError(ValueError):
    """A sim that cannot be added, with the reason."""


def ensure(conn) -> None:
    conn.executescript(_SCHEMA)


def nth_session_after(day: date, n: int) -> date:
    """The ``n``-th trading session strictly after ``day``. Pure."""
    cursor, seen = day, 0
    while seen < n:
        cursor += timedelta(days=1)
        if mc.is_trading_day(cursor):
            seen += 1
    return cursor


def entry_session(moment: datetime) -> date:
    """The session an entry price at ``moment`` belongs to. Pure.

    During a session, today's; before the open, on a weekend or a holiday, the
    last one that traded — its close is the price there is. The window starts
    the session after, so a low printed before the click never counts.
    """
    return mc.session_of(moment)


def new_sim(pick: dict, pick_day: str, price: float, source: str, now: datetime) -> dict:
    """A sim row for ``pick`` entered at ``price``. Pure. Raises SimError."""
    if price is None or not price > 0:
        raise SimError(f"no price to enter {pick.get('symbol')} at")
    plan = pick.get("plan") or {}
    pct = plan.get("stop_pct")
    stop = price * (1 - pct) if pct else None
    session = entry_session(now)
    snapshot = {
        "families": pick.get("families"),
        "price": pick.get("price"),
        "verdict": (pick.get("verdict") or {}).get("action"),
        "headline": (pick.get("verdict") or {}).get("headline"),
        "stage": plan.get("stage"),
        "group": plan.get("group"),
        "stop_basis": plan.get("stop_basis"),
        "provisional": bool(pick.get("provisional")),
    }
    return {
        "id": uuid.uuid4().hex[:12],
        "symbol": pick["symbol"].upper(),
        "pick_day": pick_day,
        "added_at": now.isoformat(),
        "session": session.isoformat(),
        "entry": float(price),
        "entry_source": source,
        "stop": stop,
        "stop_pct": pct,
        "exit_by": nth_session_after(session, HOLD).isoformat(),
        "status": "OPEN",
        "exit_day": None,
        "exit_price": None,
        "ret": None,
        "held": None,
        "pick_json": json.dumps(snapshot),
    }


def evaluate(sim: dict, bars: list[tuple[str, float | None, float | None, float]]) -> dict:
    """The exit, if the bars show one. Pure.

    ``bars``: (day, open, low, close), any order, any range; only the
    sessions after the entry session through ``exit_by`` are read. Returns
    the fields to update — empty while the trade is open.
    """
    start, end = sim["session"], sim["exit_by"]
    window = sorted(b for b in bars if start < b[0] <= end and b[3] is not None)
    stop, entry = sim.get("stop"), sim["entry"]
    for n, (day, open_, low, close) in enumerate(window, start=1):
        if stop is not None:
            o = open_ if open_ is not None else close
            lo = low if low is not None else close
            if o <= stop:
                return _closed("STOP", day, o, entry, n)
            if lo <= stop:
                return _closed("STOP", day, stop, entry, n)
    # Out at the 20th session's close; a session without a bar there closes on
    # the last one before it, once a later bar shows the window is over.
    past = any(b[0] > end for b in bars if b[3] is not None)
    if window and (window[-1][0] == end or past):
        day, *_, close = window[-1]
        return _closed("TIME", day, close, entry, len(window))
    return {}


def _closed(status: str, day: str, price: float, entry: float, held: int) -> dict:
    return {"status": status, "exit_day": day, "exit_price": float(price),
            "ret": float(price) / entry - 1, "held": held}  # fmt: skip


def _bars(conn, symbol: str, after: str) -> list[tuple]:
    return conn.execute(
        "SELECT day, open, low, close FROM breadth_bars WHERE symbol = ? AND day > ? ORDER BY day",
        (symbol, after),
    ).fetchall()


def update_open(conn) -> dict:
    """Close every open sim the stored bars have finished. Returns what closed."""
    ensure(conn)
    closed = []
    rows = conn.execute(
        "SELECT id, symbol, session, exit_by, entry, stop FROM breadth_sims WHERE status = 'OPEN'"
    ).fetchall()
    for sid, symbol, session, exit_by, entry, stop in rows:
        sim = {"session": session, "exit_by": exit_by, "entry": entry, "stop": stop}
        upd = evaluate(sim, _bars(conn, symbol, session))
        if upd:
            conn.execute(
                "UPDATE breadth_sims SET status = ?, exit_day = ?, exit_price = ?, ret = ?, "
                "held = ? WHERE id = ? AND status = 'OPEN'",
                (upd["status"], upd["exit_day"], upd["exit_price"], upd["ret"], upd["held"], sid),
            )
            closed.append(f"{symbol} {upd['status']} {upd['ret']:+.1%}")
    conn.commit()
    return {"open": len(rows) - len(closed), "closed": closed}


def add(conn, row: dict) -> dict:
    """Store a sim. One open sim per symbol: a second add is refused, not doubled."""
    ensure(conn)
    open_ = conn.execute(
        "SELECT id, added_at FROM breadth_sims WHERE symbol = ? AND status = 'OPEN'",
        (row["symbol"],),
    ).fetchone()
    if open_ is not None:
        raise SimError(f"{row['symbol']} is already in the sim since {open_[1][:16]}")
    cols = list(row)
    conn.execute(
        f"INSERT INTO breadth_sims ({', '.join(cols)}) VALUES ({', '.join('?' * len(cols))})",
        [row[c] for c in cols],
    )
    conn.commit()
    return row


def listing(conn) -> dict:
    """Open sims with their last close, closed ones, and the record so far."""
    ensure(conn)
    update_open(conn)
    cur = conn.execute("SELECT * FROM breadth_sims ORDER BY added_at DESC")
    cols = [d[0] for d in cur.description]
    rows = [dict(zip(cols, tuple(r), strict=True)) for r in cur.fetchall()]
    for r in rows:
        r["pick"] = json.loads(r.pop("pick_json") or "{}")
        if r["status"] == "OPEN":
            last = conn.execute(
                "SELECT day, close FROM breadth_bars WHERE symbol = ? AND day > ? "
                "ORDER BY day DESC LIMIT 1",
                (r["symbol"], r["session"]),
            ).fetchone()
            r["last_day"], r["last"] = (last[0], last[1]) if last else (None, None)
            r["unrealized"] = (r["last"] / r["entry"] - 1) if r["last"] else None
            r["sessions_in"] = conn.execute(
                "SELECT COUNT(*) FROM breadth_bars WHERE symbol = ? AND day > ? AND day <= ?",
                (r["symbol"], r["session"], r["exit_by"]),
            ).fetchone()[0]
    closed = [r for r in rows if r["status"] != "OPEN"]
    return {
        "open": [r for r in rows if r["status"] == "OPEN"],
        "closed": closed,
        "record": record(closed),
    }


def record(closed: list[dict]) -> dict:
    """What the closed sims made: count, mean, how many stopped, how long held. Pure."""
    rets = [r["ret"] for r in closed if r.get("ret") is not None]
    if not rets:
        return {"n": 0}
    return {
        "n": len(rets),
        "mean": sum(rets) / len(rets),
        "won": sum(1 for x in rets if x > 0),
        "stopped": sum(1 for r in closed if r["status"] == "STOP"),
        "held": sum(r.get("held") or 0 for r in closed) / len(closed),
    }


async def live_price(symbol: str) -> tuple[float | None, str]:
    """TastyTrade's mark for ``symbol``, else its last; (None, why) when there is none."""
    try:
        from tastytrade.market_data import get_market_data_by_type

        from advisor.api import deps

        session = await deps.get_tt_session()
        quotes = await get_market_data_by_type(session, equities=[symbol.upper()])
    except Exception as exc:  # noqa: BLE001
        return None, f"no live quote: {type(exc).__name__}: {exc}"
    for q in quotes:
        mark = getattr(q, "mark", None)
        if mark:
            return float(mark), "TastyTrade mark"
        last = getattr(q, "last", None)
        if last:
            return float(last), "TastyTrade last"
    return None, "no live quote: the broker returned none"
