"""E0: the universe the breadth layer searches, decided each day from what was known that day.

A listing is **eligible** when it is common stock of an SEC filer
(``listings``) and, on the day, it trades like something a position can be
built in and a signal can be computed on:

- the last close at or above ``MIN_PRICE``;
- the median ``ADV_WINDOW``-session dollar volume at or above
  ``MIN_DOLLAR_VOLUME`` — a median, so one block trade does not qualify a
  dead name;
- at least ``MIN_SESSIONS`` sessions of history, the year every later family
  needs (52-week high, 12-month momentum, a volatility to scale moves by).

Every threshold is inclusive. ``eligibility`` reads only bars at or before
the day it is asked about, so the same function decides the universe in a
replay without seeing the future. What a replay cannot recover is the
listing itself: the directory is today's, so names delisted since are
missing — survivorship the replay report must state.

Each day's snapshot keeps every common-stock listing with its reason, not
only the eligible, so a name's exit from the universe is visible.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass
from datetime import date, timedelta

from advisor.breadth.listings import Directory
from advisor.breadth.store import BreadthStore

MIN_PRICE = 5.0
MIN_DOLLAR_VOLUME = 10e6
MIN_SESSIONS = 250
ADV_WINDOW = 20
# A last bar older than this is not "the day's" price: the sync fell behind
# (a refused pull) or the name stopped trading. Either way, not judged on it.
STALE_BAR_DAYS = 7


@dataclass(frozen=True)
class Eligibility:
    eligible: bool
    reason: str | None
    price: float | None
    dollar_volume: float | None
    sessions: int


def eligibility(bars: list[tuple[date, float, float | None]], day: date) -> Eligibility:
    """Decide one name on ``day`` from ``(day, close, volume)`` rows. Pure.

    Rows after ``day`` are ignored. A session with no volume counts as zero
    dollars traded: the name was listed and nobody traded it.
    """
    rows = sorted((r for r in bars if r[0] <= day), key=lambda r: r[0])
    sessions = len(rows)
    if not rows:
        return Eligibility(False, "no bars", None, None, 0)
    if rows[-1][0] < day - timedelta(days=STALE_BAR_DAYS):
        return Eligibility(False, "stale bars", rows[-1][1], None, sessions)
    price = rows[-1][1]
    window = rows[-ADV_WINDOW:]
    dollar_volume = statistics.median(c * (v or 0.0) for _, c, v in window)
    if price is None or price < MIN_PRICE:
        return Eligibility(False, "price below floor", price, dollar_volume, sessions)
    if dollar_volume < MIN_DOLLAR_VOLUME:
        return Eligibility(False, "dollar volume below floor", price, dollar_volume, sessions)
    if sessions < MIN_SESSIONS:
        return Eligibility(False, "short history", price, dollar_volume, sessions)
    return Eligibility(True, None, price, dollar_volume, sessions)


def eligibility_panel(close, volume):
    """``eligibility`` for every (session, symbol) of a panel at once: a bool DataFrame.

    ``close`` and ``volume`` share an index of sessions and a column per
    symbol; NaN where a name has no bar. Each name is measured on its own bars
    — the dollar-volume window is its last ``ADV_WINDOW`` bars, as in
    ``eligibility`` — and then carried to every session of the panel, so a
    session a name did not trade is judged on its last bar and turns stale
    after ``STALE_BAR_DAYS``. A test holds the two functions to the same answer.
    """
    import pandas as pd

    out = {}
    index = pd.DatetimeIndex(close.index)
    for symbol in close.columns:
        c = close[symbol].dropna()
        if c.empty:
            out[symbol] = pd.Series(False, index=index)
            continue
        v = volume[symbol].reindex(c.index).fillna(0.0).clip(lower=0.0)
        own = pd.DataFrame(
            {
                "price": c,
                "dv": (c * v).rolling(ADV_WINDOW, min_periods=1).median(),
                "sessions": range(1, len(c) + 1),
                "last": pd.DatetimeIndex(c.index),
            },
            index=c.index,
        )
        own = own.reindex(index, method="ffill")
        fresh = (index - pd.DatetimeIndex(own["last"])) <= pd.Timedelta(days=STALE_BAR_DAYS)
        ok = (
            own["last"].notna()
            & fresh
            & (own["price"] >= MIN_PRICE)
            & (own["dv"] >= MIN_DOLLAR_VOLUME)
            & (own["sessions"] >= MIN_SESSIONS)
        )
        out[symbol] = ok.fillna(False).astype(bool)
    return pd.DataFrame(out, index=close.index)


def _bars(store: BreadthStore, symbol: str, day: date) -> list[tuple[date, float, float | None]]:
    # Only whether there are MIN_SESSIONS matters, so the count stored is capped there.
    return [
        (date.fromisoformat(d), c, v)
        for d, c, v in store.conn.execute(
            "SELECT day, close, volume FROM breadth_bars WHERE symbol = ? AND day <= ? "
            "ORDER BY day DESC LIMIT ?",
            (symbol, day.isoformat(), MIN_SESSIONS),
        )
    ]


def snapshot(store: BreadthStore, directory: Directory, day: date, rules: str) -> list[dict]:
    """Every listing's standing on ``day``; written to the store and returned."""
    rows = []
    for x in directory.listings:
        key = x.yahoo.upper()
        reason = directory.reasons.get(key)
        base = {
            "symbol": key,
            "cik": directory.ciks.get(key),
            "exchange": x.exchange,
            "name": x.name,
            "rules": rules,
        }
        if reason is not None:
            rows.append(
                {**base, "eligible": False, "reason": reason, "price": None,
                 "dollar_volume": None, "sessions": None}
            )  # fmt: skip
            continue
        e = eligibility(_bars(store, key, day), day)
        rows.append(
            {**base, "eligible": e.eligible, "reason": e.reason, "price": e.price,
             "dollar_volume": e.dollar_volume, "sessions": e.sessions}
        )  # fmt: skip
    store.write_universe(day, rows)
    return rows


def funnel(rows: list[dict]) -> dict:
    """Counts at each cut, and the reasons, for a snapshot's rows. Pure."""
    reasons: dict[str, int] = {}
    for r in rows:
        if not r["eligible"]:
            reasons[r["reason"] or "unknown"] = reasons.get(r["reason"] or "unknown", 0) + 1
    structural = {
        "no bars",
        "stale bars",
        "price below floor",
        "dollar volume below floor",
        "short history",
    }
    common = sum(1 for r in rows if r["eligible"] or r["reason"] in structural)
    return {
        "listings": len(rows),
        "common_stock_sec_filers": common,
        "eligible": sum(1 for r in rows if r["eligible"]),
        "excluded": dict(sorted(reasons.items(), key=lambda kv: -kv[1])),
    }
