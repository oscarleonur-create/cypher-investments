"""Families P and F, and the records they produce (B1 of ``docs/breadth-plan.md``).

Two families, chosen because each has published evidence of *drift* over one
to six months — the opposite of the overreaction the user's own data refuted
(``swing-entry-plan.md``):

**P — price.** Any of three states, each read at the session's close:

- *momentum*: the 12-month return skipping the last month (close 21
  sessions ago over close 252 sessions ago) is in the top ``momentum_top`` of
  that day's eligible names (Jegadeesh–Titman);
- *high*: the close is within ``high_within`` of the 252-session high
  (George–Hwang);
- *breakout*: the day's return is at least ``breakout_sigmas`` of the name's
  own 60-session volatility, on ``breakout_rvol`` times its median volume,
  closing above its 50-session high.

**F — fundamentals.** A quarter's revenue, the day it is treated as known
(45 days after the quarter; 75 for a derived fiscal fourth), grew at least
``growth_min`` on the same quarter a year before, and that growth is at
least ``accel_min`` above the previous quarter's (revenue acceleration).
The "known" lag is conservative — most filers file sooner — so F arrives
late, never early.

**Records**, three groups, measured apart:

- ``P``: P turns on (off the session before), at most once per
  ``COOLDOWN_SESSIONS`` per name;
- ``F``: each qualifying quarter, on the session it becomes known;
- ``F+P``: the first session within ``CONVERGE_SESSIONS`` after an F event
  at which P was on at some point in the ``CONVERGE_SESSIONS`` before it —
  two independent families agreeing. This is the candidate the plan is about;
  P and F alone are recorded so it can be told whether agreement adds
  anything over either family.

A name must be eligible (E0) on the record's session. Share classes of one
company (GOOG, GOOGL) are one company: only the class that trades the most
dollars is read, so one piece of news is not two records.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from advisor.breadth.panel import Panel
from advisor.valuation.history import Point


@dataclass(frozen=True)
class Thresholds:
    momentum_top: float = 0.10  # P: top decile of 12-1 momentum among the day's eligible
    high_within: float = 0.02  # P: close within 2% of the 252-session high
    breakout_sigmas: float = 2.0  # P: a day's move of at least 2 own sigmas...
    breakout_rvol: float = 3.0  # P: ...on at least 3x its median volume
    growth_min: float = 0.10  # F: the quarter grew at least 10% year on year...
    accel_min: float = 0.05  # F: ...and 5 points faster than the quarter before
    min_quarter_revenue: float = 10e6  # F: below this a growth rate is a small-base artefact


DEFAULT = Thresholds()

MOMENTUM_LOOKBACK = 252
MOMENTUM_SKIP = 21
MIN_CROSS_SECTION = 50  # names ranked that day before a decile means anything
HIGH_WINDOW = 252
SIGMA_WINDOW = 60
VOLUME_WINDOW = 50
BASE_WINDOW = 50
CONVERGE_SESSIONS = 21  # about a month
COOLDOWN_SESSIONS = 60  # a name is recorded again in a group only after this many sessions

GROUPS = ("P", "F", "F+P")


def dedupe_share_classes(eligible: pd.DataFrame, panel: Panel, cik_of: dict[str, int]):
    """Keep one symbol per company: the class with the most dollars traded over the panel.

    Choosing by the whole panel looks ahead, but only to pick between classes
    of the same company, which move together; it cannot pick a winner.
    """
    by_cik: dict[int, list[str]] = {}
    for s in eligible.columns:
        cik = cik_of.get(s)
        if cik is not None:
            by_cik.setdefault(cik, []).append(s)
    out = eligible.copy()
    dollars = (panel.close * panel.volume).sum()
    for symbols in by_cik.values():
        if len(symbols) < 2:
            continue
        keep = max(symbols, key=lambda s: (float(dollars.get(s, 0.0)), s))
        for s in symbols:
            if s != keep:
                out[s] = False
    return out


def price_states(panel: Panel, eligible: pd.DataFrame, t: Thresholds = DEFAULT) -> dict:
    """Bool DataFrames per P state, each already restricted to eligible names."""
    close, high, volume = panel.close, panel.high, panel.volume
    ret = close.pct_change(fill_method=None)

    mom = close.shift(MOMENTUM_SKIP) / close.shift(MOMENTUM_LOOKBACK) - 1
    ranked = mom.where(eligible)
    rank = ranked.rank(axis=1, pct=True)
    # A rank among a handful of names is not a decile: with one name, that
    # name is "the top 10%" every day.
    enough = ranked.notna().sum(axis=1) >= MIN_CROSS_SECTION
    momentum = (rank > 1 - t.momentum_top) & enough.to_numpy()[:, None]

    top = high.rolling(HIGH_WINDOW, min_periods=HIGH_WINDOW).max()
    near_high = close >= (1 - t.high_within) * top

    sigma = ret.rolling(SIGMA_WINDOW, min_periods=SIGMA_WINDOW * 2 // 3).std().shift(1)
    usual = volume.rolling(VOLUME_WINDOW, min_periods=VOLUME_WINDOW * 4 // 5).median().shift(1)
    base = close.rolling(BASE_WINDOW, min_periods=BASE_WINDOW * 4 // 5).max().shift(1)
    breakout = (
        (sigma > 0)
        & (ret >= t.breakout_sigmas * sigma)
        & (usual > 0)
        & (volume >= t.breakout_rvol * usual)
        & (close > base)
    )
    states = {"momentum": momentum, "high": near_high, "breakout": breakout}
    return {k: (v.fillna(False).astype(bool) & eligible) for k, v in states.items()}


@dataclass(frozen=True)
class FEvent:
    symbol: str
    row: int  # the session it became known
    quarter: str
    revenue: float
    growth: float
    growth_before: float


def _minus(key: tuple[int, int], n: int) -> tuple[int, int]:
    y, q = key
    for _ in range(n):
        y, q = (y - 1, 4) if q == 1 else (y, q - 1)
    return y, q


def fundamental_events(
    quarters: dict[str, dict[tuple[int, int], Point]],
    sessions: pd.DatetimeIndex,
    t: Thresholds = DEFAULT,
) -> list[FEvent]:
    """Every qualifying quarter, placed on the first session at or after it became known.

    ``quarters`` maps a symbol to its quarterly revenue points. Pure.
    """
    out = []
    if len(sessions) == 0:
        return out
    for symbol, qs in quarters.items():
        for key in sorted(qs):
            need = [_minus(key, n) for n in (1, 4, 5)]
            if not all(k in qs for k in need):
                continue
            now, before, year_ago, year_ago_before = (
                qs[key],
                qs[need[0]],
                qs[need[1]],
                qs[need[2]],
            )
            if year_ago.value <= 0 or year_ago_before.value <= 0:
                continue
            if now.value < t.min_quarter_revenue:
                continue
            growth = now.value / year_ago.value - 1
            growth_before = before.value / year_ago_before.value - 1
            if growth < t.growth_min or growth - growth_before < t.accel_min:
                continue
            row = int(sessions.searchsorted(pd.Timestamp(now.known), side="left"))
            if row >= len(sessions):
                continue  # known after the last session on file
            out.append(
                FEvent(
                    symbol=symbol,
                    row=row,
                    quarter=f"{key[0]}Q{key[1]}",
                    revenue=now.value,
                    growth=growth,
                    growth_before=growth_before,
                )
            )
    return out


def _cooled(rows: list[int], cooldown: int = COOLDOWN_SESSIONS) -> list[int]:
    kept: list[int] = []
    for r in sorted(rows):
        if not kept or r - kept[-1] >= cooldown:
            kept.append(r)
    return kept


def records(
    panel: Panel,
    eligible: pd.DataFrame,
    events: list[FEvent],
    first_row: int,
    last_row: int,
    t: Thresholds = DEFAULT,
) -> list[dict]:
    """Records of each group on sessions ``first_row..last_row``. Pure.

    History before ``first_row`` is read — a cooldown or a convergence window
    reaches back — but only records on sessions inside the range are returned,
    so a live run over the last session and a replay over two years agree on
    every session they share.
    """
    states = price_states(panel, eligible, t)
    p_on = states["momentum"] | states["high"] | states["breakout"]
    p_recent = p_on.rolling(CONVERGE_SESSIONS, min_periods=1).max().astype(bool)
    sessions = panel.sessions
    elig = eligible.to_numpy()
    cols = {s: i for i, s in enumerate(eligible.columns)}
    out: list[dict] = []

    def emit(group: str, symbol: str, row: int, detail: dict) -> None:
        if first_row <= row <= last_row:
            out.append(
                {"grp": group, "symbol": symbol, "row": row,
                 "day": sessions[row].date(), "detail": detail}
            )  # fmt: skip

    def p_detail(symbol: str, row: int) -> dict:
        return {name: bool(states[name].iat[row, cols[symbol]]) for name in states}

    def p_window(symbol: str, row: int) -> dict:
        """Which P states were on at some session of the convergence window."""
        lo, j = max(0, row - CONVERGE_SESSIONS + 1), cols[symbol]
        return {f"{name}_in_window": bool(states[name].iloc[lo : row + 1, j].any())
                for name in states}  # fmt: skip

    p_arr = p_on.to_numpy()
    for symbol, j in cols.items():
        on = p_arr[:, j]
        starts = np.flatnonzero(on & ~np.concatenate(([False], on[:-1])))
        for row in _cooled(starts.tolist()):
            emit("P", symbol, row, p_detail(symbol, row))

    recent = p_recent.to_numpy()
    fp_rows: dict[str, list[tuple[int, dict]]] = {}
    for e in events:
        j = cols.get(e.symbol)
        if j is None:
            continue
        f_detail = {
            "quarter": e.quarter,
            "revenue": e.revenue,
            "growth": round(e.growth, 4),
            "growth_before": round(e.growth_before, 4),
        }
        if elig[e.row, j]:
            emit("F", e.symbol, e.row, f_detail)
        stop = min(e.row + CONVERGE_SESSIONS, len(sessions) - 1)
        for row in range(e.row, stop + 1):
            if elig[row, j] and recent[row, j]:
                fp_rows.setdefault(e.symbol, []).append(
                    (row, {**f_detail, "f_row_lag": row - e.row})
                )
                break
    for symbol, hits in fp_rows.items():
        kept = set(_cooled([r for r, _ in hits]))
        for row, detail in hits:
            if row in kept:
                kept.discard(row)
                emit("F+P", symbol, row, {**detail, **p_window(symbol, row)})
    return out
