"""The book's risk as one number, and the bets it is really made of.

Every entry is sized on its own: 2–4% of net liq at risk to its stop. Since
the user removed the 20% per-name limit (2026-10-04) that per-entry budget is
the only bound on size, and it counts each name as if it moved alone. They do
not. On 2026-10-07, over 120 sessions, COHR, AAOI, CRDO and DRAM correlated
0.50–0.76 with each other: 21.5% of net liq that carried 43.5% of the book's
risk, and moved as one bet.

So this measures the book the way it moves:

- **The book's daily volatility** from the covariance of its holdings, and
  the loss of a two-sigma move held two weeks — the same 2·σ·√10 a position
  stop is drawn at, so the two read on one scale.
- **Each name's share of that risk** (weight × its covariance with the book,
  over the book's variance). SPCX was 22% of net liq and 42% of the risk.
- **Groups**: names whose returns correlate at least ``GROUP_CORRELATION`` on
  average (average linkage), found from the returns, never from a sector label.
- **A hedge per group, measured**: the ETF that best explains the group's
  daily moves, its beta in dollars, and how much of the moves it explains.
  A hedge that explains less than half is not named: it would trade one risk
  for another. Named, never staged (the project's hedging rule).

Nothing here makes a call. What the book should carry is the user's to set
(exposure limits: the agent proposes, the user approves).
"""

from __future__ import annotations

import logging
import math
import sqlite3
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

WINDOW = 120  # sessions of daily returns: about six months
MIN_SESSIONS = 40  # fewer, and a name's covariance is noise; it is left out and named
GROUP_CORRELATION = 0.5  # average pairwise correlation that makes names one bet
HEDGE_ETFS = ("SMH", "QQQ", "SPY")  # liquid, optionable; the best fit is named
HEDGE_MIN_R2 = 0.5  # below this the hedge explains less of the group than it leaves
# The two-week loss is read on the position stop's own scale: 2·σ·√10.
HORIZON_SIGMAS = 2.0
HORIZON_SESSIONS = 10
# A reading older than this is not today's book (the daemon was down): not used.
MAX_AGE_DAYS = 4


class NameRisk(BaseModel):
    symbol: str
    weight: float  # of net liq
    vol: float  # daily volatility of its returns
    risk_share: float  # of the book's variance; can be negative for a diversifier
    sessions: int


class Hedge(BaseModel):
    etf: str
    beta: float  # dollars of the group's move per dollar of ETF move, over net liq
    r2: float
    notional: float  # dollars of the ETF that offset the group's beta
    shares: int
    price: float


class Group(BaseModel):
    members: list[str]
    weight: float
    risk_share: float
    correlation: float  # average pairwise
    two_week: float  # dollars: a 2σ two-week move of the group alone
    hedge: Hedge | None = None
    hedge_note: str = ""

    @property
    def label(self) -> str:
        return ", ".join(self.members)


class BookRisk(BaseModel):
    asof: datetime
    book_asof: datetime | None = None
    net_liq: float
    invested: float  # long equity notional over net liq
    sessions: int  # rows of returns the covariance was read over
    daily_vol: float  # of net liq
    two_week: float  # dollars
    names: list[NameRisk] = Field(default_factory=list)
    groups: list[Group] = Field(default_factory=list)
    excluded: list[str] = Field(default_factory=list)  # "SYM: why", left out of the risk

    @property
    def two_week_pct(self) -> float:
        return self.two_week / self.net_liq if self.net_liq else 0.0


def _horizon(vol: float) -> float:
    return HORIZON_SIGMAS * vol * math.sqrt(HORIZON_SESSIONS)


def _groups(corr: pd.DataFrame) -> list[list[str]]:
    """Average-linkage groups at ``GROUP_CORRELATION``; only those of two or more. Pure."""
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    symbols = list(corr.index)
    if len(symbols) < 2:
        return []
    dist = (1.0 - corr.fillna(0.0)).clip(lower=0.0).to_numpy(copy=True)
    np.fill_diagonal(dist, 0.0)
    dist = (dist + dist.T) / 2
    labels = fcluster(
        linkage(squareform(dist, checks=False), "average"),
        t=1.0 - GROUP_CORRELATION,
        criterion="distance",
    )
    by: dict[int, list[str]] = {}
    for sym, lab in zip(symbols, labels, strict=True):
        by.setdefault(int(lab), []).append(sym)
    return [sorted(m) for m in by.values() if len(m) >= 2]


def _hedge(
    group: pd.Series, etfs: pd.DataFrame, prices: dict[str, float], net_liq: float
) -> tuple[Hedge | None, str]:
    """The ETF that best explains ``group`` (daily return in net-liq units). Pure."""
    best: Hedge | None = None
    for etf in etfs.columns:
        x = etfs[etf]
        both = group.notna() & x.notna()
        if both.sum() < MIN_SESSIONS or not prices.get(etf):
            continue
        g, e = group[both], x[both]
        var = float(e.var())
        if var <= 0:
            continue
        beta = float(np.cov(g, e)[0, 1] / var)
        r2 = float(np.corrcoef(g, e)[0, 1] ** 2)
        notional = beta * net_liq
        h = Hedge(
            etf=etf,
            beta=beta,
            r2=r2,
            notional=notional,
            shares=round(notional / prices[etf]),
            price=prices[etf],
        )
        if best is None or h.r2 > best.r2:
            best = h
    if best is None:
        return None, "no hedge ETF has enough shared history to measure"
    if best.r2 < HEDGE_MIN_R2:
        return None, (
            f"the best fit, {best.etf}, explains only {best.r2:.0%} of its daily moves: "
            "no hedge fits"
        )
    if best.shares <= 0:
        return None, f"{best.etf}'s beta rounds to no shares"
    return best, ""


def measure(
    weights: dict[str, float],
    returns: pd.DataFrame,
    etf_returns: pd.DataFrame,
    etf_prices: dict[str, float],
    net_liq: float,
    asof: datetime,
    book_asof: datetime | None = None,
) -> BookRisk | None:
    """The book's risk from its weights and daily returns. Pure.

    ``weights``: long equity notional over net liq per symbol. ``returns``:
    daily returns, one column per symbol, oldest first; only the last
    ``WINDOW`` rows are read. None when nothing can be measured (no net liq,
    no holding with enough history).
    """
    if not net_liq or net_liq <= 0 or not weights:
        return None
    rows = returns.tail(WINDOW)
    excluded, usable = [], []
    for sym in sorted(weights):
        n = int(rows[sym].notna().sum()) if sym in rows else 0
        if n < MIN_SESSIONS:
            excluded.append(f"{sym}: {n} sessions of prices, under {MIN_SESSIONS}")
        elif weights[sym] > 0:
            usable.append(sym)
    if not usable:
        return None
    r = rows[usable]
    w = pd.Series({s: weights[s] for s in usable})
    cov = r.cov(min_periods=MIN_SESSIONS).fillna(0.0)
    variance = float(w @ cov @ w)
    if variance <= 0:
        return None
    vol = math.sqrt(variance)
    marginal = cov @ w
    names = sorted(
        (
            NameRisk(
                symbol=s,
                weight=float(w[s]),
                vol=float(r[s].std()),
                risk_share=float(w[s] * marginal[s] / variance),
                sessions=int(r[s].notna().sum()),
            )
            for s in usable
        ),
        key=lambda n: -n.risk_share,
    )
    share = {n.symbol: n.risk_share for n in names}
    corr = r.corr(min_periods=MIN_SESSIONS)
    etfs = etf_returns.reindex(r.index)
    groups = []
    for members in _groups(corr):
        gw = w[members]
        # Days every member traded: a missing day read as "no move" would
        # understate the group (SPCX had 80 sessions against 121).
        g = (r[members].dropna() @ gw).reindex(r.index)
        pairs = [corr.loc[a, b] for i, a in enumerate(members) for b in members[i + 1 :]]
        hedge, note = _hedge(g, etfs, etf_prices, net_liq)
        groups.append(
            Group(
                members=members,
                weight=float(gw.sum()),
                risk_share=float(sum(share[m] for m in members)),
                correlation=float(np.nanmean(pairs)),
                two_week=_horizon(float(g.std())) * net_liq,
                hedge=hedge,
                hedge_note=note,
            )
        )
    groups.sort(key=lambda g: -g.risk_share)
    return BookRisk(
        asof=asof,
        book_asof=book_asof,
        net_liq=net_liq,
        invested=float(sum(weights.values())),
        sessions=len(r),
        daily_vol=vol,
        two_week=_horizon(vol) * net_liq,
        names=names,
        groups=groups,
        excluded=excluded,
    )


# ── Live ─────────────────────────────────────────────────────────────────


def returns_from(closes: dict[str, list[tuple[date, float]]]) -> pd.DataFrame:
    """Daily returns, one column per symbol, on the union of their sessions. Pure."""
    series = {
        sym: pd.Series({d: px for d, px in rows}, dtype=float).sort_index()
        for sym, rows in closes.items()
        if rows
    }
    if not series:
        return pd.DataFrame()
    frame = pd.DataFrame(series).sort_index()
    return frame.pct_change(fill_method=None).iloc[1:]


def measure_book(book, now: datetime, *, closes=None) -> BookRisk | None:
    """The book as it was last saved, priced from daily closes (yfinance)."""
    from advisor.entry.actionables import holdings

    if book is None or not book.net_liq or book.net_liq <= 0:
        return None
    if closes is None:
        from advisor.entry.sheet import daily_closes as closes
    held = holdings(book)
    weights = {s: h.notional / book.net_liq for s, h in held.items() if h.notional > 0}
    raw = {s: closes(s) for s in [*weights, *HEDGE_ETFS]}
    frame = returns_from(raw)
    if frame.empty:
        return None
    etfs = [e for e in HEDGE_ETFS if e in frame]
    prices = {e: raw[e][-1][1] for e in etfs if raw[e]}
    held_cols = [s for s in weights if s in frame]
    return measure(
        weights,
        frame.reindex(columns=held_cols),
        frame[etfs],
        prices,
        book.net_liq,
        now,
        book_asof=book.as_of,
    )


# ── Store ────────────────────────────────────────────────────────────────

_SCHEMA = """
CREATE TABLE IF NOT EXISTS book_risk (
    day  TEXT NOT NULL PRIMARY KEY,
    body TEXT NOT NULL
)
"""


def save_risk(db_path: Path, risk: BookRisk) -> None:
    """One row per day; a re-run the same day replaces it."""
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(_SCHEMA)
        conn.execute(
            "INSERT OR REPLACE INTO book_risk VALUES (?, ?)",
            (risk.asof.date().isoformat(), risk.model_dump_json()),
        )
        conn.commit()
    finally:
        conn.close()


def latest_risk(db_path: Path, today: date | None = None) -> BookRisk | None:
    """The newest stored reading, if measured within ``MAX_AGE_DAYS`` of ``today``. No network."""
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute("SELECT body FROM book_risk ORDER BY day DESC LIMIT 1").fetchone()
    except sqlite3.OperationalError:
        return None
    finally:
        conn.close()
    if row is None:
        return None
    risk = BookRisk.model_validate_json(row[0])
    if today is not None and (today - risk.asof.date()).days > MAX_AGE_DAYS:
        return None
    return risk
