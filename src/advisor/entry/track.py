"""What happened after each proposal, and whether the user took it.

Every proposal is scored the same way whatever its action — ENTER, IN_ZONE,
WAIT and NONE alike — because "the system said wait and it ran 8%" is as
much a lesson as "it said enter and the stop hit". Returns are from the
proposal's price, long.

    next_close   the next session's close
    d5/d10/d20   the close 5, 10, 20 sessions later
    mae20        the worst low within 20 sessions (maximum adverse excursion)
    trade_stop   1.0 if the trade leg's stop was touched before its time
                 stop (the next session's close), else 0.0
    pos_stop20   1.0 if the position leg's stop was touched within 20
                 sessions, else 0.0

A key is written once it is knowable and never rewritten.
"""

from __future__ import annotations

import logging
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta

from advisor.daemon import market_calendar as mc
from advisor.entry.proposal import Proposal
from advisor.entry.store import EntryStore
from advisor.scanner.outcomes import nth_trading_day

logger = logging.getLogger(__name__)

KEYS = ("next_close", "d5", "d10", "d20", "mae20")
SETTLE = timedelta(minutes=15)


@dataclass(frozen=True)
class Bar:
    day: date
    high: float
    low: float
    close: float


def _closed(day: date, now: datetime) -> bool:
    return now >= datetime.combine(day, mc.session_close(day), tzinfo=mc.MARKET_TZ) + SETTLE


def keys_for(p: Proposal) -> tuple[str, ...]:
    extra = tuple({"trade": "trade_stop", "position": "pos_stop20"}[leg.horizon] for leg in p.legs)
    return KEYS + extra


def score(p: Proposal, bars: list[Bar], now: datetime) -> dict[str, float | None]:
    """Every outcome knowable at ``now``. Pure."""
    out: dict[str, float | None] = {}
    if not p.price or p.price <= 0 or not bars:
        return out
    now = mc.to_et(now)
    by_day = {b.day: b for b in bars}

    def ret(px: float) -> float:
        return px / p.price - 1

    for key, n in (("next_close", 1), ("d5", 5), ("d10", 10), ("d20", 20)):
        target = nth_trading_day(p.session, n)
        if _closed(target, now) and target in by_day:
            out[key] = ret(by_day[target].close)

    d20 = nth_trading_day(p.session, 20)
    window = [b for b in bars if p.session < b.day <= d20]
    if _closed(d20, now) and window:
        out["mae20"] = ret(min(b.low for b in window))

    nxt = nth_trading_day(p.session, 1)
    for leg in p.legs:
        if leg.horizon == "trade" and _closed(nxt, now):
            span = [b for b in bars if p.session <= b.day <= nxt]
            if span:
                out["trade_stop"] = 1.0 if min(b.low for b in span) <= leg.stop else 0.0
        if leg.horizon == "position" and _closed(d20, now) and window:
            out["pos_stop20"] = 1.0 if min(b.low for b in window) <= leg.stop else 0.0
    return out


def daily_bars(symbol: str, start: date, end: date) -> list[Bar]:
    """Daily highs, lows and closes; [] on failure or for rows Yahoo lost."""
    try:
        import yfinance as yf

        df = yf.download(
            symbol,
            start=start.isoformat(),
            end=end.isoformat(),
            interval="1d",
            progress=False,
            auto_adjust=False,
            multi_level_index=False,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("track: no bars for %s: %s", symbol, exc)
        return []
    if df is None or df.empty:
        return []
    out = []
    for ts, row in df.iterrows():
        h, lo, c = float(row["High"]), float(row["Low"]), float(row["Close"])
        if all(v == v and v > 0 for v in (h, lo, c)):
            out.append(Bar(ts.date(), h, lo, c))
    return out


@dataclass
class TrackResult:
    pending: int = 0
    updated: int = 0
    fields: int = 0
    no_bars: list[str] = field(default_factory=list)

    def summary(self) -> str:
        text = f"proposals: {self.pending} pending, {self.updated} updated ({self.fields} fields)"
        if self.no_bars:
            text += f"; no bars for {', '.join(self.no_bars[:5])}"
        return text


def fill_proposal_outcomes(
    store: EntryStore,
    now: datetime,
    *,
    bars_fn: Callable[[str, date, date], list[Bar]] = daily_bars,
    max_age_days: int = 60,
) -> TrackResult:
    now = mc.to_et(now)
    result = TrackResult()
    for p in store.list(since=now.date() - timedelta(days=max_age_days)):
        if all(k in p.outcomes for k in keys_for(p)):
            continue
        if now.date() <= p.session:
            continue
        result.pending += 1
        end = nth_trading_day(p.session, 20) + timedelta(days=1)
        bars = bars_fn(p.symbol, p.session, min(end, now.date() + timedelta(days=1)))
        if not bars:
            result.no_bars.append(p.symbol)
            continue
        added = store.merge_outcomes(p, score(p, bars, now))
        if added:
            result.updated += 1
            result.fields += added
    return result


def sync_proposal_fills(
    entry_store: EntryStore, scanner_store, sessions: list[date], *, fetch=None
) -> int:
    """Mark proposals taken from broker fills, in the scanner's journal. Idempotent."""
    from advisor.scanner.journal import fetch_fills, match
    from advisor.scanner.models import DecisionSource

    proposals = [p for day in sessions for p in entry_store.list(session=day)]
    if not proposals:
        return 0
    fills = (fetch or fetch_fills)(min(sessions), max(sessions))
    standing = scanner_store.latest_decisions()
    written = 0
    for d in match(proposals, fills):
        prior = standing.get(d.candidate_id)
        if prior is not None and prior.taken and prior.source is DecisionSource.BROKER:
            continue
        scanner_store.record_decision(d)
        written += 1
    return written
