"""What happened after each proposal, and whether the user took it.

Every proposal is scored the same way whatever its action — ENTER, IN_ZONE,
WAIT and NONE alike — because "the system said wait and it ran 8%" is as
much a lesson as "it said enter and the stop hit". Returns are from the
proposal's price, long.

    next_close   the next session's close
    d5/d10/d20   the close 5, 10, 20 sessions later
    d60/d120     the close 60 and 120 sessions later: the position leg is held
                 for months, and twenty sessions cannot say whether it paid
    mae20        the worst low within 20 sessions (maximum adverse excursion)
    mae120       the same within 120 sessions
    trade_stop   1.0 if the trade leg's stop was touched before its time
                 stop (the next session's close), else 0.0
    pos_stop20   1.0 if the position leg's stop was touched within 20
                 sessions, else 0.0
    pos_exit     the position leg as its own rules would have run it: out at
                 its stop, else at its trim target, else at the close
                 POSITION_CAP sessions later. With pos_exit_sessions (how
                 long it was held) and pos_exit_stop / pos_exit_target (1.0
                 for the way it left). Fixed horizons say whether the price
                 was good; this says what the position would have made.

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

KEYS = ("next_close", "d5", "d10", "d20", "mae20", "d60", "d120", "mae120")
# The session each key waits for. A proposal is only fetched for when one of
# its missing keys has become knowable: with d120 a proposal stays pending for
# about 170 days, and asking Yahoo daily for each one would be thousands of
# downloads that can add nothing.
KEY_SESSIONS = {
    "next_close": 1,
    "trade_stop": 1,
    "d5": 5,
    "d10": 10,
    "d20": 20,
    "mae20": 20,
    "pos_stop20": 20,
    "d60": 60,
    "d120": 120,
    "mae120": 120,
    # A position leg can leave any day: checked daily while it is open (one
    # download per symbol per run, the same as every other key).
    "pos_exit": 1,
    "pos_exit_sessions": 1,
    "pos_exit_stop": 1,
    "pos_exit_target": 1,
}
LONGEST = max(KEY_SESSIONS.values())
SETTLE = timedelta(minutes=15)


@dataclass(frozen=True)
class Bar:
    day: date
    high: float
    low: float
    close: float


def _closed(day: date, now: datetime) -> bool:
    return now >= datetime.combine(day, mc.session_close(day), tzinfo=mc.MARKET_TZ) + SETTLE


POSITION_CAP = 120  # sessions a position leg is followed before it is closed at market
POS_EXIT_KEYS = ("pos_exit", "pos_exit_sessions", "pos_exit_stop", "pos_exit_target")


def keys_for(p: Proposal) -> tuple[str, ...]:
    extra: tuple[str, ...] = ()
    for leg in p.legs:
        if leg.horizon == "trade":
            extra += ("trade_stop",)
        elif leg.horizon == "position":
            extra += ("pos_stop20", *POS_EXIT_KEYS)
    return KEYS + extra


def position_exit(p: Proposal, leg, bars: list[Bar], now: datetime) -> dict[str, float]:
    """The position leg run by its own rules; {} while it is still open.

    Each session after the entry, in order: the stop is checked before the
    target (a day that touched both is scored as stopped — the cautious
    reading, since daily bars cannot say which came first), then the target;
    at POSITION_CAP sessions it is closed at that day's close. A stop is
    filled at the stop price: daily bars carry no open, so a gap through it
    is not charged (an optimistic fill, said here).
    """
    cap = nth_trading_day(p.session, POSITION_CAP)
    held = 0
    for b in sorted(bars, key=lambda x: x.day):
        if b.day <= p.session or b.day > cap:
            continue
        held += 1
        if not _closed(b.day, now):
            return {}
        if b.low <= leg.stop:
            px, how = leg.stop, "stop"
        elif leg.target and b.high >= leg.target:
            px, how = leg.target, "target"
        elif b.day == cap:
            px, how = b.close, "time"
        else:
            continue
        return {
            "pos_exit": px / leg.entry - 1,
            "pos_exit_sessions": float(held),
            "pos_exit_stop": 1.0 if how == "stop" else 0.0,
            "pos_exit_target": 1.0 if how == "target" else 0.0,
        }
    return {}


def entered_at_close(p: Proposal) -> bool:
    """Built at or after its session's close: the entry is the closing price."""
    close = datetime.combine(p.session, mc.session_close(p.session), tzinfo=mc.MARKET_TZ)
    return mc.to_et(p.built_at) >= close


def score(p: Proposal, bars: list[Bar], now: datetime) -> dict[str, float | None]:
    """Every outcome knowable at ``now``. Pure."""
    out: dict[str, float | None] = {}
    if not p.price or p.price <= 0 or not bars:
        return out
    now = mc.to_et(now)
    by_day = {b.day: b for b in bars}

    def ret(px: float) -> float:
        return px / p.price - 1

    for key, n in (
        ("next_close", 1),
        ("d5", 5),
        ("d10", 10),
        ("d20", 20),
        ("d60", 60),
        ("d120", 120),
    ):
        target = nth_trading_day(p.session, n)
        if _closed(target, now) and target in by_day:
            out[key] = ret(by_day[target].close)

    d20 = nth_trading_day(p.session, 20)
    window = [b for b in bars if p.session < b.day <= d20]
    if _closed(d20, now) and window:
        out["mae20"] = ret(min(b.low for b in window))
    d120 = nth_trading_day(p.session, 120)
    long_window = [b for b in bars if p.session < b.day <= d120]
    if _closed(d120, now) and long_window:
        out["mae120"] = ret(min(b.low for b in long_window))

    nxt = nth_trading_day(p.session, 1)
    # A proposal made at or after the close enters at the close: that day's
    # low came before the entry and cannot touch its stop. (One made during
    # the session still counts its whole day, which can overstate touches:
    # daily bars cannot say whether the low came before or after it.)
    first = nxt if entered_at_close(p) else p.session
    for leg in p.legs:
        if leg.horizon == "trade" and _closed(nxt, now):
            span = [b for b in bars if first <= b.day <= nxt]
            if span:
                out["trade_stop"] = 1.0 if min(b.low for b in span) <= leg.stop else 0.0
        if leg.horizon == "position" and _closed(d20, now) and window:
            out["pos_stop20"] = 1.0 if min(b.low for b in window) <= leg.stop else 0.0
        if leg.horizon == "position":
            out.update(position_exit(p, leg, bars, now))
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


def due(p: Proposal, now: datetime) -> bool:
    """Whether a missing key of ``p`` has become knowable since it was last filled."""
    missing = [k for k in keys_for(p) if k not in p.outcomes]
    if not missing:
        return False
    first = min(KEY_SESSIONS[k] for k in missing)
    return _closed(nth_trading_day(p.session, first), now)


def fill_proposal_outcomes(
    store: EntryStore,
    now: datetime,
    *,
    bars_fn: Callable[[str, date, date], list[Bar]] = daily_bars,
    max_age_days: int = 200,  # d120 is ~170 calendar days out; room for holidays
) -> TrackResult:
    """Fill whatever outcomes have become knowable. One download per symbol per run."""
    now = mc.to_et(now)
    result = TrackResult()
    pending = [
        p
        for p in store.list(since=now.date() - timedelta(days=max_age_days))
        if now.date() > p.session and any(k not in p.outcomes for k in keys_for(p))
    ]
    result.pending = len(pending)
    by_symbol: dict[str, list[Proposal]] = {}
    for p in pending:
        if due(p, now):
            by_symbol.setdefault(p.symbol, []).append(p)
    for symbol, props in by_symbol.items():
        start = min(p.session for p in props)
        end = max(nth_trading_day(p.session, LONGEST) for p in props) + timedelta(days=1)
        bars = bars_fn(symbol, start, min(end, now.date() + timedelta(days=1)))
        if not bars:
            result.no_bars.append(symbol)
            continue
        for p in props:
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
