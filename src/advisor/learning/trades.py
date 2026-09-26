"""The user's trades as round trips, from the broker, and what the rules would have done.

The scanner's journal knows only openings: a fill marks a candidate taken.
Learning needs the whole trade — where the user got out, what it made, how
long it was held — because "the setup works" and "my trades on it work" are
different claims, and the gap between them is where the user's own exits
live.

A trade is an **episode** per (account, instrument): from flat, through any
adds and partial exits, back to flat. Scaling in and out is one trade, not
several. An episode still open at the end of the history is OPEN and has no
outcome yet.

Books are kept apart (the user's two strategies are never analysed together):

    short         closed on the entry session or the next one
    long          held more than five sessions
    unclassified  anything between, and every multi-leg option trade — the
                  user's rule does not say which book a three-day hold is,
                  and guessing would contaminate both

Every price comes from the broker. Nothing is inferred about a trade the
broker did not report.
"""

from __future__ import annotations

import asyncio
import logging
import re
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import StrEnum

from pydantic import BaseModel, Field

from advisor.daemon import market_calendar as mc

logger = logging.getLogger(__name__)

OPEN_ACTIONS = frozenset({"Buy to Open", "Sell to Open"})
CLOSE_ACTIONS = frozenset({"Sell to Close", "Buy to Close"})
SHORT_MAX_SESSIONS = 1  # closed on the entry session or the next
LONG_MIN_SESSIONS = 6  # more than five sessions: over a trading week
OPTION_MULTIPLIER = 100.0
# Rebuilt from here every time, so a long hold's entry is never cut off.
HISTORY_START = date(2024, 1, 1)


class Book(StrEnum):
    SHORT = "short"
    LONG = "long"
    UNCLASSIFIED = "unclassified"


@dataclass(frozen=True)
class Execution:
    account: str
    underlying: str
    symbol: str  # the instrument: ticker, or the OCC symbol for an option
    instrument: str  # "Equity" or "Equity Option"
    action: str
    quantity: float
    price: float
    executed_at: datetime  # tz-aware
    tx_id: str | None = None  # the broker's transaction id, when known

    @property
    def opens(self) -> bool:
        return self.action in OPEN_ACTIONS

    @property
    def signed(self) -> float:
        """Position change: + for a buy, - for a sell."""
        return self.quantity if self.action.startswith("Buy") else -self.quantity


class Trade(BaseModel):
    account: str
    underlying: str
    symbol: str
    instrument: str
    direction: str  # "long" | "short"
    opened_at: datetime
    closed_at: datetime | None = None
    entry_session: date
    exit_session: date | None = None
    quantity: float  # largest size held
    entry_price: float  # size-weighted over the opening fills
    exit_price: float | None = None  # size-weighted over the closing fills
    pnl: float | None = None  # dollars, before fees
    ret: float | None = None  # on the entry price, in the trade's direction
    sessions_held: int | None = None  # trading sessions from entry to exit
    book: Book = Book.UNCLASSIFIED
    multi_leg: bool = False  # an option leg opened with others on the same underlying
    executions: int = 0
    # What the system said about it, and what its rule would have done.
    candidate_ids: list[str] = Field(default_factory=list)
    proposal_id: str | None = None
    rule_exit: str | None = None  # how the rule would have left: "stop" | "time"
    rule_exit_price: float | None = None
    rule_ret: float | None = None
    notes: list[str] = Field(default_factory=list)

    @property
    def id(self) -> str:
        return f"{self.account}:{self.symbol}:{self.opened_at.isoformat()}"

    @property
    def closed(self) -> bool:
        return self.closed_at is not None

    @property
    def matched(self) -> bool:
        return bool(self.candidate_ids or self.proposal_id)


def sessions_between(a: date, b: date) -> int:
    """Trading sessions from ``a`` to ``b``: 0 the same day, 1 the next session."""
    if b <= a:
        return 0
    n, day = 0, a
    while day < b:
        day = _next_session(day)
        n += 1
    return n


def _next_session(day: date) -> date:
    from advisor.scanner.outcomes import next_trading_day

    return next_trading_day(day)


def classify(t: Trade) -> Book:
    if t.multi_leg or t.sessions_held is None:
        return Book.UNCLASSIFIED
    if t.sessions_held <= SHORT_MAX_SESSIONS:
        return Book.SHORT
    if t.sessions_held >= LONG_MIN_SESSIONS:
        return Book.LONG
    return Book.UNCLASSIFIED


def _vwap(fills: list[Execution]) -> float:
    qty = sum(f.quantity for f in fills)
    return sum(f.price * f.quantity for f in fills) / qty if qty else 0.0


_OCC_EXPIRY = re.compile(r"(\d{6})[CP]\d{8}$")


def option_expiry(symbol: str) -> date | None:
    """The expiry in an OCC symbol ("SPY   260619C00500000" -> 2026-06-19)."""
    m = _OCC_EXPIRY.search(symbol.replace(" ", ""))
    if not m:
        return None
    try:
        return datetime.strptime(m.group(1), "%y%m%d").date()
    except ValueError:
        return None


def round_trips(executions: list[Execution], today: date | None = None) -> list[Trade]:
    """Episodes per (account, instrument), flat to flat. Pure given ``today``.

    A closing fill with no open position before it (history that starts
    mid-position) is dropped: that trade's entry is outside the window and
    cannot be priced. An option still open after its expiry is left OPEN with
    a note: an expiry or assignment arrives as a non-trade transaction this
    does not read, and its price is not guessed.
    """
    by_key: dict[tuple[str, str], list[Execution]] = defaultdict(list)
    seen: set[str] = set()
    for e in executions:
        if e.tx_id is not None:
            if e.tx_id in seen:
                continue  # the same fill reported twice must not double a trade
            seen.add(e.tx_id)
        if e.quantity > 0 and e.price >= 0:
            by_key[(e.account, e.symbol)].append(e)

    trades: list[Trade] = []
    for (_account, _symbol), fills in by_key.items():
        fills.sort(key=lambda f: f.executed_at)
        position = 0.0
        episode: list[Execution] = []
        for f in fills:
            if position == 0 and not f.opens:
                continue  # entered before the window: unpriceable
            episode.append(f)
            position += f.signed
            if abs(position) < 1e-9:
                trades.append(_trade(episode, closed=True))
                episode, position = [], 0.0
        if episode:
            trades.append(_trade(episode, closed=False))

    _flag_multi_leg(trades)
    for t in trades:
        t.book = classify(t)
        expiry = option_expiry(t.symbol) if t.instrument == "Equity Option" else None
        if not t.closed and today is not None and expiry is not None and expiry < today:
            t.notes.append(f"open past its {expiry} expiry: expired or assigned, not read")
    return sorted(trades, key=lambda t: t.opened_at)


def _trade(fills: list[Execution], *, closed: bool) -> Trade:
    first = fills[0]
    direction = "long" if first.action.startswith("Buy") else "short"
    opens = [f for f in fills if f.opens]
    closes = [f for f in fills if not f.opens]
    running, peak = 0.0, 0.0
    for f in fills:
        running += f.signed
        peak = max(peak, abs(running))
    mult = OPTION_MULTIPLIER if first.instrument == "Equity Option" else 1.0
    entry = _vwap(opens)
    t = Trade(
        account=first.account,
        underlying=first.underlying.upper(),
        symbol=first.symbol,
        instrument=first.instrument,
        direction=direction,
        opened_at=first.executed_at,
        entry_session=mc.to_et(first.executed_at).date(),
        quantity=peak,
        entry_price=entry,
        executions=len(fills),
    )
    if closed and closes:
        last = fills[-1]
        exit_ = _vwap(closes)
        sign = 1.0 if direction == "long" else -1.0
        bought = sum(f.price * f.quantity for f in fills if f.action.startswith("Buy"))
        sold = sum(f.price * f.quantity for f in fills if f.action.startswith("Sell"))
        t.closed_at = last.executed_at
        t.exit_session = mc.to_et(last.executed_at).date()
        t.exit_price = exit_
        t.pnl = (sold - bought) * mult
        t.ret = sign * (exit_ / entry - 1) if entry > 0 else None
        t.sessions_held = sessions_between(t.entry_session, t.exit_session)
    return t


def _flag_multi_leg(trades: list[Trade]) -> None:
    """Option legs on one underlying opened within a minute of each other."""
    options = [t for t in trades if t.instrument == "Equity Option"]
    for t in options:
        for o in options:
            if (
                o is not t
                and o.account == t.account
                and o.underlying == t.underlying
                and abs((o.opened_at - t.opened_at).total_seconds()) <= 60
            ):
                t.multi_leg = True
                t.notes.append("option leg opened with others: a spread, not scored per leg")
                break


# ── Matching to what the system said ─────────────────────────────────────


def match(trades: list[Trade], candidates: list, proposals: list) -> None:
    """Attach scanner candidates and the day's proposal to each long trade. In place.

    A candidate matches on underlying and entry session. The proposal is the
    last one written for that symbol on the entry session (a WAIT at 09:45 and
    an ENTER at 11:45 are both kept; the trade was taken against the latest).
    """
    by_cand: dict[tuple[str, date], list[str]] = defaultdict(list)
    for c in candidates:
        by_cand[(c.symbol.upper(), c.session)].append(c.id)
    by_prop: dict[tuple[str, date], object] = {}
    for p in sorted(proposals, key=lambda p: p.built_at):
        by_prop[(p.symbol.upper(), p.session)] = p
    for t in trades:
        if t.direction != "long":
            continue
        key = (t.underlying, t.entry_session)
        t.candidate_ids = sorted(by_cand.get(key, []))
        p = by_prop.get(key)
        t.proposal_id = p.id if p is not None else None


def rule_exit(t: Trade, proposal, bars: list) -> None:
    """What the trade leg's own rule would have done with this entry. In place.

    Only for a closed, long, equity trade matched to a proposal with a trade
    leg: the leg's stop, else out at the next session's close. The rule is
    applied to the user's own entry price, so the comparison is exits only.
    ``bars`` are daily (day, high, low, close).
    """
    if not t.closed or t.direction != "long" or t.instrument != "Equity" or proposal is None:
        return
    leg = next((g for g in getattr(proposal, "legs", []) if g.horizon == "trade"), None)
    if leg is None:
        return
    stop_pct = 1 - leg.stop / leg.entry if leg.entry else None
    if stop_pct is None or stop_pct <= 0:
        return
    stop = t.entry_price * (1 - stop_pct)
    nxt = _next_session(t.entry_session)
    span = [b for b in bars if t.entry_session <= b.day <= nxt]
    if not span or span[-1].day != nxt:
        t.notes.append("rule exit not knowable: bars through the next session missing")
        return
    if min(b.low for b in span) <= stop:
        t.rule_exit, t.rule_exit_price = "stop", stop
    else:
        t.rule_exit, t.rule_exit_price = "time", span[-1].close
    t.rule_ret = t.rule_exit_price / t.entry_price - 1


# ── Broker ────────────────────────────────────────────────────────────────


def execution_from_transaction(tx, account: str) -> Execution | None:
    from advisor.scanner.journal import _text

    action = _text(getattr(tx, "action", None))
    if _text(getattr(tx, "transaction_type", None)) != "Trade":
        return None
    if action not in OPEN_ACTIONS | CLOSE_ACTIONS:
        return None
    return Execution(
        account=account,
        underlying=str(tx.underlying_symbol or tx.symbol or ""),
        symbol=str(tx.symbol or ""),
        instrument=_text(getattr(tx, "instrument_type", None)),
        action=action,
        quantity=float(tx.quantity or 0),
        price=float(tx.price or 0),
        executed_at=tx.executed_at,
        tx_id=str(tx.id) if getattr(tx, "id", None) is not None else None,
    )


def fetch_executions(start: date, end: date) -> tuple[list[Execution], str | None]:
    """Every opening and closing trade in every account. ([], error) on failure."""

    async def _get():
        from tastytrade import Account

        from advisor.market.tastytrade_client import get_session

        session = await get_session()
        out = []
        for account in await Account.get(session):
            txs = await account.get_history(
                session, start_date=start, end_date=end, page_offset=None
            )
            for tx in txs:
                e = execution_from_transaction(tx, account.account_number)
                if e is not None:
                    out.append(e)
        return out

    try:
        return asyncio.run(_get()), None
    except Exception as exc:  # noqa: BLE001
        logger.warning("trades: broker history unavailable: %s", exc)
        return [], str(exc)


@dataclass
class SyncResult:
    executions: int = 0
    trades: int = 0
    closed: int = 0
    written: int = 0
    matched: int = 0
    error: str | None = None

    def summary(self) -> str:
        if self.error:
            return f"broker unavailable: {self.error}"
        return (
            f"{self.executions} executions → {self.trades} trades ({self.closed} closed); "
            f"{self.written} written; {self.matched} matched to a candidate or proposal"
        )


def sync_trades(
    trade_store,
    scanner_store,
    entry_store,
    start: date,
    end: date,
    *,
    fetch: Callable[[date, date], tuple[list[Execution], str | None]] = fetch_executions,
    bars_fn: Callable | None = None,
) -> SyncResult:
    """Rebuild trades from the broker for [start, end] and store them. Idempotent.

    A trade is recomputed whole each time, so an OPEN trade becomes CLOSED
    when its last exit arrives; the store replaces a row only when it changed.
    """
    executions, error = fetch(start, end)
    if error:
        return SyncResult(error=error)
    trades = round_trips(executions, today=mc.now_et().date())
    candidates = scanner_store.list(since=start, limit=100000) if scanner_store else []
    proposals = entry_store.list(since=start) if entry_store else []
    match(trades, candidates, proposals)
    if bars_fn is None:
        from advisor.entry.track import daily_bars as bars_fn
    props = {p.id: p for p in proposals}
    for t in trades:
        p = props.get(t.proposal_id) if t.proposal_id else None
        if p is not None and t.closed:
            nxt = _next_session(t.entry_session)
            rule_exit(t, p, bars_fn(t.underlying, t.entry_session, nxt + timedelta(days=1)))
    result = SyncResult(
        executions=len(executions),
        trades=len(trades),
        closed=sum(t.closed for t in trades),
        matched=sum(t.matched for t in trades),
    )
    for t in trades:
        result.written += trade_store.upsert(t)
    return result


# ── Review ────────────────────────────────────────────────────────────────


def review(trades: list[Trade]) -> list[dict]:
    """Per book: count, hit rate, mean and median return, P&L, holding; exits vs the rule.

    Option legs of a spread are left out of every return figure — one leg of
    a spread "returning" -80% says nothing — and reported as a row of their
    own with P&L only, which does add up across legs.
    """
    import statistics

    rows = []
    for book in Book:
        members = [
            t
            for t in trades
            if t.book is book and t.closed and t.ret is not None and not t.multi_leg
        ]
        if not members:
            continue
        rets = [t.ret for t in members]
        compared = [t for t in members if t.rule_ret is not None]
        rows.append(
            {
                "book": book.value,
                "n": len(members),
                "hit_rate": sum(r > 0 for r in rets) / len(rets),
                "mean_ret": statistics.fmean(rets),
                "median_ret": statistics.median(rets),
                "pnl": sum(t.pnl or 0 for t in members),
                "mean_sessions": statistics.fmean(t.sessions_held or 0 for t in members),
                "matched": sum(t.matched for t in members),
                "vs_rule": {
                    "n": len(compared),
                    "yours": statistics.fmean(t.ret for t in compared) if compared else None,
                    "rule": statistics.fmean(t.rule_ret for t in compared) if compared else None,
                },
            }
        )
    legs = [t for t in trades if t.multi_leg and t.closed]
    if legs:
        rows.append(
            {
                "book": "spread legs",
                "n": len(legs),
                "pnl": sum(t.pnl or 0 for t in legs),
                "note": "P&L only: a leg's own return is not a trade's",
            }
        )
    return rows


def unmatched(trades: list[Trade], coverage_start: date | None) -> tuple[list[Trade], int]:
    """Closed long trades the system could have seen and did not, and how many predate it.

    Only trades entered on or after ``coverage_start`` (the first session the
    scanner or the entry module recorded anything) can be "missed": before
    it the system was not looking, and calling those misses would be false.
    """
    closed = [t for t in trades if t.closed and t.direction == "long" and not t.multi_leg]
    if coverage_start is None:
        return [], len(closed)
    after = [t for t in closed if t.entry_session >= coverage_start]
    return [t for t in after if not t.matched], len(closed) - len(after)
