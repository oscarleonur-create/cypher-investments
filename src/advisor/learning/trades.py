"""The user's trades as round trips, from the broker, and what the rules would have done.

The scanner's journal knows only openings: a fill marks a candidate taken.
Learning needs the whole trade — where the user got out, what it made, how
long it was held — because "the setup works" and "my trades on it work" are
different claims, and the gap between them is where the user's own exits
live.

A trade is **one exit decision and the shares it closed**. Closing fills of
one session with no opening fill between them are one exit (an order filled
in pieces). Each exit closes lots: first those opened the same session (a
buy and sell within a day is a day trade, even inside a longer holding), then
the oldest (FIFO, the broker's own cost basis). What is still held is one
OPEN trade at the cost of the lots that remain — the broker's average price.

Why not flat to flat: AAOI was never flat from 2026-05-28 on. As one
episode it showed 28 shares at $155 (the peak and every buy's average) with
no outcome, while $1,184 had already been realized in three partial exits
and the broker held 12 shares at $129.32. Realized money must reach a book
when it is realized.

An exit that closes lots of different ages is split by book, so a day
trade on top of a holding counts as quick and the holding as hold.

Books are kept apart (the user's two strategies are never analysed together):

    quick         closed on the entry session or the next one (not a short sale:
                  every trade has its own direction, and these were all buys)
    hold          held more than five sessions
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
QUICK_MAX_SESSIONS = 1  # closed on the entry session or the next
HOLD_MIN_SESSIONS = 6  # more than five sessions: over a trading week
OPTION_MULTIPLIER = 100.0
# Rebuilt from here every time, so a long hold's entry is never cut off.
HISTORY_START = date(2024, 1, 1)


class Book(StrEnum):
    QUICK = "quick"  # closed by the next session: the trading book, NOT a short sale
    HOLD = "hold"  # held more than five sessions: the long-term book
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
    quantity: float  # shares (contracts) closed by this exit; still held if OPEN
    entry_price: float  # cost of those lots, size-weighted
    exit_price: float | None = None  # size-weighted over the exit's fills
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
        end = self.closed_at.isoformat() if self.closed_at else "open"
        return f"{self.account}:{self.symbol}:{self.opened_at.isoformat()}:{end}:{self.book.value}"

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
    return _book_for(t.sessions_held)


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


@dataclass
class _Lot:
    fill: Execution
    left: float  # signed: + long, - short

    @property
    def session(self) -> date:
        return mc.to_et(self.fill.executed_at).date()


def round_trips(executions: list[Execution], today: date | None = None) -> list[Trade]:
    """Closed trades per exit, and one OPEN trade per instrument still held. Pure given ``today``.

    A close with no open lot before it (history that starts mid-position) is
    dropped for the part it cannot match: that entry is outside the window and
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
    for fills in by_key.values():
        fills.sort(key=lambda f: f.executed_at)
        lots: list[_Lot] = []
        for exit_fills in _walk(fills, lots):
            trades.extend(_close(exit_fills, lots))
        if lots:
            trades.append(_open_trade(lots))

    _flag_multi_leg(trades)
    for t in trades:
        if t.closed:
            t.book = classify(t)
        expiry = option_expiry(t.symbol) if t.instrument == "Equity Option" else None
        if not t.closed and today is not None and expiry is not None and expiry < today:
            t.notes.append(f"open past its {expiry} expiry: expired or assigned, not read")
    return sorted(trades, key=lambda t: (t.opened_at, t.closed_at or t.opened_at))


def _walk(fills: list[Execution], lots: list[_Lot]):
    """Yield each exit (its closing fills) in order; opening fills become lots."""
    pending: list[Execution] = []
    for f in fills:
        if f.opens:
            if pending:
                yield pending
                pending = []
            lots.append(_Lot(f, f.signed))
            continue
        day = mc.to_et(f.executed_at).date()
        if pending and mc.to_et(pending[0].executed_at).date() != day:
            yield pending
            pending = []
        pending.append(f)
    if pending:
        yield pending


def _close(exit_fills: list[Execution], lots: list[_Lot]) -> list[Trade]:
    """Close lots against one exit — same-session lots first, then FIFO. One trade per book."""
    last = exit_fills[-1]
    day = mc.to_et(last.executed_at).date()
    exit_px = _vwap(exit_fills)
    want = sum(f.quantity for f in exit_fills)
    # Same-session lots, latest first; then the rest, oldest first.
    order = [x for x in reversed(lots) if x.session == day] + [x for x in lots if x.session != day]
    taken: list[tuple[_Lot, float]] = []
    for lot in order:
        if want <= 1e-9:
            break
        q = min(want, abs(lot.left))
        if q <= 1e-9:
            continue
        taken.append((lot, q))
        lot.left -= q if lot.left > 0 else -q
        want -= q
    lots[:] = [x for x in lots if abs(x.left) > 1e-9]
    if not taken:
        return []  # entered before the window: unpriceable

    by_book: dict[Book, list[tuple[_Lot, float]]] = defaultdict(list)
    for lot, q in taken:
        held = sessions_between(lot.session, day)
        by_book[_book_for(held)].append((lot, q))
    out = []
    for book, part in by_book.items():
        first = min(part, key=lambda x: x[0].fill.executed_at)[0]
        qty = sum(q for _, q in part)
        cost = sum(q * lot.fill.price for lot, q in part) / qty
        long_ = first.fill.action.startswith("Buy")
        sign = 1.0 if long_ else -1.0
        mult = OPTION_MULTIPLIER if first.fill.instrument == "Equity Option" else 1.0
        t = _base(first.fill, qty, cost, executions=len(part) + len(exit_fills))
        t.closed_at = last.executed_at
        t.exit_session = day
        t.exit_price = exit_px
        t.pnl = sign * (exit_px - cost) * qty * mult
        t.ret = sign * (exit_px / cost - 1) if cost > 0 else None
        t.sessions_held = sessions_between(t.entry_session, day)
        t.book = book
        out.append(t)
    return out


def _open_trade(lots: list[_Lot]) -> Trade:
    qty = sum(abs(x.left) for x in lots)
    cost = sum(abs(x.left) * x.fill.price for x in lots) / qty
    return _base(lots[0].fill, qty, cost, executions=len(lots))


def _base(first: Execution, qty: float, cost: float, *, executions: int) -> Trade:
    return Trade(
        account=first.account,
        underlying=first.underlying.upper(),
        symbol=first.symbol,
        instrument=first.instrument,
        direction="long" if first.action.startswith("Buy") else "short",
        opened_at=first.executed_at,
        entry_session=mc.to_et(first.executed_at).date(),
        quantity=qty,
        entry_price=cost,
        executions=executions,
    )


def _book_for(sessions_held: int) -> Book:
    if sessions_held <= QUICK_MAX_SESSIONS:
        return Book.QUICK
    if sessions_held >= HOLD_MIN_SESSIONS:
        return Book.HOLD
    return Book.UNCLASSIFIED


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
    removed: int = 0
    error: str | None = None

    def summary(self) -> str:
        if self.error:
            return f"broker unavailable: {self.error}"
        return (
            f"{self.executions} executions → {self.trades} trades ({self.closed} closed); "
            f"{self.written} written, {self.removed} superseded; "
            f"{self.matched} matched to a candidate or proposal"
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
    # Rebuilt whole: a row no longer produced (an OPEN trade whose lots have
    # since closed, a trade under a superseded id) is gone from the broker's
    # story and must not linger in the books.
    result.removed = trade_store.keep_only({t.id for t in trades})
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
