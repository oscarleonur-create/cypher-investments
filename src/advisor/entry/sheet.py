"""One name, today: what happened, how it moved, and where the price sits.

The deterministic half of a reading. No model, no judgement, nothing that
costs a credit — so it can be built for every watched name every day, and the
model is spent only where a sheet shows something changed.

It answers the user's own framing, in order: *today X events happened, the
stock moved Y%, and here is how demanding the price is against what the
business delivers*. Every field says where it came from, and every gap is
named: a sheet for a name with no valuation says so instead of reading as
"nothing to report".
"""

from __future__ import annotations

import logging
import math
import statistics
from collections.abc import Callable
from datetime import date, datetime

from pydantic import BaseModel, Field

from advisor.daemon import market_calendar as mc
from advisor.daemon.store import DaemonStore
from advisor.entry.zone import Absolute, RelativeZone, absolute_context, relative_zone

logger = logging.getLogger(__name__)

SIGMA_SESSIONS = 60


class Move(BaseModel):
    price: float
    asof: date  # the session the price belongs to
    day: float | None = None
    d5: float | None = None
    d20: float | None = None
    sigma: float | None = None  # daily, previous 60 sessions, today excluded
    z: float | None = None  # today's move in sigmas


class EventLine(BaseModel):
    ts: datetime
    kind: str
    tier: str
    text: str


class Holding(BaseModel):
    quantity: float
    weight: float  # of net liq
    unrealized: float  # fraction of cost


class Sheet(BaseModel):
    symbol: str
    built_at: datetime
    move: Move | None = None
    events_today: list[EventLine] = Field(default_factory=list)  # since the previous close
    events_week: int = 0
    holding: Holding | None = None
    zone: RelativeZone | None = None  # the entry zone: P/S vs its own two-year median
    zone_prev: RelativeZone | None = None  # the same, at the previous close: did it cross?
    context: Absolute | None = None  # what the price requires, generic and own margin
    candidates: list[str] = Field(default_factory=list)  # scanner ids for today
    # The user's long-term thesis on this name, if one is written: "intact"
    # (no rule broken or standing), "broken" (listing which), or None.
    thesis: str | None = None
    thesis_broken: list[str] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)

    @property
    def changed(self) -> bool:
        """Whether today is worth a model's reading: something material moved.

        A material event (tier A or B), a move of 2 sigma or more, or a
        scanner candidate. Quiet days get the sheet and nothing else.
        """
        if any(e.tier in ("A", "B") for e in self.events_today):
            return True
        if self.move and self.move.z is not None and abs(self.move.z) >= 2.0:
            return True
        return bool(self.candidates)


def move_from_closes(closes: list[tuple[date, float]], today: date) -> Move | None:
    """Day, 5- and 20-session moves and the move in sigmas. Pure.

    ``closes`` ascending, possibly ending with today's partial session. Sigma
    is taken from sessions before the latest, so today's move never inflates
    the yardstick it is measured with.
    """
    clean = [(d, c) for d, c in closes if c and math.isfinite(c) and c > 0]
    if len(clean) < 2:
        return None
    last_day, last = clean[-1]
    prior = [c for _, c in clean[:-1]]

    def back(n: int) -> float | None:
        return last / prior[-n] - 1 if len(prior) >= n else None

    returns = [b / a - 1 for a, b in zip(prior, prior[1:])][-SIGMA_SESSIONS:]
    sigma = statistics.stdev(returns) if len(returns) >= 20 else None
    day = back(1)
    return Move(
        price=last,
        asof=last_day,
        day=day,
        d5=back(5),
        d20=back(20),
        sigma=sigma if sigma and sigma > 0 else None,
        z=(day / sigma) if day is not None and sigma else None,
    )


def daily_closes(symbol: str) -> list[tuple[date, float]]:
    """Three years of daily closes: two for the zone's median, one of slack.

    [] on failure. Today's row is live during the session.
    """
    try:
        import yfinance as yf

        df = yf.download(
            symbol,
            period="3y",
            interval="1d",
            progress=False,
            auto_adjust=False,
            multi_level_index=False,
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("sheet: no prices for %s: %s", symbol, exc)
        return []
    if df is None or df.empty:
        return []
    return [(ts.date(), float(px)) for ts, px in df["Close"].items() if px == px]


def _event_line(event) -> EventLine:
    from advisor.daemon.summarize import summarize

    payload = event.payload or {}
    text = summarize(event) or event.kind.replace("_", " ").lower()
    if payload.get("lead"):
        text += f" — {payload['lead']}"
    return EventLine(ts=event.ts, kind=event.kind, tier=event.tier.value, text=text[:300])


THESIS_LOOKBACK_DAYS = 90


def thesis_status(store, symbol: str, book, now: datetime) -> tuple[str | None, list[str]]:
    """Is the user's thesis intact enough to deserve more risk? Pure over stored rows.

    Not the action card's reading. The card looks at *this week's* events: a
    rule broken ten days ago reads "untested" there, which is right for an
    alert and wrong here. AAOI's 6.7% at-the-market raise on 2026-09-06 broke
    the user's 5% dilution rule; by 09-25 the card said "your rules are
    intact", and an entry would have taken a bonus for a thesis the user's own
    rule had broken.

    So a rule counts as broken if any event in the last 90 days tripped it, or
    the current state violates it — unless the user has since recorded a
    decision on that rule. Returns:

        ("intact", [])         a thesis, nothing tripped            → more risk
        ("answered", [...])    tripped, and the user has answered   → no bonus
        ("broken", [...])      tripped, never answered              → wait
        (None, [])             no thesis written
    """
    from datetime import timedelta

    from advisor.thesis.match import evaluate_claim
    from advisor.thesis.repo import load_thesis
    from advisor.thesis.state import evaluate_against_state

    thesis = load_thesis(store, symbol)
    if thesis is None or not thesis.claims:
        return None, []
    events = store.recent_events(
        symbol=symbol, since=now - timedelta(days=THESIS_LOOKBACK_DAYS), limit=5000
    )
    decided = store.latest_decisions(symbol)
    broken, answered = [], []
    for claim in thesis.claims:
        if not claim.monitored or thesis.blocked.get(claim.id or ""):
            continue
        tripped = any(
            r is not None and r.tripped for r in (evaluate_claim(claim, e) for e in events)
        )
        if not tripped:
            standing = evaluate_against_state(store, symbol, claim, book)
            tripped = standing is not None and standing.tripped
        if tripped:
            (answered if claim.id and claim.id in decided else broken).append(claim.text)
    if broken:
        return "broken", broken
    if answered:
        return "answered", answered
    return "intact", []


def build_sheet(
    store: DaemonStore,
    symbol: str,
    now: datetime,
    *,
    closes: Callable[[str], list[tuple[date, float]]] = daily_closes,
    consensus_loader: Callable | None = None,
    series_loader: Callable | None = None,
    margin_loader: Callable | None = None,
    scanner_store=None,
) -> Sheet:
    """Everything stored plus prices and SEC series, for one name. Never raises on gaps."""
    from datetime import timedelta

    symbol = symbol.upper()
    now = mc.to_et(now)
    sheet = Sheet(symbol=symbol, built_at=now)

    history = closes(symbol)
    sheet.move = move_from_closes(history, now.date())
    if sheet.move is None:
        sheet.gaps.append("no price history")

    prev = mc.previous_trading_day(now.date())
    since = datetime.combine(prev, mc.session_close(prev), tzinfo=mc.MARKET_TZ)
    today_events = store.recent_events(symbol=symbol, since=since, limit=50)
    sheet.events_today = [_event_line(e) for e in today_events]
    sheet.events_week = len(
        store.recent_events(symbol=symbol, since=now - timedelta(days=7), limit=500)
    )

    book = store.load_latest_book()
    if book is not None:
        held = [p for p in book.positions if p.underlying.upper() == symbol]
        if held and book.net_liq > 0:
            basis = sum(p.cost_basis for p in held)
            sheet.holding = Holding(
                quantity=sum(p.quantity for p in held),
                weight=sum(p.notional for p in held) / book.net_liq,
                unrealized=(sum(p.unrealized_pnl for p in held) / basis) if basis else 0.0,
            )

    price = sheet.move.price if sheet.move else None

    if series_loader is None:
        from advisor.valuation.history import load_series as series_loader
    series = None
    try:
        series = series_loader(symbol)
    except Exception as exc:  # noqa: BLE001
        logger.info("sheet: SEC series unavailable for %s: %s", symbol, exc)
    if price:
        sheet.zone = relative_zone(history, series, now.date(), price)
        clean = [(d, c) for d, c in history if c and c == c and c > 0]
        if len(clean) >= 2:
            prev_day, prev_close = clean[-2]
            sheet.zone_prev = relative_zone(clean[:-1], series, prev_day, prev_close)
    if sheet.zone is None:
        sheet.gaps.append(
            "no entry zone (needs a price and six months of recent revenue and share history)"
        )

    snapshot = store.load_latest_valuation(symbol)
    if snapshot is None:
        sheet.gaps.append("no valuation (implied expectations never computed for this name)")
    else:
        consensus = None
        if consensus_loader is None:
            from advisor.valuation.consensus import load_consensus

            consensus_loader = load_consensus
        try:
            consensus = consensus_loader(store, symbol)
        except Exception as exc:  # noqa: BLE001
            logger.info("sheet: consensus unavailable for %s: %s", symbol, exc)
        if consensus is None:
            sheet.gaps.append("no consensus")
        own_margin = None
        if snapshot.margin_trailing is None:
            # A snapshot from before margins were recorded: fetch the trailing
            # one live rather than show a range of one.
            if margin_loader is None:
                from advisor.valuation.margins import load_trailing_margin

                def margin_loader(sym):
                    return load_trailing_margin(sym).margin

            own_margin = margin_loader(symbol)
        sheet.context = absolute_context(
            snapshot,
            price or snapshot.price,
            own_margin=own_margin,
            consensus=consensus,
        )

    if store.load_sensitivity(symbol) is None:
        sheet.gaps.append("no factor estimate (market vs own move cannot be split)")

    if book is not None and store.load_claims(symbol):
        try:
            sheet.thesis, sheet.thesis_broken = thesis_status(store, symbol, book, now)
        except Exception as exc:  # noqa: BLE001
            logger.info("sheet: thesis status unavailable for %s: %s", symbol, exc)
            sheet.gaps.append("thesis written but its status could not be read")

    if scanner_store is not None:
        sheet.candidates = [
            c.id for c in scanner_store.list(session=now.date(), limit=500) if c.symbol == symbol
        ]
    return sheet
