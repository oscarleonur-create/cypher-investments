"""The tracking board: for each name, what the system said, the risk it carries, how it closed.

One row per name held or proposed on lately: the latest proposal (action,
blockers, stale inputs, rules version), the risk it stands in (for a held
name, the volatility stop measured from cost — the same stop an EXIT fires
on; for a proposed entry, the leg's own stop), the proposal history as a
timeline of sessions, and the user's trades on it with what the system said
the day each was opened.

Read-only over the stores; nothing here reaches the broker or the network.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta

from pydantic import BaseModel, Field

from advisor.entry.proposal import Action, Proposal, position_stop_pct

TIMELINE_SESSIONS = 30
CLOSED_TRADES_SHOWN = 10
NEWS_DAYS = 7  # the news agent's judgments a row shows


class Point(BaseModel):
    session: date
    action: str
    price: float | None = None


class TradeLine(BaseModel):
    opened: date
    closed: date | None = None
    quantity: float
    entry_price: float
    exit_price: float | None = None
    pnl: float | None = None
    ret: float | None = None
    book: str
    # The system's call on the session the trade was opened: ENTER/ADD means
    # the user followed it, anything else means they acted without one.
    call: str | None = None


class Risk(BaseModel):
    basis: str  # "held": stop from cost | "planned": a proposed leg
    entry: float  # average cost, or the proposed entry
    price: float | None = None
    stop: float | None = None
    stop_basis: str = ""
    target: float | None = None  # where a trim is reviewed (P/S at its 2y 80th pct)
    shares: float = 0.0
    to_stop: float | None = None  # price / stop - 1: how far before the stop fires
    at_risk: float | None = None  # dollars lost if it stops from here
    at_risk_pct: float | None = None  # of net liq
    past_stop: bool = False


class Latest(BaseModel):
    session: date
    built_at: datetime
    action: str
    price: float | None = None
    blockers: list[str] = Field(default_factory=list)
    reasons: list[str] = Field(default_factory=list)
    stale: list[str] = Field(default_factory=list)
    rules: str | None = None  # the entry rules version it was made under
    legs: list[dict] = Field(default_factory=list)


class NewsLine(BaseModel):
    """One judged item: context only, never an action (user decision, 2026-09-27)."""

    published_at: datetime
    title: str
    provider: str
    url: str | None = None
    about_company: bool
    direction: str
    materiality: str
    event_type: str
    novelty: str
    basis: str
    why: str
    thesis: list[str] = Field(default_factory=list)  # "against (INVALIDATION)" ...
    what: str = ""
    magnitude: str = ""
    watch: str = ""
    market_read: str = "UNKNOWN"
    market: str | None = None
    read_from: str = "feed"  # article | feed | headline


class Row(BaseModel):
    symbol: str
    held: bool
    quantity: float = 0.0
    cost: float | None = None
    price: float | None = None
    weight: float | None = None
    unrealized: float | None = None  # dollars
    unrealized_pct: float | None = None
    latest: Latest | None = None
    risk: Risk | None = None
    timeline: list[Point] = Field(default_factory=list)
    open_trades: list[TradeLine] = Field(default_factory=list)
    closed_trades: list[TradeLine] = Field(default_factory=list)
    realized: float = 0.0  # dollars, every closed trade on file
    wins: int = 0
    losses: int = 0
    news: list[NewsLine] = Field(default_factory=list)
    news_summary: dict[str, int] = Field(default_factory=dict)
    # The news agent's weekly synthesis for the name, when one was written.
    news_week: dict | None = None


def _latest(p: Proposal) -> Latest:
    stale = (p.features or {}).get("stale")
    return Latest(
        session=p.session,
        built_at=p.built_at,
        action=p.action.value,
        price=p.price,
        blockers=list(p.blockers),
        reasons=[r.text for r in p.reasons[:4]],
        stale=stale.split(",") if isinstance(stale, str) and stale else [],
        rules=p.rules.version if p.rules else None,
        legs=[leg.model_dump() for leg in p.legs],
    )


def timeline(proposals: list[Proposal], sessions: int = TIMELINE_SESSIONS) -> list[Point]:
    """The last call of each session, oldest first, over the last ``sessions`` sessions."""
    last: dict[date, Proposal] = {}
    for p in proposals:
        cur = last.get(p.session)
        if cur is None or p.built_at > cur.built_at:
            last[p.session] = p
    days = sorted(last)[-sessions:]
    return [Point(session=d, action=last[d].action.value, price=last[d].price) for d in days]


def _target(proposals: list[Proposal]) -> float | None:
    """The newest trim-review price a position leg carried."""
    for p in sorted(proposals, key=lambda p: p.built_at, reverse=True):
        for leg in p.legs:
            if leg.horizon == "position" and leg.target:
                return leg.target
    return None


def held_risk(
    qty: float,
    cost: float,
    price: float | None,
    sigma: float | None,
    net_liq: float | None,
    target: float | None = None,
) -> Risk:
    """The stop an EXIT fires on — 2·σ·√10 below cost, in 8–25% — and what it puts at risk."""
    r = Risk(basis="held", entry=cost, price=price, shares=qty, target=target)
    pct = position_stop_pct(sigma)
    if pct is None or cost <= 0:
        r.stop_basis = "no volatility estimate: the stop cannot be set"
        return r
    r.stop = cost * (1 - pct)
    r.stop_basis = f"{pct:.1%} below cost (σ {sigma:.2%}/day)"
    if price:
        r.to_stop = price / r.stop - 1
        r.past_stop = price <= r.stop
        r.at_risk = max(price - r.stop, 0.0) * qty
        if net_liq and net_liq > 0:
            r.at_risk_pct = r.at_risk / net_liq
    return r


def planned_risk(p: Proposal) -> Risk | None:
    """The leg a proposal would open (the position leg first), and its risk."""
    legs = sorted(p.legs, key=lambda g: g.horizon != "position")
    if not legs:
        return None
    leg = legs[0]
    at_risk = max(leg.entry - leg.stop, 0.0) * leg.shares
    return Risk(
        basis="planned",
        entry=leg.entry,
        price=p.price,
        stop=leg.stop,
        stop_basis=f"{leg.horizon} leg: {leg.stop_basis}",
        target=leg.target,
        shares=leg.shares,
        to_stop=(p.price / leg.stop - 1) if p.price and leg.stop else None,
        at_risk=at_risk,
        at_risk_pct=leg.risk_pct,
    )


def _trade_line(t, calls: dict[tuple[str, date], str]) -> TradeLine:
    return TradeLine(
        opened=t.entry_session,
        closed=t.exit_session,
        quantity=t.quantity,
        entry_price=t.entry_price,
        exit_price=t.exit_price,
        pnl=t.pnl,
        ret=t.ret,
        book=t.book.value,
        call=calls.get((t.underlying.upper(), t.entry_session)),
    )


def _news_line(j) -> NewsLine:
    return NewsLine(
        published_at=j.published_at,
        title=j.title,
        provider=j.provider,
        url=j.url,
        about_company=j.about_company,
        direction=j.direction.value,
        materiality=j.materiality.value,
        event_type=j.event_type.value,
        novelty=j.novelty.value,
        basis=j.basis.value,
        why=j.why,
        thesis=[f"{'against' if c.against_thesis else 'for'} ({c.kind})" for c in j.claims],
        what=j.what,
        magnitude=j.magnitude,
        watch=j.watch,
        market_read=j.market_read.value,
        market=j.market,
        read_from=j.read_from,
    )


def build_board(
    proposals: list[Proposal],
    trades: list,
    book,
    news: list | None = None,
    weeks: dict | None = None,
) -> list[Row]:
    """Every name held or proposed on, held names first, largest first.

    ``news``: the news agent's judgments; a name with news but no proposal and
    no position is not given a row of its own.
    """
    from advisor.news.judge import summary

    news_by: dict[str, list] = {}
    for j in news or []:
        news_by.setdefault(j.symbol.upper(), []).append(j)
    by_symbol: dict[str, list[Proposal]] = {}
    for p in proposals:
        by_symbol.setdefault(p.symbol.upper(), []).append(p)

    # The call per (symbol, session): the strongest action of the session wins,
    # so a morning ENTER that turned WAIT at noon still reads as a call to enter.
    calls: dict[tuple[str, date], str] = {}
    for p in proposals:
        key = (p.symbol.upper(), p.session)
        if calls.get(key) not in (Action.ENTER.value, Action.ADD.value):
            calls[key] = p.action.value

    held: dict[str, list] = {}
    if book is not None:
        for pos in book.positions:
            if pos.quantity > 0 and not pos.is_option:
                held.setdefault(pos.underlying.upper(), []).append(pos)
    net_liq = book.net_liq if book is not None else None

    trades_by: dict[str, list] = {}
    for t in trades:
        if t.instrument == "Equity":
            trades_by.setdefault(t.underlying.upper(), []).append(t)

    rows = []
    for sym in sorted(set(by_symbol) | set(held)):
        props = by_symbol.get(sym, [])
        newest = max(props, key=lambda p: p.built_at) if props else None
        row = Row(symbol=sym, held=sym in held)
        if newest is not None:
            row.latest = _latest(newest)
        row.timeline = timeline(props)
        if sym in held:
            pos = held[sym]
            qty = sum(p.quantity for p in pos)
            basis = sum(p.cost_basis for p in pos)
            row.quantity = qty
            row.cost = basis / qty if qty else None
            row.price = pos[0].price or None
            row.unrealized = sum(p.unrealized_pnl for p in pos)
            row.unrealized_pct = row.unrealized / basis if basis else None
            row.weight = sum(p.notional for p in pos) / net_liq if net_liq else None
            sigma = (newest.features or {}).get("sigma") if newest is not None else None
            if row.cost:
                row.risk = held_risk(
                    qty, row.cost, row.price, sigma, net_liq, target=_target(props)
                )
        elif newest is not None:
            row.price = newest.price
            row.risk = planned_risk(newest)
        mine = trades_by.get(sym, [])
        closed = sorted((t for t in mine if t.closed), key=lambda t: t.closed_at, reverse=True)
        row.open_trades = [_trade_line(t, calls) for t in mine if not t.closed]
        row.closed_trades = [_trade_line(t, calls) for t in closed[:CLOSED_TRADES_SHOWN]]
        mine_news = sorted(news_by.get(sym, []), key=lambda j: j.published_at, reverse=True)
        row.news = [_news_line(j) for j in mine_news]
        row.news_summary = summary(mine_news) if mine_news else {}
        week = (weeks or {}).get(sym)
        row.news_week = week.model_dump(mode="json") if week is not None else None
        row.realized = sum(t.pnl or 0.0 for t in closed)
        row.wins = sum(1 for t in closed if (t.pnl or 0) > 0)
        row.losses = sum(1 for t in closed if (t.pnl or 0) < 0)
        rows.append(row)
    rows.sort(key=lambda r: (not r.held, -(r.weight or 0.0), r.symbol))
    return rows


def load_board(db_path, now: datetime, *, sessions: int = TIMELINE_SESSIONS) -> list[Row]:
    """``build_board`` over the stores in ``db_path``."""
    import sqlite3

    from advisor.daemon.store import DaemonStore
    from advisor.entry.store import EntryStore
    from advisor.learning.store import TradeStore
    from advisor.news.judge import NewsJudgmentStore, prompt_version

    since = now.date() - timedelta(days=int(sessions * 1.6) + 7)
    entries, daemon = EntryStore(db_path), DaemonStore(db_path)
    conn = sqlite3.connect(str(db_path))
    try:
        proposals = entries.list(since=since)
        book = daemon.load_latest_book()
        trades = TradeStore(conn).list()
        judged = NewsJudgmentStore(conn)
        news = judged.list(since=now - timedelta(days=NEWS_DAYS), version=prompt_version())
        weeks = {}
        for sym in {j.symbol for j in news}:
            week = judged.latest_summary(sym)
            if week is not None and (now.date() - week.day).days <= NEWS_DAYS:
                weeks[sym] = week
    finally:
        entries.close()
        daemon.close()
        conn.close()
    return build_board(proposals, trades, book, news, weeks)
