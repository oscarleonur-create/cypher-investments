"""The morning read: what changed on each position since the last close, and what to do.

User, 2026-10-15: "se me está haciendo muy difícil consumir información sobre
mis posiciones y tomar decisiones, ayúdame a tener la información importante
en un lugar conciso con las fuentes para poder tener un morning read".

Everything here already exists somewhere — the Actions tab, each ticker's
overview, the news agent's judgments and weekly synthesis, the filings — but
spread over a page per name. This puts one card per holding on one page,
largest first, and says only:

    the position     shares, weight, from cost, the last session's move
    the call         the entry engine's newest call on it, and its first reason
    what is new      material news and filings since the last close (or the
                     last 24 hours, whichever is longer), each with its source,
                     time and link; HIGH before MEDIUM, newest first
    the week         the news agent's one-line synthesis, dated
    what is next     results date, and the agent's first thing to watch

Above the cards: what to do (the Actions tab's items), the book, and the
macro dates ahead. Below: watched names with material news since the close.

Deterministic: nothing is written by a model here that was not already
written and gated elsewhere (the judgments and the synthesis are the news
agent's, quoted as they are stored).
"""

from __future__ import annotations

import re
from datetime import date, datetime, timedelta

from pydantic import BaseModel, Field

from advisor.daemon import market_calendar as mc

NEW_SHOWN = 3  # items of news and filings per holding; the rest are counted
WATCHED_SHOWN = 6  # watched names with material news
MACRO_SESSIONS = 10  # macro dates this many sessions ahead
_RESULTS = re.compile(r"next results (\d{4}-\d{2}-\d{2}) \(in (\d+) sessions?\)")
_RANK = {"HIGH": 0, "MEDIUM": 1}
# Two items naming the same people, firms and figures are one story told twice.
# Measured 2026-10-15 on the held book: Barclays' Tesla-merger note at 247wallst
# and at Yahoo shared 0.54 of their names and numbers (Barclays, Dan, Levy,
# Tesla-SpaceX, 6); two different META stories at most 0.36. Plain words did
# not separate them (0.32 for the same story, long texts share filler).
SAME_STORY = 0.5
SAME_STORY_MIN = 3  # names and numbers in common, at least
_ENTITY = re.compile(r"\b(?:[A-Z][\w\-.]*[A-Za-z0-9]|\$?\d[\d,.]*[%MBbnk]*)\b")
_NOT_NAMES = frozenset(
    {
        "The",
        "A",
        "An",
        "In",
        "On",
        "It",
        "Its",
        "This",
        "That",
        "And",
        "For",
        "With",
        "As",
        "At",
        "By",
        "Of",
        "To",
        "But",
        "If",
        "He",
        "She",
        "They",
    }
)


class Item(BaseModel):
    kind: str  # news | filing
    text: str
    source: str  # the outlet or "SEC EDGAR <form>"
    at: datetime
    url: str | None = None
    direction: str | None = None  # POSITIVE | NEGATIVE | MIXED | NEUTRAL (news)
    materiality: str | None = None  # HIGH | MEDIUM (news)
    also: list[str] = Field(default_factory=list)  # other outlets carrying the same story


class Card(BaseModel):
    symbol: str
    shares: float
    weight: float | None = None
    from_cost: float | None = None
    day_move: float | None = None
    day_move_on: date | None = None  # the session the move is for
    price: float | None = None
    call: str | None = None
    call_why: str | None = None
    call_at: datetime | None = None
    action: str | None = None  # the Actions tab's line for it, when there is one
    answered: str | None = None  # an Actions item you answered that the book still shows
    new: list[Item] = Field(default_factory=list)
    more: int = 0  # material items not shown
    week: str | None = None
    week_net: str | None = None
    week_day: date | None = None
    results: date | None = None
    results_in: int | None = None
    watch: str | None = None


class Watched(BaseModel):
    symbol: str
    item: Item
    more: int = 0


class Morning(BaseModel):
    asof: datetime
    since: datetime
    book_asof: datetime | None = None
    net_liq: float | None = None
    actions: list[str] = Field(default_factory=list)
    macro: list[str] = Field(default_factory=list)
    cards: list[Card] = Field(default_factory=list)
    watched: list[Watched] = Field(default_factory=list)


def since_last_close(now: datetime) -> datetime:
    """The last close before ``now``, or 24 hours back if that is earlier. Pure.

    Before Monday's open that is Friday 16:00, so the weekend is in it; during a
    session, yesterday's close, so this morning's news is.
    """
    et = mc.to_et(now)
    d = et.date()
    if mc.is_trading_day(d) and et.time() >= mc.session_close(d):
        last = d
    else:
        last = mc.previous_trading_day(d)
    close = datetime.combine(last, mc.session_close(last), tzinfo=mc.MARKET_TZ)
    return min(close, et - timedelta(hours=24))


def news_items(judgments: list, since: datetime) -> list[Item]:
    """Material judged news about the company since ``since``: HIGH first, then newest. Pure."""
    keep = [
        j for j in judgments
        if j.about_company and j.materiality.value in _RANK and mc.to_et(j.published_at) >= since
    ]  # fmt: skip
    keep.sort(key=lambda j: (_RANK[j.materiality.value], -j.published_at.timestamp()))
    return [
        Item(
            kind="news",
            text=(j.what or j.title).strip(),
            source=j.provider,
            at=j.published_at,
            url=j.url,
            direction=j.direction.value,
            materiality=j.materiality.value,
        )  # fmt: skip
        for j in keep
    ]


def _names(text: str) -> set[str]:
    """The capitalised names and the figures in ``text``. Pure."""
    return {e for e in _ENTITY.findall(text) if e not in _NOT_NAMES}


def one_per_story(items: list[Item]) -> list[Item]:
    """Fold items that tell the same story into the first, naming the other outlets. Pure.

    ``items`` come in rank order, so the kept one is the most material and newest.
    """
    kept: list[tuple[Item, set[str]]] = []
    for item in items:
        words = _names(item.text)
        for k, kw in kept:
            common = len(words & kw)
            if common >= SAME_STORY_MIN and common / min(len(words), len(kw)) >= SAME_STORY:
                if item.source not in k.also and item.source != k.source:
                    k.also.append(item.source)
                break
        else:
            kept.append((item, words))
    return [k for k, _ in kept]


def filing_items(items: list, since: datetime) -> list[Item]:
    """The company's own filings since ``since``, newest first. Pure."""
    out = []
    for i in items:
        if i.tier.value != "PRIMARY" or mc.to_et(i.published_at) < since:
            continue
        form = i.doc_type or "filing"
        codes = f" [{', '.join(i.item_codes)}]" if getattr(i, "item_codes", None) else ""
        lead = (i.summary or "").strip()
        out.append(Item(kind="filing", text=f"{form}{codes}: {lead or i.title}", at=i.published_at,
                        source=f"{i.provider} {form}", url=i.url))  # fmt: skip
    return sorted(out, key=lambda x: x.at, reverse=True)


def order(x: Item) -> tuple:
    """Filings first (the company said it), then HIGH before MEDIUM news, newest first. Pure."""
    rank = 0 if x.kind == "filing" else 1 + _RANK.get(x.materiality or "", 1)
    return (rank, -x.at.timestamp())


def results_date(reasons: list[str]) -> tuple[date | None, int | None]:
    """The next results date a proposal's reasons name. Pure."""
    for r in reasons:
        m = _RESULTS.search(r)
        if m:
            return date.fromisoformat(m.group(1)), int(m.group(2))
    return None, None


def day_move(bars: list[tuple[str, float]], holding: dict) -> tuple[float | None, date | None]:
    """The last session's move: from the stored daily bars, else the book. Pure.

    The broker's ``close_price`` on a lot opened or added that day is the fill,
    not the prior close (SPCX and NBIS on 2026-10-15 read their move from cost
    as the day's move), so the book is trusted only where it differs from cost.
    """
    if len(bars) >= 2 and bars[-2][1]:
        return bars[-1][1] / bars[-2][1] - 1, date.fromisoformat(bars[-1][0])
    close, price, cost = holding.get("close"), holding.get("price"), holding.get("cost")
    if close and price and not (cost and abs(close - cost) < 1e-6):
        return price / close - 1, None
    return None, None


def card_for(
    sym, holding, net_liq, proposal, actionable, judgments, filings, summary, since,
    bars=(), answered=None,
) -> Card:  # fmt: skip
    """One holding's card. Pure."""
    c = Card(symbol=sym, shares=holding["shares"], price=holding["price"])
    if net_liq:
        c.weight = holding["notional"] / net_liq
    if holding["cost"] and holding["price"]:
        c.from_cost = holding["price"] / holding["cost"] - 1
    c.day_move, c.day_move_on = day_move(list(bars), holding)
    c.answered = answered
    if proposal is not None:
        c.call = proposal.action.value
        c.call_at = proposal.built_at
        reasons = [r.text for r in proposal.reasons]
        c.call_why = reasons[0] if reasons else None
        c.results, c.results_in = results_date(reasons)
    c.action = actionable
    items = one_per_story(
        sorted(news_items(judgments, since) + filing_items(filings, since), key=order)
    )
    c.new, c.more = items[:NEW_SHOWN], max(len(items) - NEW_SHOWN, 0)
    if summary is not None:
        c.week, c.week_net, c.week_day = summary.headline, summary.net.value, summary.day
        c.watch = summary.watch[0] if summary.watch else None
    return c


def holdings(book) -> dict[str, dict]:
    """Long equity positions per symbol, accounts summed. Pure."""
    out: dict[str, dict] = {}
    for p in book.positions if book is not None else []:
        if p.quantity <= 0 or p.is_option:
            continue
        h = out.setdefault(p.underlying.upper(), {"shares": 0.0, "basis": 0.0, "notional": 0.0,
                                                  "price": p.price or None,
                                                  "close": p.close_price or None})  # fmt: skip
        h["shares"] += p.quantity
        h["basis"] += p.cost_basis
        h["notional"] += p.notional
    for h in out.values():
        h["cost"] = h["basis"] / h["shares"] if h["shares"] else None
    return out


def build(book, proposals: dict, actionables: dict[str, str], judgments: dict, filings: dict,
          summaries: dict, watched: list[str], now: datetime, bars: dict | None = None,
          answered: dict | None = None) -> Morning:  # fmt: skip
    """The morning read from what is stored. Pure.

    ``proposals``: newest per symbol; ``actionables``: symbol -> the Actions
    tab's title; ``judgments``/``filings``/``summaries``: per symbol.
    """
    since = since_last_close(now)
    held = holdings(book)
    net_liq = book.net_liq if book is not None else None
    m = Morning(asof=now, since=since, book_asof=book.as_of if book is not None else None,
                net_liq=net_liq)  # fmt: skip
    from advisor.macro.calendar import upcoming

    m.macro = [
        f"{d.isoformat()}: {label} ({'today' if n == 0 else f'in {n} sessions'})"
        for d, label, n in upcoming(mc.to_et(now).date(), MACRO_SESSIONS)
    ]
    m.actions = list(actionables.get("_all", []))
    for sym, h in sorted(held.items(), key=lambda kv: -kv[1]["notional"]):
        m.cards.append(
            card_for(sym, h, net_liq, proposals.get(sym), actionables.get(sym),
                     judgments.get(sym, []), filings.get(sym, []), summaries.get(sym), since,
                     (bars or {}).get(sym, ()), (answered or {}).get(sym))
        )  # fmt: skip
    for sym in watched:
        if sym in held:
            continue
        items = news_items(judgments.get(sym, []), since) + filing_items(
            filings.get(sym, []), since
        )
        if items:
            items.sort(key=order)
            m.watched.append(Watched(symbol=sym, item=items[0], more=len(items) - 1))
    m.watched = m.watched[:WATCHED_SHOWN]
    return m


def load(db_path, now: datetime, *, symbols: list[str] | None = None) -> dict:
    """``build`` over the stores. ``symbols``: the watched universe (default: proposals on file)."""
    import sqlite3
    from pathlib import Path

    from advisor.daemon.store import DaemonStore
    from advisor.entry import actionables as A
    from advisor.entry.store import EntryStore
    from advisor.news.judge import NewsJudgmentStore, prompt_version

    db_path = Path(db_path)
    since = since_last_close(now)
    daemon, entries = DaemonStore(db_path), EntryStore(db_path)
    conn = sqlite3.connect(str(db_path))
    try:
        book = daemon.load_latest_book()
        recent = entries.list(since=now.date() - timedelta(days=A.PROPOSAL_DAYS))
        newest = A.newest_session(recent)
        universe = sorted(set(symbols or newest) | set(holdings(book)))
        judged = NewsJudgmentStore(conn)
        judgments, summaries, filings = {}, {}, {}
        for sym in universe:
            judgments[sym] = judged.list(symbol=sym, since=since, version=prompt_version())
            summaries[sym] = judged.latest_summary(sym)
            filings[sym] = daemon.source_items_between(sym, since, now + timedelta(hours=1))
    finally:
        daemon.close()
        entries.close()
        conn.close()
    loaded = A.load(db_path, now)
    acts = loaded["items"]
    by_sym: dict = {"_all": [a["title"] for a in acts]}
    for a in acts:
        by_sym.setdefault(a["symbol"], a["title"])
    answered = {a["id"].split(":")[0]: f"{a['title']} — {a['why']}" for a in loaded["answered"]}
    m = build(book, newest, by_sym, judgments, filings, summaries, universe, now,
              bars=_bars(db_path, list(holdings(book)), now), answered=answered)  # fmt: skip
    return m.model_dump(mode="json")


def _bars(db_path, symbols: list[str], now: datetime) -> dict[str, list[tuple[str, float]]]:
    """The last two stored daily closes per symbol, on or before today. Empty when absent."""
    import sqlite3

    from advisor.breadth.store import breadth_path

    path = breadth_path(db_path)
    if not path.exists() or not symbols:
        return {}
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        out = {}
        for s in symbols:
            rows = conn.execute(
                "SELECT day, close FROM breadth_bars WHERE symbol = ? AND day <= ? "
                "ORDER BY day DESC LIMIT 2",
                (s, mc.to_et(now).date().isoformat()),
            ).fetchall()
            out[s] = list(reversed(rows))
        return out
    finally:
        conn.close()
