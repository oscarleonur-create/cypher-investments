"""Distress the SEC taxonomy has not mapped yet: news read for an exit.

A bankruptcy is filed on an 8-K, item 1.03 — and ``entry.exits`` reads that.
But the news comes first: advisers hired, a missed coupon, a going-concern
doubt, an exchange letter. The user's example (2026-09-26): a report of a
possible bankruptcy that nothing maps must still become an action, with its
rationale.

Three steps, and only the middle one is a model:

1. **Sweep.** Twice a day, per held name, Tavily searches the
   distress keywords in two short queries (``bankruptcy default delisting``,
   ``fraud investigation restructuring``) and Google News runs one boolean
   query over the same terms (``news.google_news``, free), all archived as
   tier-C context.
2. **Read.** The model sees the name's news of the last week, numbered, and
   answers three things: is any of it exit-grade (the company itself in or
   near bankruptcy, default, going-concern doubt, delisting, fraud or
   accounting investigation, a regulator stopping its main business, the
   loss of the customer that is most of its revenue, guidance withdrawn for
   liquidity), which situation, and which items report it. Speculation and
   opinion ("could X go bankrupt?") is WATCH, not EXIT_GRADE.
3. **Count, without the model.** Outlets are counted from the cited items'
   publishers, not from anything the model says. The user's rule
   (2026-09-26): an exit-grade report from two or more independent outlets
   with no filing yet is an EXIT, labeled unconfirmed; one outlet is a
   REVIEW. A filing, when it comes, is ``entry.exits``'s own EXIT.
   **The company's own word needs no second outlet** (user decision,
   2026-09-27): an exit-grade item published on the company's own website
   is an EXIT on its own. Press-release wires count together as one outlet,
   ``wire``: a wire distributes whoever pays, law firms' "investor alerts"
   included, so three wires are not three reporters.

The rationale shown is built from the cited items' own titles, publishers
and dates. The model's one-line reason is kept only if it carries no digit,
so it cannot introduce a number that is not in a source.

An EXIT_GRADE reading keeps asking for 14 days, like an exit filing: a
bankruptcy report does not expire because a week passed. A later reading of
NONE does not clear it; the user answers it (``advisor entry skip``).

Outlets are counted per story (``independent_outlets``): one article
syndicated under the same title on several sites is one outlet, and the
wires count as one between them. Known limit: two newsrooms rewriting one
agency story under different titles still count as two.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Callable
from datetime import datetime, timedelta
from enum import StrEnum

from pydantic import BaseModel, Field

from advisor.daemon.models import Event, EventSource, EventTier

logger = logging.getLogger(__name__)

# The two searches of one sweep; their keywords live in news.ingest.REASON_KEYWORDS,
# three words at most each (the measured rule: keywords, never sentences).
DISTRESS_REASONS = ("DISTRESS", "DISTRESS_PROBE")
SWEEP_DAYS = 3  # how far back one sweep searches
READ_WINDOW_DAYS = 7  # the news the model reads
STICKY_DAYS = 14  # how long an EXIT_GRADE reading keeps asking
MAX_ITEMS = 25
MIN_OUTLETS_FOR_EXIT = 2  # user decision, 2026-09-26
_NEWS_KINDS = ("NEWS_CONTEXT", "NEWS_ANGLE")


class Verdict(StrEnum):
    EXIT_GRADE = "EXIT_GRADE"
    WATCH = "WATCH"
    NONE = "NONE"


class Situation(StrEnum):
    BANKRUPTCY = "BANKRUPTCY"  # filed, preparing to, or advisers hired for a restructuring
    DEFAULT = "DEFAULT"  # missed payment, covenant breach, accelerated debt
    GOING_CONCERN = "GOING_CONCERN"  # auditor or company doubts it survives the year
    DELISTING = "DELISTING"  # exchange notice, suspension
    FRAUD_OR_INVESTIGATION = "FRAUD_OR_INVESTIGATION"  # SEC/DOJ probe, accounting fraud
    REGULATOR_HALT = "REGULATOR_HALT"  # a regulator stops its main business
    LOST_MAJOR_CUSTOMER = "LOST_MAJOR_CUSTOMER"  # the customer most revenue depends on
    GUIDANCE_WITHDRAWN = "GUIDANCE_WITHDRAWN"  # withdrawn for liquidity reasons
    OTHER = "OTHER"
    NONE = "NONE"


LABELS = {
    Situation.BANKRUPTCY: "bankruptcy or a restructuring",
    Situation.DEFAULT: "a default or missed payment",
    Situation.GOING_CONCERN: "doubt that it continues as a going concern",
    Situation.DELISTING: "a delisting or trading suspension",
    Situation.FRAUD_OR_INVESTIGATION: "a fraud or accounting investigation",
    Situation.REGULATOR_HALT: "a regulator stopping its main business",
    Situation.LOST_MAJOR_CUSTOMER: "the loss of its main customer",
    Situation.GUIDANCE_WITHDRAWN: "guidance withdrawn for liquidity",
    Situation.OTHER: "a threat to the company's survival",
    Situation.NONE: "nothing exit-grade",
}


class Draft(BaseModel):
    """What the model returns. Nothing here is trusted until checked."""

    verdict: Verdict
    situation: Situation
    fact_ids: list[str] = Field(default_factory=list)
    reason: str = ""


class Item(BaseModel):
    id: str
    title: str
    provider: str
    date: str
    url: str | None = None
    issuer: bool = False  # published on the company's own website


class DistressReading(BaseModel):
    symbol: str
    as_of: datetime
    status: str  # OK | NO_ITEMS | UNAVAILABLE | INVALID
    verdict: Verdict = Verdict.NONE
    situation: Situation = Situation.NONE
    cited: list[Item] = Field(default_factory=list)
    outlets: list[str] = Field(default_factory=list)
    reason: str = ""
    model: str | None = None
    items_read: int = 0

    @property
    def label(self) -> str:
        return LABELS[self.situation]

    @property
    def company_said(self) -> bool:
        """A cited item was published on the company's own website."""
        return any(i.issuer for i in self.cited)


SYSTEM_PROMPT = """\
You screen a stock's recent news for one question: does any item report that \
the company itself is in, or close to, a situation that ends the case for \
holding it?

Exit-grade situations: bankruptcy or restructuring (filed, preparing to, or \
advisers hired); default (missed payment, covenant breach, debt accelerated); \
going-concern doubt; delisting or trading suspension; a fraud or accounting \
investigation by a regulator or prosecutor; a regulator stopping its main \
business; the loss of the customer most of its revenue depends on; guidance \
withdrawn for liquidity reasons. Anything else of that severity is OTHER.

Rules:
- EXIT_GRADE only when an item REPORTS it as happening or announced. \
Speculation, opinion, listicles and questions ("could X go bankrupt?", \
"3 stocks to avoid") are WATCH at most.
- A situation at a different company (a customer, a peer) is not this \
company's, unless the item says it directly hits this company's survival.
- A law firm's "investor alert", a solicitation for a class action, or a \
shareholder lawsuit is not a regulator's investigation: WATCH at most.
- fact_ids: every item id that reports the situation, and only those.
- reason: one short sentence, no numbers or digits.
- If nothing qualifies: verdict NONE, situation NONE, fact_ids empty.
"""


def outlet(provider: str) -> str:
    """A publisher's name, normalised so one outlet is counted once. Pure."""
    p = provider.lower().strip()
    p = re.sub(r"^https?://", "", p)
    p = re.sub(r"^www\.", "", p)
    if "." in p and " " not in p:  # a domain: keep the registrable name
        labels = [x for x in p.split("/")[0].split(".") if x]
        while len(labels) > 1 and labels[-1] in _SUFFIXES:
            labels.pop()
        p = labels[-1]
    p = re.sub(r"[^a-z0-9]", "", p)
    return _ALIASES.get(p, p)


_SUFFIXES = {"com", "net", "org", "io", "co", "uk", "st", "ca", "au", "de", "news", "info", "biz"}

_ALIASES = {
    "themotleyfool": "fool",
    "motleyfool": "fool",
    "simplywallst": "simplywall",
    "yahoofinance": "yahoo",
    "reutersnews": "reuters",
    "bloombergnews": "bloomberg",
    "wsj": "wsj",
    "thewallstreetjournal": "wsj",
    "wallstreetjournal": "wsj",
    # Press-release wires: one outlet between them (see the module docstring).
    "businesswire": "wire",
    "prnewswire": "wire",
    "globenewswire": "wire",
    "accesswire": "wire",
    "newsfilecorp": "wire",
    "newsfile": "wire",
    "einpresswire": "wire",
    "einpresswirecom": "wire",
}


def news_items(store, symbol: str, now: datetime) -> list[Item]:
    """The name's archived news of the last READ_WINDOW_DAYS, newest first, numbered."""
    from advisor.news.verify import usable

    since = now - timedelta(days=READ_WINDOW_DAYS)
    items, seen = [], set()
    for e in store.recent_events(symbol=symbol, since=since, limit=500, kinds=_NEWS_KINDS):
        p = e.payload or {}
        if not usable(p) or _published(p, e) < since:
            continue  # unverified, or its checked date is older than the window
        title = str(p.get("title") or "").strip()
        if not title or title.lower() in seen:
            continue
        seen.add(title.lower())
        items.append(
            Item(
                id=f"N{len(items) + 1}",
                title=title[:200],
                provider=str(p.get("provider") or "unknown"),
                date=str(p.get("published_at") or e.ts.isoformat())[:10],
                url=p.get("url"),
                issuer=bool(p.get("issuer")),
            )
        )
        if len(items) >= MAX_ITEMS:
            break
    return items


def independent_outlets(cited: list[Item]) -> list[str]:
    """The outlets that reported independently: one per story, not per copy. Pure.

    A syndicated piece runs under one title on several sites — in the store on
    2026-09-27, "Cerebras Drops 19% in 3 Months" on Zacks and TradingView, a
    Motley Fool piece on The Globe and Mail — and counting each site would let
    one article make an EXIT. Items whose titles are the same story
    (``news.verify.same_story``) count once, under their earliest outlet.
    """
    from advisor.news.verify import same_story

    stories: list[list[Item]] = []
    for item in sorted(cited, key=lambda i: i.date):
        for story in stories:
            if same_story(story[0].title, item.title):
                story.append(item)
                break
        else:
            stories.append([item])
    return sorted({outlet(story[0].provider) for story in stories})


def _published(p: dict, e) -> datetime:
    raw = p.get("published_at")
    try:
        return datetime.fromisoformat(raw) if raw else e.ts
    except ValueError:
        return e.ts


def _user_prompt(symbol: str, company: str | None, items: list[Item]) -> str:
    lines = [f"Company: {company or symbol} ({symbol})", "", "News:"]
    lines += [
        f"{i.id} [{i.date}] ({i.provider}{', the company itself' if i.issuer else ''}) {i.title}"
        for i in items
    ]
    return "\n".join(lines)


def check(draft: Draft, items: list[Item]) -> tuple[Draft, list[str]]:
    """Keep only what the sources support. Pure. Returns (cleaned draft, problems)."""
    problems = []
    known = {i.id: i for i in items}
    ids = [f for f in dict.fromkeys(draft.fact_ids) if f in known]
    if len(ids) < len(set(draft.fact_ids)):
        problems.append("cited ids that do not exist were dropped")
    verdict, situation = draft.verdict, draft.situation
    if verdict is not Verdict.NONE and not ids:
        problems.append(f"{verdict.value} with no valid citation: read as NONE")
        verdict, situation = Verdict.NONE, Situation.NONE
    if verdict is Verdict.NONE:
        situation, ids = Situation.NONE, []
    elif situation is Situation.NONE:
        situation = Situation.OTHER
    reason = draft.reason.strip()
    if re.search(r"\d", reason):
        problems.append("the reason carried a number and was dropped")
        reason = ""
    return Draft(verdict=verdict, situation=situation, fact_ids=ids, reason=reason), problems


def read_distress(
    store,
    symbol: str,
    now: datetime,
    *,
    company: str | None = None,
    complete: Callable[[str, str], Draft] | None = None,
    model: str | None = None,
) -> DistressReading:
    """Read one name's news for exit-grade distress. Never raises."""
    symbol = symbol.upper()
    items = news_items(store, symbol, now)
    base = {"symbol": symbol, "as_of": now, "items_read": len(items)}
    if not items:
        return DistressReading(status="NO_ITEMS", **base)
    if complete is None:
        configured = _openrouter()
        if configured is None:
            return DistressReading(status="UNAVAILABLE", **base)
        complete, model = configured
    try:
        draft = complete(SYSTEM_PROMPT, _user_prompt(symbol, company, items))
    except Exception as exc:  # noqa: BLE001
        logger.warning("distress: model failed for %s: %s", symbol, exc)
        return DistressReading(status="UNAVAILABLE", model=model, **base)
    if not isinstance(draft, Draft):
        return DistressReading(status="INVALID", model=model, **base)
    draft, problems = check(draft, items)
    if problems:
        logger.info("distress %s: %s", symbol, "; ".join(problems))
    by_id = {i.id: i for i in items}
    cited = [by_id[f] for f in draft.fact_ids]
    return DistressReading(
        status="OK",
        verdict=draft.verdict,
        situation=draft.situation,
        cited=cited,
        outlets=independent_outlets(cited),
        reason=draft.reason,
        model=model,
        **base,
    )


def _openrouter():
    from research_agent.config import ResearchConfig
    from research_agent.llm import OpenRouterLLM

    config = ResearchConfig()
    if not config.openrouter_api_key:
        return None
    llm = OpenRouterLLM(config)
    return (lambda s, u: llm.complete(s, u, response_model=Draft)), config.llm_model


def reading_event(reading: DistressReading) -> Event:
    """The reading as a tier-C event: context, never an interrupt on its own."""
    cited = sorted(i.id + i.title for i in reading.cited)
    digest = hashlib.sha256(
        json.dumps([reading.verdict.value, reading.situation.value, cited]).encode()
    ).hexdigest()[:12]
    return Event(
        ts=reading.as_of,
        source=EventSource.COMPUTED,
        kind="DISTRESS_READING",
        tier=EventTier.C,
        symbol=reading.symbol,
        dedup_key=f"{reading.as_of.date().isoformat()}:{digest}",
        payload=json.loads(reading.model_dump_json()),
    )


def latest_distress(store, symbol: str, now: datetime) -> DistressReading | None:
    """The reading an exit should act on: an EXIT_GRADE one within STICKY_DAYS, else the newest."""
    rows = [
        e
        for e in store.recent_events(
            symbol=symbol, since=now - timedelta(days=STICKY_DAYS), limit=200
        )
        if e.kind == "DISTRESS_READING"
    ]
    readings = []
    for e in rows:
        try:
            readings.append(DistressReading.model_validate(e.payload))
        except Exception:  # noqa: BLE001
            continue
    if not readings:
        return None
    graded = [r for r in readings if r.verdict is Verdict.EXIT_GRADE]
    pool = graded or readings
    return max(pool, key=lambda r: r.as_of)


def sweep(
    store,
    symbol: str,
    *,
    company: str | None = None,
    website: str | None = None,
    google=None,
    verify_kwargs: dict | None = None,
) -> int:
    """The distress searches for one name, archived and emitted. Returns items found."""
    import asyncio

    from advisor.news import google_news
    from advisor.news.ingest import context_events, explain_symbol

    found = 0
    for reason in DISTRESS_REASONS:
        items = asyncio.run(
            explain_symbol(store, symbol, reason=reason, company_name=company, days=SWEEP_DAYS)
        )
        for event in context_events(items, reason=reason):
            store.emit(event)
        found += len(items)
    search = google or google_news.search_news
    items = search(
        symbol,
        google_news.distress_query(symbol, company, website),
        company_name=company,
        website=website,
        days=SWEEP_DAYS,
        title_filter=google_news.DISTRESS_TITLE,
    )
    from advisor.news.verify import verify_items

    items = verify_items(items, store=store, **(verify_kwargs or {}))
    for item in items:
        store.save_source_item(item)
    for event in context_events(items, reason="DISTRESS"):
        store.emit(event)
    return found + len(items)


def distress_all(
    store,
    symbols: list[str],
    now: datetime,
    *,
    search: bool = True,
    searcher=sweep,
    reader=read_distress,
    names: Callable[[str], str | None] | None = None,
    websites: Callable[[str], str | None] | None = None,
) -> tuple[list[DistressReading], list[str]]:
    """Sweep and read each name; store each reading. Returns (readings, errors)."""
    if names is None:
        from advisor.daemon.handlers import _company_name as names
    if websites is None:
        from advisor.daemon.handlers import _company_website as websites
    readings, errors = [], []
    for symbol in symbols:
        company = website = None
        try:
            company = names(symbol)
        except Exception as exc:  # noqa: BLE001
            errors.append(f"{symbol}: name lookup failed: {exc}")
        if search:
            try:
                website = websites(symbol)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{symbol}: website lookup failed: {exc}")
            try:
                searcher(store, symbol, company=company, website=website)
            except Exception as exc:  # noqa: BLE001
                errors.append(f"{symbol}: distress search failed: {exc}")
        reading = reader(store, symbol, now, company=company)
        if reading.status == "OK":
            store.emit(reading_event(reading))
        elif reading.status in ("UNAVAILABLE", "INVALID"):
            errors.append(f"{symbol}: distress reading {reading.status}")
        readings.append(reading)
    return readings, errors
