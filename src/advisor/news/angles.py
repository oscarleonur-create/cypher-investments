"""Angles: the products and segments a holding is really a bet on.

SPCX is 21% of net liq and its thesis is about Starlink cash funding an AI
segment. Grok 4.7 shipped on 2026-09-21 and the advisor never saw it: the
news query asked for "SPACE EXPLORATION TECHNOLOGIES CORP SPCX stock news",
and when Tavily was asked about Grok directly it returned six articles that
the entity matcher dropped, because "SpaceXAI Releases Grok 4.7" names
neither the registered company nor its ticker.

An angle is a term the holder has confirmed stands for part of the company.
It changes two things and nothing else:

- **Retrieval.** Confirmed angles of held positions are searched once a day
  by the review job, under a hard credit budget. This is the one deliberate
  exception to "pulled, never polled", decided by the user on 2026-09-24.
- **Matching.** An article that names a confirmed angle is linked to the
  holding as ``MatchMethod.ALIAS`` — the weakest match, because the text did
  not earn it; the holder vouched for it. Its events are capped at Tier C:
  context for the digest and the reading, never an interrupt.

Suggestions come from the holder's own claims, deterministically, and are
only suggestions: nothing is searched until the holder confirms it.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from enum import StrEnum

from advisor.daemon.models import Event, EventSource, EventTier
from advisor.daemon.store import DaemonStore
from advisor.news.lead import lead_for
from advisor.news.models import SourceItem, capped_tier

logger = logging.getLogger(__name__)

MAX_ANGLES_PER_SYMBOL = 6
# Tavily credits the daily scan may spend across the whole book.
DAILY_QUERY_BUDGET = 20
SCAN_DAYS = 3


class AngleStatus(StrEnum):
    SUGGESTED = "SUGGESTED"
    CONFIRMED = "CONFIRMED"
    REJECTED = "REJECTED"


# Capitalised words that are not products: sentence openers, and the finance
# vocabulary the holder's claims are written in. A suggestion the list misses
# costs one click to reject; nothing is searched until it is confirmed.
_NOT_ANGLES = {
    # English and Spanish function words that open a sentence.
    "the", "a", "an", "any", "if", "when", "this", "that", "it", "its", "and", "or",
    "el", "la", "los", "las", "un", "una", "unos", "unas", "si", "cualquier", "cualquiera",
    "cuando", "este", "esta", "ese", "esa", "su", "sus", "que", "y", "o", "de", "del",
    "en", "por", "para", "con", "sin", "hoy", "no",
    # Finance and filing vocabulary.
    "ebitda", "opv", "ipo", "fcf", "cagr", "eps", "ev", "ia", "ai", "kpi", "sec", "ceo",
    "cfo", "coo", "us", "usa", "eeuu", "gaap", "atm", "capex", "roic", "yoy", "qoq",
    "q1", "q2", "q3", "q4", "fy", "10-q", "10-k", "8-k", "6-k", "20-f", "nasdaq", "nyse",
}  # fmt: skip

# A name-like token: capitalised ("Starlink"), camel-case ("xAI", "SpaceXAI"),
# or an acronym, with inner dots or hyphens ("Grok-4").
_NAME = re.compile(r"(?<![\w$])(?:[A-Z][A-Za-z0-9]*|[a-z]+[A-Z][A-Za-z0-9]*)(?:[.\-][A-Za-z0-9]+)*")


def suggest_angles(
    claims: list[tuple[str | None, str]], *, symbol: str, company_name: str | None = None
) -> list[tuple[str, str | None]]:
    """(term, claim id) candidates from the holder's claim texts, first seen first."""
    company = {w.lower() for w in re.findall(r"[A-Za-z]+", company_name or "")}
    seen: dict[str, str | None] = {}
    for claim_id, text in claims:
        for token in _NAME.findall(re.sub(r"[+/]", " ", text)):
            key = token.lower()
            if (
                len(token) < 3
                or key in _NOT_ANGLES
                or key in company
                or key == symbol.lower()
                or token.isdigit()
            ):
                continue
            seen.setdefault(token, claim_id)
    return list(seen.items())


def refresh_suggestions(store: DaemonStore, symbol: str, *, company_name: str | None) -> int:
    """Record new suggestions from the holder's claims. Never touches a decided angle."""
    claims = [(c.id, c.text) for c in store.load_claims(symbol)]
    added = 0
    for term, claim_id in suggest_angles(claims, symbol=symbol, company_name=company_name):
        if store.add_angle(symbol, term, AngleStatus.SUGGESTED.value, source=claim_id):
            added += 1
    return added


@dataclass
class AngleScan:
    queries: int = 0
    items_stored: int = 0
    events: list[Event] = field(default_factory=list)
    skipped_for_budget: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def angle_event(item: SourceItem, term: str) -> Event:
    """One tier-C context row for an article found through an angle."""
    return Event(
        source=EventSource.CALENDAR,
        kind="NEWS_ANGLE",
        tier=capped_tier(item.tier, EventTier.C),
        symbol=item.entity.symbol,
        # Keyed on the article, so one piece matched by two angles is one event.
        dedup_key=item.dedup_key(),
        payload={
            "title": item.title,
            "url": item.url,
            "provider": item.provider,
            "published_at": item.published_at.isoformat(),
            "match": item.entity.method.value,
            "confidence": item.entity.confidence,
            "angle": term,
            "lead": lead_for(item),
        },
    )


def scan_angles(
    store: DaemonStore,
    symbols: list[str],
    *,
    search=None,
    company_names: dict[str, str | None] | None = None,
    days: int = SCAN_DAYS,
    budget: int = DAILY_QUERY_BUDGET,
) -> AngleScan:
    """Search each confirmed angle of each symbol once, within ``budget`` queries.

    Symbols are taken in the order given — the caller passes the book largest
    position first — so a budget that runs out starves the smallest holdings.
    """
    if search is None:
        from advisor.news.tavily import search_news as search

    result = AngleScan()
    names = company_names or {}
    for symbol in symbols:
        terms = [a["term"] for a in store.list_angles(symbol, status=AngleStatus.CONFIRMED.value)]
        for term in terms[:MAX_ANGLES_PER_SYMBOL]:
            if result.queries >= budget:
                result.skipped_for_budget.append(f"{symbol}:{term}")
                continue
            result.queries += 1
            try:
                items = search(
                    symbol, term, company_name=names.get(symbol), days=days, aliases=[term]
                )
            except Exception as exc:  # noqa: BLE001
                result.errors.append(f"{symbol}:{term}: {exc}")
                continue
            for item in items:
                if store.save_source_item(item):
                    result.items_stored += 1
                event = angle_event(item, term)
                if store.emit(event):
                    result.events.append(event)
    if result.skipped_for_budget:
        logger.warning(
            "angles: budget of %d queries reached; skipped %s", budget, result.skipped_for_budget
        )
    return result
