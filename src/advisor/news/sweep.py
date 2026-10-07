"""A daily news pull for every watched name: held, watchlist and Swing.

User, 2026-10-04: "signals no está actualizado para todos los tickers, la
mayoría no tiene news". Measured that day over the last seven days: AMZN,
INTC, MSFT, PENG and WOLF had no item at all, AMD one; of the held names AAOI
and CRDO had one each. Until then news reached a name only to explain
something already detected, or through the distress sweep and the angles —
both for held names only. Asked how to close it, the user chose Google News
and Tavily for every name.

Once a day, before the news agent's morning read (``news_judge`` at 08:30),
each name gets:

- **Google News**, free: the company's reviewed names (``news.names``) or its
  ticker, no keywords; a title must name the company to be kept, and at most
  ``GOOGLE_PER_NAME`` are kept, the strongest match first.
- **Tavily**, one paid query ("<company> <ticker> stock news") with the Yahoo
  feed that rides along, held names first, under ``TAVILY_BUDGET`` a run.

Everything goes through the same gates as any other item: entity match, the
date check (``news.verify``), the tier caps. What is found is archived and
emitted as Tier C context with ``explains: DAILY_NEWS``: it reaches the
digest and the news agent, never an interrupt.
"""

from __future__ import annotations

import asyncio
import logging
from collections.abc import Callable
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)

REASON = "DAILY_NEWS"
# Google items kept per name a run: the strongest match first, then the newest.
# Measured 2026-10-04: AMZN alone returned 42 items in two days, among them a
# patio-furniture deal, and every item kept is one the news agent must read.
GOOGLE_PER_NAME = 8
SWEEP_DAYS = 2  # a daily run reads two days, so a late-dated item is not missed
# Paid queries a run may spend, held names first (user decision, 2026-10-04:
# Tavily for every name; the universe was 15 names that day).
TAVILY_BUDGET = 15


@dataclass
class SweepResult:
    found: dict[str, int] = field(default_factory=dict)  # symbol -> items archived
    tavily: int = 0  # paid queries spent
    over_budget: list[str] = field(default_factory=list)  # names Tavily was not asked about
    errors: list[str] = field(default_factory=list)

    def detail(self) -> str:
        total = sum(self.found.values())
        empty = [s for s, n in self.found.items() if n == 0]
        out = f"{len(self.found)} names, {total} items, {self.tavily} Tavily queries"
        if self.over_budget:
            out += f"; over budget: {', '.join(self.over_budget)}"
        if empty:
            out += f"; nothing for {', '.join(empty)}"
        return out


def strongest(items: list) -> list:
    """The company named before a bare ticker, then the newest. Pure."""
    return sorted(items, key=lambda i: (-i.entity.confidence, -i.published_at.timestamp()))


def _google(store, symbol, company, website, *, search, verify) -> int:
    from advisor.news import google_news
    from advisor.news.ingest import context_events

    items = search(
        symbol,
        google_news.who(symbol, company),
        company_name=company,
        website=website,
        days=SWEEP_DAYS,
    )
    items = strongest(items)[:GOOGLE_PER_NAME]
    items = verify(items, store=store)
    for item in items:
        store.save_source_item(item)
    for event in context_events(items, reason=REASON):
        store.emit(event)
    return len(items)


def _tavily(store, symbol, company, *, explain) -> int:
    from advisor.news.ingest import context_events

    items = asyncio.run(
        explain(store, symbol, reason=REASON, company_name=company, days=SWEEP_DAYS)
    )
    for event in context_events(items, reason=REASON):
        store.emit(event)
    return len(items)


def sweep_universe(
    store,
    symbols: list[str],
    *,
    names: Callable[[str], str | None],
    websites: Callable[[str], str | None],
    budget: int = TAVILY_BUDGET,
    search=None,
    explain=None,
    verify=None,
) -> SweepResult:
    """Pull and archive the day's news for each name, in the order given. Never raises.

    ``symbols`` comes held-first (``daemon.universe.research_symbols``), so a
    budget that runs out leaves the watchlist's tail without Tavily, never the
    book. A failure on one name is recorded and the next name runs.
    """
    if search is None:
        from advisor.news.google_news import search_news as search
    if explain is None:
        from advisor.news.ingest import explain_symbol as explain
    if verify is None:
        from advisor.news.verify import verify_items as verify
    result = SweepResult()
    for symbol in symbols:
        found = 0
        company = website = None
        try:
            company = names(symbol)
            website = websites(symbol)
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"{symbol}: name lookup failed: {exc}")
        try:
            found += _google(store, symbol, company, website, search=search, verify=verify)
        except Exception as exc:  # noqa: BLE001
            result.errors.append(f"{symbol}: google news failed: {exc}")
        if result.tavily < budget:
            result.tavily += 1
            try:
                found += _tavily(store, symbol, company, explain=explain)
            except Exception as exc:  # noqa: BLE001
                result.errors.append(f"{symbol}: tavily failed: {exc}")
        else:
            result.over_budget.append(symbol)
        result.found[symbol] = found
    return result
