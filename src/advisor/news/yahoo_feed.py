"""Tier 3: the yfinance per-ticker feed, kept strictly as context.

This feed is thematic rather than about the company: three headlines filed
under AAOI concerned Nvidia, Broadcom and HPE, and none named AAOI. It is kept
because it is free and occasionally first, and because context sitting beside
a real event is useful even when it could never justify one on its own.

Every item is put through the same entity resolver as any other source, so
what survives is only what genuinely names the company. Whatever survives is
capped at Tier C by ``MAX_EVENT_TIER`` and can never interrupt.
"""

from __future__ import annotations

import logging

from advisor.data.news import news_items
from advisor.news.entities import resolve_entity
from advisor.news.models import SourceItem, SourceTier

logger = logging.getLogger(__name__)


def recent_context(
    symbol: str,
    *,
    company_name: str | None = None,
    max_age_hours: float = 48.0,
    limit: int = 5,
) -> list[SourceItem]:
    """Entity-resolved yfinance items, newest first."""
    out: list[SourceItem] = []
    dropped = 0
    for item in news_items(symbol, max_age_hours=max_age_hours, limit=limit * 4):
        entity = resolve_entity(symbol, text=item.title, company_name=company_name)
        if not entity.resolved:
            dropped += 1
            continue
        out.append(
            SourceItem(
                tier=SourceTier.UNTAGGED,
                provider=item.provider or "yfinance",
                url=item.url or "",
                title=item.title,
                published_at=item.published_at,
                entity=entity,
                doc_type="NEWS",
            )
        )
        if len(out) >= limit:
            break
    if dropped:
        logger.info("yfinance %s: %d item(s) did not name the company, dropped", symbol, dropped)
    return out
