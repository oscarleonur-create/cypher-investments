"""Tier 2: dated, entity-tagged reporting, pulled only when something fires.

Tavily is already configured in this repo for the research agent. Two things
make it usable as a monitoring source that yfinance is not:

- ``topic="news"`` returns a real ``published_date``. Without that field an
  item cannot be aged or ordered against a price move. The existing client
  never sends it, which is why every result came back undated.
- Its results actually concern the company asked about. Measured on AAOI:
  6 of 6 named the company, against 0 of 3 from yfinance.

**It is pulled, not polled.** A query costs a credit, so news is fetched to
*explain* something a free detector already found — a residual divergence, a
stop crossing, a filing — rather than polled in the hope of finding one. That
is cheaper (roughly 70 credits a month against 330) and produces a better
query, because "why did AAOI fall 14% on 2026-08-24" beats "AAOI news".
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

import httpx

from advisor.news.entities import resolve_entity
from advisor.news.models import SourceItem, SourceTier

logger = logging.getLogger(__name__)

DEFAULT_LOOKBACK_DAYS = 7
MIN_RELEVANCE_SCORE = 0.5


def _parse_published(raw: str | None) -> datetime | None:
    """Tavily returns RFC 1123, e.g. 'Fri, 21 Aug 2026 20:23:01 GMT'."""
    if not raw:
        return None
    for fmt in ("%a, %d %b %Y %H:%M:%S %Z", "%a, %d %b %Y %H:%M:%S %z", "%Y-%m-%dT%H:%M:%S%z"):
        try:
            parsed = datetime.strptime(raw, fmt)
        except ValueError:
            continue
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)
    try:
        parsed = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        logger.debug("tavily: unparseable published_date %r", raw)
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def search_news(
    symbol: str,
    query: str,
    *,
    company_name: str | None = None,
    days: int = DEFAULT_LOOKBACK_DAYS,
    max_results: int = 6,
) -> list[SourceItem]:
    """Dated news for ``symbol``, entity-resolved and score-filtered.

    Returns an empty list on any failure — no key, no network, a rate limit —
    because a news outage must never stop the position or macro pillars.
    """
    try:
        from research_agent.config import ResearchConfig

        config = ResearchConfig()
    except Exception as exc:  # noqa: BLE001
        logger.info("tavily: config unavailable: %s", exc)
        return []

    if not config.tavily_api_key:
        logger.info("tavily: no API key configured, skipping news lookup")
        return []

    payload = {
        "api_key": config.tavily_api_key,
        "query": query,
        "topic": "news",  # the only mode that returns published_date
        "days": days,
        "max_results": max_results,
        "search_depth": "basic",
    }
    try:
        response = httpx.post(
            config.search_endpoint, json=payload, timeout=config.http_timeout_seconds
        )
        response.raise_for_status()
        results = response.json().get("results", [])
    except Exception as exc:  # noqa: BLE001
        logger.warning("tavily: news search failed for %s: %s", symbol, exc)
        return []

    out: list[SourceItem] = []
    undated = unresolved = 0
    for row in results:
        published = _parse_published(row.get("published_date"))
        if published is None:
            undated += 1
            continue
        if float(row.get("score") or 0.0) < MIN_RELEVANCE_SCORE:
            continue
        title = str(row.get("title") or "")
        body = f"{title} {row.get('content') or ''}"
        entity = resolve_entity(symbol, text=body, company_name=company_name)
        if not entity.resolved:
            unresolved += 1
            continue
        out.append(
            SourceItem(
                tier=SourceTier.AGGREGATOR,
                provider=str(row.get("url", "")).split("/")[2] if row.get("url") else "tavily",
                url=str(row.get("url") or ""),
                title=title,
                published_at=published,
                entity=entity,
                doc_type="NEWS",
                summary=(row.get("content") or None),
            )
        )

    if undated or unresolved:
        logger.info(
            "tavily %s: dropped %d undated, %d unresolvable to the ticker",
            symbol,
            undated,
            unresolved,
        )
    out.sort(key=lambda i: i.published_at, reverse=True)
    return out
