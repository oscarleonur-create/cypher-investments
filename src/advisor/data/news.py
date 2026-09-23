"""Free-tier news and earnings-proximity helpers.

Salvaged from the removed scalping catalyst layer. These are the zero-cost
sources the daemon's ingest layer runs on: yfinance headlines and the yfinance
earnings calendar. No LLM, no paid feed.

**Everything here fails closed.** A free feed will hand you a plainly wrong
value with exactly the same confidence as a right one — yfinance reports AMD's
ex-dividend date as 1995-04-26, for a company that pays no dividend — so an
item whose timestamp cannot be established is dropped rather than passed
through undated. Silence is a recoverable failure; a headline from March
presented as this morning's is not.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timezone

logger = logging.getLogger(__name__)


def earnings_context(symbol: str) -> tuple[bool, int | None]:
    """Return ``(earnings_today, days_to_earnings)`` using the yfinance calendar.

    The date is yfinance's, and yfinance does not distinguish a company-
    confirmed date from one it estimated off last year's pattern. Treat the
    result as approximate: fine for "earnings are near, size accordingly",
    not sound as the trigger for a dated action.
    """
    from advisor.data.yahoo import fetch_earnings_dates

    today = date.today()
    deltas = [(d - today).days for d in fetch_earnings_dates(symbol)]
    if not deltas:
        return False, None
    # nearest absolute, but prefer the soonest upcoming
    upcoming = [x for x in deltas if x >= 0]
    nearest = min(upcoming) if upcoming else max(deltas)
    return (nearest == 0), nearest


@dataclass(frozen=True)
class NewsItem:
    """One headline with the provenance needed to judge it."""

    title: str
    published_at: datetime  # always timezone-aware UTC
    provider: str | None = None
    url: str | None = None

    def age_hours(self, now: datetime | None = None) -> float:
        return ((now or datetime.now(timezone.utc)) - self.published_at).total_seconds() / 3600


def _published_at(raw: dict) -> datetime | None:
    """Extract a timezone-aware publish time, or None if there isn't one.

    yfinance has changed this shape at least twice. The flat schema carried
    ``providerPublishTime`` as a unix epoch; the current nested one carries
    ``content.pubDate`` as ISO 8601 and drops the old field entirely. Reading
    only the old name silently disabled every age filter in this module —
    ``news_headlines(symbol, max_age_hours=0)`` returned five headlines — so
    both are read, and an item matching neither is refused.
    """
    content = raw.get("content", raw)

    epoch = raw.get("providerPublishTime") or content.get("providerPublishTime")
    if epoch is not None:
        try:
            return datetime.fromtimestamp(float(epoch), tz=timezone.utc)
        except (TypeError, ValueError, OSError, OverflowError):
            pass

    for field in ("pubDate", "displayTime"):
        value = content.get(field)
        if not value:
            continue
        try:
            parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        except ValueError:
            continue
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)

    return None


def news_items(symbol: str, max_age_hours: float = 48.0, limit: int = 5) -> list[NewsItem]:
    """Recent yfinance news for a symbol, newest first, undated items dropped.

    Note that yfinance's per-ticker feed is thematic rather than strict: a
    query for AMD routinely returns headlines whose subject is Nvidia or
    Broadcom. Callers that need "news *about* this company" must filter
    further; this returns "news yfinance filed under this ticker".
    """
    try:
        import yfinance as yf

        raw_items = yf.Ticker(symbol.upper()).news or []
    except Exception as exc:  # noqa: BLE001
        logger.debug("news_items failed for %s: %s", symbol, exc)
        return []

    now = datetime.now(timezone.utc)
    out: list[NewsItem] = []
    undated = 0
    for raw in raw_items:
        content = raw.get("content", raw)
        title = content.get("title") or raw.get("title")
        if not title:
            continue
        published = _published_at(raw)
        if published is None:
            undated += 1
            continue
        item = NewsItem(
            title=str(title),
            published_at=published,
            provider=(content.get("provider") or {}).get("displayName")
            if isinstance(content.get("provider"), dict)
            else content.get("provider"),
            url=(content.get("canonicalUrl") or {}).get("url")
            if isinstance(content.get("canonicalUrl"), dict)
            else content.get("link"),
        )
        if item.age_hours(now) > max_age_hours:
            continue
        out.append(item)

    if undated:
        logger.info("news: dropped %d undated item(s) for %s", undated, symbol.upper())
    out.sort(key=lambda i: i.published_at, reverse=True)
    return out[:limit]


def news_headlines(symbol: str, max_age_hours: float = 48.0, limit: int = 5) -> list[str]:
    """Recent yfinance news titles for a symbol (free, no LLM)."""
    return [item.title for item in news_items(symbol, max_age_hours=max_age_hours, limit=limit)]
