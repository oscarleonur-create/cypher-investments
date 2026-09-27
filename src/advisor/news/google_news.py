"""Google News RSS: a second, free aggregator for the distress sweep.

Tavily is a search engine: for a held name it returns what the web holds,
mostly commentary (Motley Fool, Simply Wall St, Zacks). Google News indexes
the wires and the newsrooms that report first — Reuters, CNBC, the company's
own site — and names the publisher of every item with its domain. Measured
2026-09-27 over the held book: 2 to 100 items a name in a week, 5 to 58
publishers, and the company's own newsroom among them for AAOI, CBRS, CRDO,
NBIS and SPCX.

Two things set an item apart:

- **The publisher is a domain**, so ``entry.distress.outlet`` counts it the
  same way as Tavily's.
- **An item published on the company's own website is the company
  speaking** (tier PRIMARY, ``doc_type`` ``COMPANY_STATEMENT``), decided by
  domain against the website the company lists, never by the text. A press
  release on a wire is not: law firms put "investor alerts" about a company
  on the same wires, so a wire is an outlet like any other.

The feed carries titles only. Matching is on the title, like every other
source (``news.entities``), plus the brand in the company's web address
when it is one word ("spacex" for Space Exploration Technologies, whose
registered name no headline uses).

Free and unmetered, so it is not under the Tavily budget; any failure returns
nothing, because a news outage must never stop the pillars.
"""

from __future__ import annotations

import logging
import re
import xml.etree.ElementTree as ET
from collections.abc import Callable
from datetime import datetime, timedelta
from email.utils import parsedate_to_datetime
from urllib.parse import quote, urlparse

from advisor.daemon.market_calendar import now_et
from advisor.news.entities import is_ambiguous, normalize_name, resolve_entity
from advisor.news.models import EntityMatch, MatchMethod, SourceItem, SourceTier

logger = logging.getLogger(__name__)

SEARCH_URL = "https://news.google.com/rss/search?q={q}&hl=en-US&gl=US&ceid=US:en"
MAX_RESULTS = 40

# The distress sweep's terms. Google takes boolean OR, so this is one query,
# still keywords, never a sentence.
DISTRESS_TERMS = (
    "bankruptcy",
    '"Chapter 11"',
    "default",
    "delisting",
    '"going concern"',
    "fraud",
    "investigation",
    "restructuring",
)

# Hosting labels that are not a brand ("www.ao-inc.com" -> "ao-inc" is kept,
# but not its "www").
_HOST_NOISE = {"www", "investors", "investor", "ir", "newsroom", "news", "press"}


def domain(url: str | None) -> str:
    """The registrable part of a URL's host: 'https://newsroom.ao-inc.com/x' -> 'ao-inc.com'."""
    if not url:
        return ""
    host = urlparse(url if "//" in url else f"//{url}").hostname or ""
    labels = [x for x in host.lower().split(".") if x]
    if len(labels) >= 3 and labels[-2] in {"co", "com"} and len(labels[-1]) == 2:
        return ".".join(labels[-3:])  # example.co.uk
    return ".".join(labels[-2:])


def brand(website: str | None) -> str | None:
    """The one-word brand in a company's web address, if it is one: spacex.com -> 'spacex'."""
    d = domain(website)
    stem = d.split(".")[0] if d else ""
    if len(stem) >= 4 and stem.isalpha() and stem not in _HOST_NOISE:
        return stem
    return None


def distress_query(symbol: str, company: str | None, website: str | None = None) -> str:
    """One boolean query: the company's names, and the distress terms."""
    names = []
    core = normalize_name(company or "")
    if len(core) >= 4:
        names.append(f'"{core}"')
    b = brand(website)
    if b and b not in core.replace(" ", ""):
        names.append(b)
    if not is_ambiguous(symbol) or not names:
        names.append(symbol.upper())
    who = names[0] if len(names) == 1 else f"({' OR '.join(names)})"
    return f"{who} ({' OR '.join(DISTRESS_TERMS)})"


def _http_get(url: str) -> str:
    import httpx

    r = httpx.get(url, headers={"User-Agent": "Mozilla/5.0 advisor"}, timeout=20)
    r.raise_for_status()
    return r.text


def parse(xml: str) -> list[dict]:
    """Raw items: title (publisher suffix removed), link, published, publisher and its URL."""
    out = []
    for item in ET.fromstring(xml).iter("item"):
        src = item.find("source")
        publisher = (src.text or "").strip() if src is not None else ""
        title = (item.findtext("title") or "").strip()
        if publisher and title.endswith(f" - {publisher}"):
            title = title[: -len(publisher) - 3].rstrip()
        try:
            published = parsedate_to_datetime(item.findtext("pubDate") or "")
        except (TypeError, ValueError):
            published = None
        out.append(
            {
                "title": title,
                "link": (item.findtext("link") or "").strip(),
                "published": published,
                "publisher": publisher,
                "publisher_url": src.get("url") if src is not None else None,
            }
        )
    return out


def search_news(
    symbol: str,
    query: str,
    *,
    company_name: str | None = None,
    website: str | None = None,
    days: int = 3,
    now: datetime | None = None,
    get: Callable[[str], str] = _http_get,
) -> list[SourceItem]:
    """Dated, entity-resolved items for ``symbol``, newest first. Never raises."""
    now = now or now_et()
    url = SEARCH_URL.format(q=quote(f"{query} when:{days}d"))
    try:
        rows = parse(get(url))
    except Exception as exc:  # noqa: BLE001
        logger.warning("google news: search failed for %s: %s", symbol, exc)
        return []

    own = domain(website)
    b = brand(website)
    out: list[SourceItem] = []
    undated = unresolved = 0
    for row in rows[:MAX_RESULTS]:
        published = row["published"]
        if published is None or published.tzinfo is None:
            undated += 1
            continue
        if published < now - timedelta(days=days + 1) or not row["title"] or not row["link"]:
            continue
        title = row["title"]
        entity = resolve_entity(symbol, text=title, company_name=company_name)
        if not entity.resolved and b and re.search(rf"(?<![a-z0-9]){b}(?![a-z0-9])", title.lower()):
            entity = EntityMatch(symbol=symbol.upper(), method=MatchMethod.COMPANY_NAME)
        if not entity.resolved:
            unresolved += 1
            continue
        pub_domain = domain(row["publisher_url"])
        issuer = bool(own) and pub_domain == own
        out.append(
            SourceItem(
                tier=SourceTier.PRIMARY if issuer else SourceTier.AGGREGATOR,
                provider=pub_domain or row["publisher"] or "news.google.com",
                url=row["link"],
                title=title,
                published_at=published,
                entity=entity,
                doc_type="COMPANY_STATEMENT" if issuer else "NEWS",
            )
        )
    if undated or unresolved:
        logger.info(
            "google news %s: dropped %d undated, %d not naming the company",
            symbol,
            undated,
            unresolved,
        )
    out.sort(key=lambda i: i.published_at, reverse=True)
    return out
