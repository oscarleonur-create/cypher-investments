"""A news item's date, checked before the item may support any action.

The date a source claims is not the date the story happened. Measured
2026-09-27 against the publishers' own pages:

- Google News dated a TradingKey piece on Nebius 2026-09-26; the page says it
  was published 2026-06-06. A three-month-old story read as new.
- yfinance dated an IBD article 43 hours after the page did: IBD rewrites the
  same URL daily and yfinance reports the rewrite.
- Google gives evergreen pages (an investor-relations home page, a MarketBeat
  data page) a date of its own, 07:00Z, that no one published.

An entry is as grave as an exit (the holder, 2026-09-27): a stale story that
turns a reading AT_RISK blocks a good entry just as one can fake an exit. So
every item is checked when it is ingested, whatever it will be used for:

1. **The publisher's page.** Its own published date (JSON-LD
   ``datePublished``, ``article:published_time``, a ``<time>`` tag) is the
   publisher's word. Agreeing with the claim within ``AGREE_HOURS`` (or on
   the same New York day, when the page gives only a date) is CONFIRMED; a
   different date is CORRECTED, and the page's date is used.
2. **Widen the search when the page cannot say** (a 403, no date on the
   page, a Google link that cannot be resolved): the same story elsewhere.
   The items already gathered, then the exact title searched in Google News.
   Two or more independent outlets dating it within ``CORROBORATE_HOURS`` of
   each other is CORROBORATED, at the earliest of their dates.
3. Otherwise UNVERIFIED: kept as context, never a fact a decision rests on.

Only CONFIRMED, CORRECTED and CORROBORATED items reach a reading or a count
(``USABLE``).
"""

from __future__ import annotations

import html as htmllib
import json
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import StrEnum

from advisor.daemon import market_calendar as mc

logger = logging.getLogger(__name__)

AGREE_HOURS = 1.5
CORROBORATE_HOURS = 24
MIN_CORROBORATING_OUTLETS = 2
SAME_STORY = 0.8  # share of one title's words found in the other
MAX_RESOLVES = 15  # Google link resolutions per batch; its endpoint starts refusing past ~40

_UA = {
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}


class Status(StrEnum):
    CONFIRMED = "CONFIRMED"
    CORRECTED = "CORRECTED"
    CORROBORATED = "CORROBORATED"
    UNVERIFIED = "UNVERIFIED"


USABLE = frozenset({Status.CONFIRMED, Status.CORRECTED, Status.CORROBORATED})


@dataclass(frozen=True)
class Verification:
    status: Status
    published_at: datetime | None  # the date to use; None when unverified
    how: str


# ---------------------------------------------------------------------------
# The publisher's page


_PUBLISHED = (
    r'"datePublished"\s*:\s*"([^"]+)"',
    r'<meta[^>]+property=["\']article:published_time["\'][^>]+content=["\']([^"\']+)',
    r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']article:published_time',
    r'<meta[^>]+name=["\'](?:pubdate|publishdate|publish-date|parsely-pub-date|'
    r'sailthru\.date|article\.published)["\'][^>]+content=["\']([^"\']+)',
    r'<time[^>]+datetime=["\']([^"\']+)',
)


def page_date(html: str) -> tuple[datetime, bool] | None:
    """(published, date_only) as the page itself states it; None when it does not. Pure."""
    for pattern in _PUBLISHED:
        m = re.search(pattern, html or "", re.IGNORECASE)
        if not m:
            continue
        raw = htmllib.unescape(m.group(1)).strip()
        try:
            d = datetime.fromisoformat(raw.replace("Z", "+00:00"))
        except ValueError:
            continue
        date_only = len(raw) <= 10
        if d.tzinfo is None:
            # A date alone is a New York day; a time without a zone is not trusted.
            if not date_only:
                continue
            d = d.replace(hour=12, tzinfo=mc.MARKET_TZ)
        return d, date_only
    return None


def check_page(claimed: datetime, html: str) -> Verification | None:
    """The claim against the page's own date. None when the page states none. Pure."""
    found = page_date(html)
    if found is None:
        return None
    published, date_only = found
    if date_only:
        if published.astimezone(mc.MARKET_TZ).date() == claimed.astimezone(mc.MARKET_TZ).date():
            return Verification(Status.CONFIRMED, claimed, "page date (day only) agrees")
        return Verification(Status.CORRECTED, published, "page gives a different day")
    if abs((claimed - published).total_seconds()) <= AGREE_HOURS * 3600:
        return Verification(Status.CONFIRMED, published, "page date agrees")
    return Verification(Status.CORRECTED, published, "page gives a different date")


def fetch_page(url: str) -> str | None:
    import httpx

    try:
        r = httpx.get(url, headers=_UA, timeout=20, follow_redirects=True)
    except Exception as exc:  # noqa: BLE001
        logger.info("verify: fetch failed for %s: %s", url, type(exc).__name__)
        return None
    return r.text if r.status_code < 400 else None


def resolve_google(link: str) -> str | None:
    """The publisher's URL behind a Google News link, or None.

    Google's own endpoint, undocumented: it refuses after a few dozen calls,
    so a failure here is ordinary and ends in corroboration, not an error.
    """
    import httpx

    try:
        art_id = link.split("/articles/")[1].split("?")[0]
        page = httpx.get(
            f"https://news.google.com/articles/{art_id}",
            headers=_UA,
            timeout=20,
            follow_redirects=True,
        )
        sg = re.search(r'data-n-a-sg="([^"]+)"', page.text)
        ts = re.search(r'data-n-a-ts="([^"]+)"', page.text)
        if not (sg and ts):
            return None
        inner = [
            "garturlreq",
            [
                ["X", "X", ["X", "X"], None, None, 1, 1, "US:en", None, 1]
                + [None, None, None, None, None, 0, 1],
                "X",
                "X",
                1,
                [1, 1, 1],
                1,
                1,
                None,
                0,
                0,
                None,
                0,
            ],
            art_id,
            int(ts.group(1)),
            sg.group(1),
        ]
        r = httpx.post(
            "https://news.google.com/_/DotsSplashUi/data/batchexecute",
            data={"f.req": json.dumps([[["Fbv4je", json.dumps(inner)]]])},
            headers={**_UA, "Content-Type": "application/x-www-form-urlencoded;charset=UTF-8"},
            timeout=20,
        )
        m = re.search(r'\[\\"garturlres\\",\\"(.*?)\\"', r.text)
        return m.group(1) if m else None
    except Exception as exc:  # noqa: BLE001
        logger.info("verify: google link unresolved: %s", type(exc).__name__)
        return None


# ---------------------------------------------------------------------------
# The same story elsewhere


def _words(title: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", title.lower()))


def same_story(a: str, b: str) -> bool:
    """Two titles for one story: most of the shorter one's words are in the other. Pure."""
    wa, wb = _words(a), _words(b)
    if not wa or not wb:
        return False
    return len(wa & wb) / min(len(wa), len(wb)) >= SAME_STORY


def corroborate(item, others) -> Verification | None:
    """Independent outlets dating the same story alike. Pure over (item, others).

    ``others`` are items with ``title``, ``provider`` and ``published_at``. One
    claim per outlet (its earliest); the item's own outlet counts once.
    """
    from advisor.entry.distress import outlet

    claims: dict[str, datetime] = {outlet(item.provider): item.published_at}
    for o in others:
        if o is item or not same_story(item.title, o.title):
            continue
        name = outlet(o.provider)
        if name not in claims or o.published_at < claims[name]:
            claims[name] = o.published_at
    if len(claims) < MIN_CORROBORATING_OUTLETS:
        return None
    dates = sorted(claims.values())
    # The largest group of outlets whose dates sit within the window of each other.
    best: list[datetime] = []
    for i, start in enumerate(dates):
        group = [d for d in dates[i:] if d - start <= timedelta(hours=CORROBORATE_HOURS)]
        if len(group) > len(best):
            best = group
    if len(best) < MIN_CORROBORATING_OUTLETS:
        return None
    return Verification(
        Status.CORROBORATED, best[0], f"{len(best)} outlets date the same story alike"
    )


def title_search(title: str) -> list:
    """The exact title in Google News: other outlets' copies, with their claimed dates."""
    from urllib.parse import quote

    import httpx

    from advisor.news.google_news import SEARCH_URL, domain, parse

    try:
        xml = httpx.get(
            SEARCH_URL.format(q=quote(f'"{title[:180]}"')), headers=_UA, timeout=20
        ).text
        rows = parse(xml)
    except Exception as exc:  # noqa: BLE001
        logger.info("verify: title search failed: %s", type(exc).__name__)
        return []
    return [
        _Claim(r["title"], domain(r["publisher_url"]) or r["publisher"], r["published"])
        for r in rows
        if r["published"] is not None and r["title"]
    ]


@dataclass(frozen=True)
class _Claim:
    title: str
    provider: str
    published_at: datetime


# ---------------------------------------------------------------------------
# One item, one batch


def verify(
    item,
    *,
    peers=(),
    fetch: Callable[[str], str | None] | None = None,
    resolve: Callable[[str], str | None] | None = None,
    search: Callable[[str], list] | None = None,
) -> Verification:
    """The date this item may be used at, and how that was established. Never raises.

    The network steps default to this module's functions, looked up per call.
    """
    fetch = fetch or fetch_page
    resolve = resolve or resolve_google
    search = search or title_search
    url = item.url or ""
    if "news.google.com/" in url:
        url = resolve(url) or ""
    if url:
        html = fetch(url)
        if html:
            found = check_page(item.published_at, html)
            if found is not None:
                return found
    found = corroborate(item, list(peers))
    if found is not None:
        return found
    try:
        wider = search(item.title)
    except Exception as exc:  # noqa: BLE001
        logger.info("verify: widening failed: %s", type(exc).__name__)
        wider = []
    found = corroborate(item, [*peers, *wider])
    if found is not None:
        return found
    return Verification(Status.UNVERIFIED, None, "no page date and no second outlet dating it")


def verify_items(items, *, store=None, **kwargs) -> list:
    """Each item with its checked date, ``verified`` status and ``claimed_at``.

    An item already archived with a status is not checked again. Google link
    resolutions are capped at ``MAX_RESOLVES`` per batch and stop at the first
    refusal: past that, Google items go straight to corroboration.
    """
    resolves = {"left": MAX_RESOLVES}
    base_resolve = kwargs.pop("resolve", None) or resolve_google

    def resolve(link: str) -> str | None:
        if resolves["left"] <= 0:
            return None
        resolves["left"] -= 1
        found = base_resolve(link)
        if found is None:
            resolves["left"] = 0
        return found

    out = []
    for item in items:
        known = store.get_source_item(item.dedup_key()) if store is not None else None
        if known is not None and known.verified:
            out.append(known)
            continue
        v = verify(item, peers=items, resolve=resolve, **kwargs)
        out.append(
            item.model_copy(
                update={
                    "verified": v.status.value,
                    "claimed_at": item.published_at,
                    "published_at": v.published_at or item.published_at,
                }
            )
        )
    counts: dict[str, int] = {}
    for i in out:
        counts[i.verified] = counts.get(i.verified, 0) + 1
    if out:
        logger.info("verify: %s", counts)
    return out


def usable(payload: dict) -> bool:
    """Whether a stored news payload may support a decision."""
    return payload.get("verified") in {s.value for s in USABLE}


@dataclass(frozen=True)
class _Stored:
    """A stored news event, shaped like an item for ``verify``."""

    title: str
    provider: str
    url: str
    published_at: datetime


def backfill(store, since: datetime, **kwargs) -> dict[str, int]:
    """Check the dates of stored news never checked. Returns counts per status.

    News archived before verification existed carries no status, so no reading
    or count may use it until this has run over it; each event is checked once.
    """
    events = store.unverified_news(since)
    stored = []
    for e in events:
        p = e.payload or {}
        raw = p.get("published_at")
        try:
            published = datetime.fromisoformat(raw) if raw else e.ts
        except ValueError:
            published = e.ts
        stored.append(
            _Stored(
                str(p.get("title") or ""),
                str(p.get("provider") or ""),
                str(p.get("url") or ""),
                published,
            )
        )
    resolves = {"left": MAX_RESOLVES}
    base_resolve = kwargs.pop("resolve", None) or resolve_google

    def resolve(link: str) -> str | None:
        if resolves["left"] <= 0:
            return None
        resolves["left"] -= 1
        found = base_resolve(link)
        if found is None:
            resolves["left"] = 0
        return found

    counts: dict[str, int] = {}
    for e, item in zip(events, stored, strict=True):
        if not item.title:
            continue
        v = verify(item, peers=stored, resolve=resolve, **kwargs)
        store.set_news_verification(
            e.id, v.status.value, v.published_at or item.published_at, item.published_at
        )
        counts[v.status.value] = counts.get(v.status.value, 0) + 1
    if counts:
        logger.info("verify backfill: %s", counts)
    return counts
