"""The body of a news article, when the feed gave only a headline and a teaser.

Measured 2026-09-27: 17 of 21 items the news agent judged on held names came
from yfinance with 0-500 characters — "Cerebras Shares Are Coming Out of Lockup
in Waves" was judged from its headline alone. The agent cannot size a lockup,
an order or a raise it cannot read. This fetches the publisher's page and keeps
its paragraphs, no more: no scripts, navigation, captions or footers.

Stdlib only. A page that blocks, times out or yields too little text is
recorded as unreadable, and the agent is told it is reading a teaser.
"""

from __future__ import annotations

import re
import sqlite3
from datetime import datetime
from html.parser import HTMLParser

from advisor.daemon.market_calendar import now_et

# Below this the feed's own text is too thin to judge on: fetch the page.
THIN_CHARS = 600
# What the model is shown of a page: enough for the facts, not the boilerplate.
ARTICLE_CHARS = 4000
# A page that yields less than this is a paywall, a consent wall or a stub.
MIN_BODY_CHARS = 300

_SKIP = {"script", "style", "noscript", "nav", "footer", "header", "aside", "form", "figcaption"}
_KEEP = {"p", "h1", "h2", "h3", "li"}


class _Paragraphs(HTMLParser):
    """Text of <p>/<h*>/<li> outside navigation and scripts, in page order."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.skip = 0
        self.keep = 0
        self.buf: list[str] = []
        self.out: list[str] = []

    def handle_starttag(self, tag, attrs):
        if tag in _SKIP:
            self.skip += 1
        elif tag in _KEEP and not self.skip:
            self.keep += 1

    def handle_endtag(self, tag):
        if tag in _SKIP and self.skip:
            self.skip -= 1
        elif tag in _KEEP and self.keep:
            self.keep -= 1
            if not self.keep:
                text = re.sub(r"\s+", " ", "".join(self.buf)).strip()
                if len(text) >= 40:  # menu items and bylines are short
                    self.out.append(text)
                self.buf = []

    def handle_data(self, data):
        if self.keep and not self.skip:
            self.buf.append(data)


def body_text(html: str) -> str:
    """The article's paragraphs, deduplicated, joined. "" when there are none. Pure."""
    parser = _Paragraphs()
    try:
        parser.feed(html)
    except Exception:  # noqa: BLE001 — a malformed page reads as unreadable
        return ""
    seen, kept = set(), []
    for p in parser.out:
        if p in seen:
            continue
        seen.add(p)
        kept.append(p)
    return "\n".join(kept)


def fetch_article(url: str, *, fetch=None) -> str | None:
    """The page's body text, capped; None when it cannot be read."""
    from advisor.news import verify

    fetch = fetch or verify.fetch_page
    if "news.google.com" in url:
        url = verify.resolve_google(url) or url
    html = fetch(url)
    if not html:
        return None
    text = body_text(html)
    return text[:ARTICLE_CHARS] if len(text) >= MIN_BODY_CHARS else None


_SCHEMA = """\
CREATE TABLE IF NOT EXISTS news_articles (
    url        TEXT NOT NULL PRIMARY KEY,
    body       TEXT,                     -- NULL: fetched and unreadable
    fetched_at TEXT NOT NULL
);
"""


class ArticleCache:
    """One fetch per URL: a re-judgment or a retry never downloads again."""

    def __init__(self, conn: sqlite3.Connection) -> None:
        self._conn = conn
        conn.executescript(_SCHEMA)

    def get(self, url: str) -> tuple[bool, str | None]:
        """(known, body). known=False: never fetched."""
        row = self._conn.execute("SELECT body FROM news_articles WHERE url = ?", (url,)).fetchone()
        return (row is not None, row[0] if row else None)

    def put(self, url: str, body: str | None, when: datetime | None = None) -> None:
        self._conn.execute(
            "INSERT OR REPLACE INTO news_articles (url, body, fetched_at) VALUES (?, ?, ?)",
            (url, body, (when or now_et()).isoformat()),
        )
        self._conn.commit()


def text_for(item, cache: ArticleCache | None, *, fetch=None) -> tuple[str, str]:
    """(text the model reads, how it was read): "article", "feed" or "headline".

    The feed's own text is kept when it is long enough; otherwise the page is
    fetched once and cached. A filing's lead is the company's own words and is
    never replaced.
    """
    summary = (item.summary or "").strip()
    if item.tier.value == "PRIMARY" or len(summary) >= THIN_CHARS or not item.url:
        return summary, ("feed" if summary else "headline")
    body = None
    if cache is not None:
        known, body = cache.get(item.url)
        if not known:
            body = fetch_article(item.url, fetch=fetch)
            cache.put(item.url, body)
    else:
        body = fetch_article(item.url, fetch=fetch)
    if body and len(body) > len(summary):
        return body, "article"
    return summary, ("feed" if summary else "headline")
