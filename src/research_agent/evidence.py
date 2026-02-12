"""Source registry, tier classification, and citation mapping."""

from __future__ import annotations

from datetime import datetime
from urllib.parse import urlparse

from research_agent.models import Source


# ── Tier classification ──────────────────────────────────────────────────────

_TIER_1_PATTERNS = [
    "sec.gov",
    "investor.",
    "ir.",
    ".gov",
]

_TIER_2_DOMAINS = {
    "reuters.com",
    "bloomberg.com",
    "wsj.com",
    "ft.com",
    "cnbc.com",
    "barrons.com",
    "marketwatch.com",
    "seekingalpha.com",
    "finance.yahoo.com",
    "morningstar.com",
}


def classify_tier(url: str) -> int:
    """Classify a URL into evidence tier (1=primary, 2=press, 3=other)."""
    try:
        parsed = urlparse(url)
        host = parsed.hostname or ""
    except Exception:
        return 3

    # Tier 1: primary sources (SEC filings, investor relations, government)
    for pattern in _TIER_1_PATTERNS:
        if pattern in host:
            return 1

    # Tier 2: reputable financial press
    for domain in _TIER_2_DOMAINS:
        if host.endswith(domain) or host == domain:
            return 2

    return 3


# ── Source registry ──────────────────────────────────────────────────────────


class SourceRegistry:
    """Tracks all sources across iterations, assigns IDs, deduplicates by URL."""

    def __init__(self) -> None:
        self._sources: dict[str, Source] = {}  # url -> Source
        self._id_map: dict[str, str] = {}  # url -> source_id (s1, s2, ...)
        self._counter = 0

    def add(self, url: str, title: str = "", publisher: str = "", snippet: str = "") -> str:
        """Add a source and return its ID (e.g. 's1'). Deduplicates by URL."""
        if url in self._id_map:
            return self._id_map[url]

        self._counter += 1
        source_id = f"s{self._counter}"
        tier = classify_tier(url)
        source = Source(
            url=url,
            title=title,
            publisher=publisher,
            tier=tier,
            snippet=snippet,
            accessed_at=datetime.now(),
        )
        self._sources[url] = source
        self._id_map[url] = source_id
        return source_id

    def get_id(self, url: str) -> str | None:
        return self._id_map.get(url)

    def get_source(self, source_id: str) -> Source | None:
        for url, sid in self._id_map.items():
            if sid == source_id:
                return self._sources[url]
        return None

    def all_sources(self) -> list[Source]:
        """Return all sources in registration order."""
        id_to_url = {sid: url for url, sid in self._id_map.items()}
        ordered = []
        for i in range(1, self._counter + 1):
            sid = f"s{i}"
            url = id_to_url.get(sid)
            if url and url in self._sources:
                ordered.append(self._sources[url])
        return ordered

    def source_id_for_citation(self, url: str) -> str:
        """Get or create a source ID for citation use."""
        if url in self._id_map:
            return self._id_map[url]
        return self.add(url)

    @property
    def count(self) -> int:
        return self._counter
