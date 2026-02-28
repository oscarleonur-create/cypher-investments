"""Perplexity Sonar search client with caching and curated-first strategy."""

from __future__ import annotations

import time
from dataclasses import dataclass

import httpx

from research_agent.config import ResearchConfig
from research_agent.store import Store


@dataclass
class SearchResult:
    """A single result from a Perplexity search."""

    url: str
    title: str
    content: str
    score: float = 0.0


@dataclass
class SearchOptions:
    """Per-call search parameters for Perplexity API."""

    search_mode: str | None = None
    search_after_date_filter: str | None = None
    search_recency_filter: str | None = None


class PerplexityClient:
    """Perplexity Sonar API client with caching, rate limiting, and curated-first policy."""

    def __init__(self, config: ResearchConfig, store: Store) -> None:
        self._config = config
        self._store = store
        self._last_request_time: float = 0.0

    def _rate_limit(self) -> None:
        """Enforce 2 requests/second via simple sleep."""
        elapsed = time.time() - self._last_request_time
        if elapsed < 0.5:
            time.sleep(0.5 - elapsed)
        self._last_request_time = time.time()

    def _build_cache_key(self, query: str, options: SearchOptions | None) -> str:
        """Build a cache key incorporating query and search options."""
        parts = [query]
        if options:
            if options.search_mode:
                parts.append(f"mode:{options.search_mode}")
            if options.search_after_date_filter:
                parts.append(f"after:{options.search_after_date_filter}")
            if options.search_recency_filter:
                parts.append(f"recency:{options.search_recency_filter}")
        return "|".join(parts)

    def _call_api(
        self,
        query: str,
        domains: list[str] | None,
        max_results: int,
        options: SearchOptions | None = None,
    ) -> dict:
        """Make a raw Perplexity Sonar API call."""
        self._rate_limit()
        payload: dict = {
            "model": self._config.perplexity_model,
            "messages": [{"role": "user", "content": query}],
        }

        recency = (
            options.search_recency_filter
            if options and options.search_recency_filter
            else self._config.search_recency_filter
        )
        if recency:
            payload["search_recency_filter"] = recency

        search_mode = (
            options.search_mode
            if options and options.search_mode
            else self._config.default_search_mode
        )
        if search_mode:
            payload["search_mode"] = search_mode

        if options and options.search_after_date_filter:
            payload["search_after_date_filter"] = options.search_after_date_filter

        if domains and not search_mode:
            payload["search_domain_filter"] = domains
        headers = {
            "Authorization": f"Bearer {self._config.perplexity_api_key}",
            "Content-Type": "application/json",
        }
        resp = httpx.post(
            self._config.search_endpoint,
            json=payload,
            headers=headers,
            timeout=self._config.http_timeout_seconds,
        )
        resp.raise_for_status()
        return resp.json()

    def search(
        self,
        query: str,
        domains: list[str] | None = None,
        max_results: int = 5,
        options: SearchOptions | None = None,
    ) -> list[SearchResult]:
        """Search Perplexity with optional domain filtering and caching.

        Uses curated-first strategy: tries curated domains first, then
        falls back to open web if allowed and needed.  When a search_mode
        is active (e.g. ``"sec"``), curated-first is bypassed.
        """
        cache_key = self._build_cache_key(query, options)

        # Check cache first
        cached = self._store.get_cached_search(cache_key)
        if cached is not None:
            return self._parse_results(cached)

        # In offline mode, return empty if no cache hit — skip all API calls
        if self._config.offline_mode:
            return []

        results: list[SearchResult] = []
        has_search_mode = (options and options.search_mode) or self._config.default_search_mode

        # Curated-first strategy (skip when search_mode is active)
        if self._config.curated_first and not domains and not has_search_mode:
            curated = self._config.curated_domain_list
            if curated:
                try:
                    data = self._call_api(query, curated, max_results, options)
                    results = self._parse_results(data)
                    self._store.cache_search(cache_key, data)
                except httpx.HTTPError:
                    pass  # Fall through to open web

        # Fallback to open web (or explicit domain filter, or search_mode)
        if not results and (self._config.allow_fallback_web or domains or has_search_mode):
            try:
                data = self._call_api(query, domains, max_results, options)
                results = self._parse_results(data)
                self._store.cache_search(cache_key, data)
            except httpx.HTTPError:
                pass

        return results

    def search_sec(
        self,
        query: str,
        after_date: str | None = None,
        max_results: int = 5,
    ) -> list[SearchResult]:
        """Search SEC filings via Perplexity's SEC search mode.

        Args:
            query: Search query (e.g. "AAPL 10-K revenue segments 2024").
            after_date: Only return filings after this date (M/D/YYYY format).
            max_results: Maximum number of results.
        """
        options = SearchOptions(
            search_mode="sec",
            search_after_date_filter=after_date,
        )
        return self.search(query, max_results=max_results, options=options)

    @staticmethod
    def _parse_results(data: dict) -> list[SearchResult]:
        """Parse Perplexity Sonar response into SearchResult objects."""
        items = data.get("search_results", [])
        return [
            SearchResult(
                url=item.get("url", ""),
                title=item.get("title", ""),
                content=item.get("snippet", ""),
                score=item.get("score", 0.0),
            )
            for item in items
            if item.get("url")
        ]
