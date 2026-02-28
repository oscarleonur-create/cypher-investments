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

    def _call_api(self, query: str, domains: list[str] | None, max_results: int) -> dict:
        """Make a raw Perplexity Sonar API call."""
        self._rate_limit()
        payload: dict = {
            "model": self._config.perplexity_model,
            "messages": [{"role": "user", "content": query}],
            "search_recency_filter": "month",
        }
        if domains:
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
    ) -> list[SearchResult]:
        """Search Perplexity with optional domain filtering and caching.

        Uses curated-first strategy: tries curated domains first, then
        falls back to open web if allowed and needed.
        """
        # Check cache first
        cached = self._store.get_cached_search(query)
        if cached is not None:
            return self._parse_results(cached)

        # In offline mode, return empty if no cache hit — skip all API calls
        if self._config.offline_mode:
            return []

        results: list[SearchResult] = []

        # Curated-first strategy
        if self._config.curated_first and not domains:
            curated = self._config.curated_domain_list
            if curated:
                try:
                    data = self._call_api(query, curated, max_results)
                    results = self._parse_results(data)
                    self._store.cache_search(query, data)
                except httpx.HTTPError:
                    pass  # Fall through to open web

        # Fallback to open web (or explicit domain filter)
        if not results and (self._config.allow_fallback_web or domains):
            try:
                data = self._call_api(query, domains, max_results)
                results = self._parse_results(data)
                self._store.cache_search(query, data)
            except httpx.HTTPError:
                pass

        return results

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
