"""Tests for research_agent.search (mocked HTTP, no network)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from research_agent.config import ResearchConfig
from research_agent.search import TavilyClient
from research_agent.store import Store


def _make_config(**overrides) -> ResearchConfig:
    defaults = dict(
        _env_file=None,
        tavily_api_key="test-key",
        anthropic_api_key="test-key",
        curated_first=False,
        allow_fallback_web=True,
    )
    defaults.update(overrides)
    return ResearchConfig(**defaults)


def _mock_tavily_response(results=None):
    """Create a mock Tavily API response."""
    if results is None:
        results = [
            {
                "url": "https://reuters.com/article/1",
                "title": "Test Article",
                "content": "Test content about stock",
                "score": 0.95,
            }
        ]
    return {"results": results}


class TestTavilyClient:
    def test_search_with_cache_hit(self, tmp_path):
        """Cached results are returned without API call."""
        config = _make_config()
        store = Store(tmp_path / "test.db")
        try:
            # Pre-populate cache
            cached_data = _mock_tavily_response()
            store.cache_search("AAPL stock", cached_data)

            client = TavilyClient(config, store)
            results = client.search("AAPL stock")

            assert len(results) == 1
            assert results[0].url == "https://reuters.com/article/1"
            assert results[0].title == "Test Article"
        finally:
            store.close()

    @patch("research_agent.search.httpx.post")
    def test_search_calls_api(self, mock_post, tmp_path):
        """When no cache, search calls the Tavily API."""
        config = _make_config()
        store = Store(tmp_path / "test.db")
        try:
            mock_resp = MagicMock()
            mock_resp.json.return_value = _mock_tavily_response()
            mock_resp.raise_for_status = MagicMock()
            mock_post.return_value = mock_resp

            client = TavilyClient(config, store)
            results = client.search("AAPL earnings")

            assert len(results) == 1
            mock_post.assert_called_once()
        finally:
            store.close()

    @patch("research_agent.search.httpx.post")
    def test_search_caches_result(self, mock_post, tmp_path):
        """Results from API are cached for subsequent queries."""
        config = _make_config()
        store = Store(tmp_path / "test.db")
        try:
            mock_resp = MagicMock()
            mock_resp.json.return_value = _mock_tavily_response()
            mock_resp.raise_for_status = MagicMock()
            mock_post.return_value = mock_resp

            client = TavilyClient(config, store)
            client.search("AAPL earnings")

            # Second call should use cache
            cached = store.get_cached_search("AAPL earnings")
            assert cached is not None
        finally:
            store.close()

    @patch("research_agent.search.httpx.post")
    def test_curated_first_strategy(self, mock_post, tmp_path):
        """With curated_first, searches curated domains first."""
        config = _make_config(curated_first=True)
        store = Store(tmp_path / "test.db")
        try:
            mock_resp = MagicMock()
            mock_resp.json.return_value = _mock_tavily_response()
            mock_resp.raise_for_status = MagicMock()
            mock_post.return_value = mock_resp

            client = TavilyClient(config, store)
            client.search("AAPL stock")

            # Should have been called with include_domains
            call_args = mock_post.call_args
            payload = call_args.kwargs.get("json") or call_args[1].get("json")
            assert "include_domains" in payload
        finally:
            store.close()

    def test_parse_results_empty(self):
        results = TavilyClient._parse_results({"results": []})
        assert results == []

    def test_parse_results_filters_empty_urls(self):
        results = TavilyClient._parse_results({
            "results": [
                {"url": "", "title": "No URL", "content": "test"},
                {"url": "https://example.com", "title": "Has URL", "content": "test"},
            ]
        })
        assert len(results) == 1
        assert results[0].url == "https://example.com"
