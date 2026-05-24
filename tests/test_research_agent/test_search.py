"""Tests for research_agent.search (mocked HTTP, no network)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from research_agent.config import ResearchConfig
from research_agent.search import SearchOptions, TavilyClient
from research_agent.store import Store


def _make_config(**overrides) -> ResearchConfig:
    defaults = dict(
        _env_file=None,
        tavily_api_key="test-key",
        openrouter_api_key="test-key",
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

    @patch("research_agent.search.httpx.post")
    def test_offline_mode_skips_api_on_cache_miss(self, mock_post, tmp_path):
        """In offline mode, return empty list when no cache hit — no API call."""
        config = _make_config(offline_mode=True)
        store = Store(tmp_path / "test.db")
        try:
            client = TavilyClient(config, store)
            results = client.search("AAPL earnings")

            assert results == []
            mock_post.assert_not_called()
        finally:
            store.close()

    def test_offline_mode_returns_cached_results(self, tmp_path):
        """In offline mode, cached results are still returned."""
        config = _make_config(offline_mode=True)
        store = Store(tmp_path / "test.db")
        try:
            cached_data = _mock_tavily_response()
            store.cache_search("AAPL stock", cached_data)

            client = TavilyClient(config, store)
            results = client.search("AAPL stock")

            assert len(results) == 1
            assert results[0].title == "Test Article"
        finally:
            store.close()

    def test_parse_results_empty(self):
        results = TavilyClient._parse_results({"results": []})
        assert results == []

    def test_parse_results_filters_empty_urls(self):
        results = TavilyClient._parse_results(
            {
                "results": [
                    {"url": "", "title": "No URL", "content": "test"},
                    {"url": "https://example.com", "title": "Has URL", "content": "test"},
                ]
            }
        )
        assert len(results) == 1
        assert results[0].url == "https://example.com"

    @patch("research_agent.search.httpx.post")
    def test_api_sends_api_key_in_body(self, mock_post, tmp_path):
        """API calls include api_key in request body."""
        config = _make_config(tavily_api_key="tvly-test-key")
        store = Store(tmp_path / "test.db")
        try:
            mock_resp = MagicMock()
            mock_resp.json.return_value = _mock_tavily_response()
            mock_resp.raise_for_status = MagicMock()
            mock_post.return_value = mock_resp

            client = TavilyClient(config, store)
            client.search("AAPL stock")

            call_args = mock_post.call_args
            payload = call_args.kwargs.get("json") or call_args[1].get("json")
            assert payload["api_key"] == "tvly-test-key"
        finally:
            store.close()

    @patch("research_agent.search.httpx.post")
    def test_search_with_sec_mode(self, mock_post, tmp_path):
        """Passing search_mode='sec' uses include_domains for sec.gov."""
        config = _make_config()
        store = Store(tmp_path / "test.db")
        try:
            mock_resp = MagicMock()
            mock_resp.json.return_value = _mock_tavily_response()
            mock_resp.raise_for_status = MagicMock()
            mock_post.return_value = mock_resp

            client = TavilyClient(config, store)
            options = SearchOptions(search_mode="sec")
            client.search("AAPL 10-K", options=options)

            payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
            assert payload["include_domains"] == ["sec.gov"]
        finally:
            store.close()

    @patch("research_agent.search.httpx.post")
    def test_search_sec_convenience(self, mock_post, tmp_path):
        """search_sec() convenience method filters to sec.gov domain."""
        config = _make_config()
        store = Store(tmp_path / "test.db")
        try:
            mock_resp = MagicMock()
            mock_resp.json.return_value = _mock_tavily_response()
            mock_resp.raise_for_status = MagicMock()
            mock_post.return_value = mock_resp

            client = TavilyClient(config, store)
            client.search_sec("AAPL 10-K revenue")

            payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
            assert payload["include_domains"] == ["sec.gov"]
        finally:
            store.close()

    @patch("research_agent.search.httpx.post")
    def test_cache_key_differs_by_mode(self, mock_post, tmp_path):
        """Same query with different search_mode produces separate cache entries."""
        config = _make_config()
        store = Store(tmp_path / "test.db")
        try:
            mock_resp = MagicMock()
            mock_resp.json.return_value = _mock_tavily_response()
            mock_resp.raise_for_status = MagicMock()
            mock_post.return_value = mock_resp

            client = TavilyClient(config, store)

            # First call: web search
            client.search("AAPL earnings")
            # Second call: SEC search
            client.search("AAPL earnings", options=SearchOptions(search_mode="sec"))

            # Both should hit the API (different cache keys)
            assert mock_post.call_count == 2
        finally:
            store.close()

    @patch("research_agent.search.httpx.post")
    def test_options_none_uses_days_for_recency(self, mock_post, tmp_path):
        """Passing options=None sends days parameter based on recency config."""
        config = _make_config()
        store = Store(tmp_path / "test.db")
        try:
            mock_resp = MagicMock()
            mock_resp.json.return_value = _mock_tavily_response()
            mock_resp.raise_for_status = MagicMock()
            mock_post.return_value = mock_resp

            client = TavilyClient(config, store)
            client.search("AAPL earnings")

            payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
            assert payload["days"] == 30  # "month" -> 30 days
        finally:
            store.close()

    @patch("research_agent.search.httpx.post")
    def test_recency_filter_from_config(self, mock_post, tmp_path):
        """search_recency_filter is mapped to days parameter."""
        config = _make_config(search_recency_filter="week")
        store = Store(tmp_path / "test.db")
        try:
            mock_resp = MagicMock()
            mock_resp.json.return_value = _mock_tavily_response()
            mock_resp.raise_for_status = MagicMock()
            mock_post.return_value = mock_resp

            client = TavilyClient(config, store)
            client.search("AAPL earnings")

            payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
            assert payload["days"] == 7  # "week" -> 7 days
        finally:
            store.close()

    @patch("research_agent.search.httpx.post")
    def test_sec_mode_skips_curated_first(self, mock_post, tmp_path):
        """With curated_first=True and search_mode='sec', curated-first is bypassed."""
        config = _make_config(curated_first=True)
        store = Store(tmp_path / "test.db")
        try:
            mock_resp = MagicMock()
            mock_resp.json.return_value = _mock_tavily_response()
            mock_resp.raise_for_status = MagicMock()
            mock_post.return_value = mock_resp

            client = TavilyClient(config, store)
            options = SearchOptions(search_mode="sec")
            client.search("AAPL 10-K", options=options)

            # Should only call API once (no curated-first attempt)
            assert mock_post.call_count == 1
            payload = mock_post.call_args.kwargs.get("json") or mock_post.call_args[1].get("json")
            assert payload["include_domains"] == ["sec.gov"]
        finally:
            store.close()
