"""Tests for the sentiment confluence agent."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from advisor.confluence.sentiment import _SentimentScore, check_sentiment
from research_agent.search import SearchResult


class TestCheckSentiment:
    @patch("advisor.confluence.sentiment.Store")
    @patch("advisor.confluence.sentiment.ClaudeLLM")
    @patch("advisor.confluence.sentiment.TavilyClient")
    @patch("advisor.confluence.sentiment.ResearchConfig")
    def test_bullish_sentiment(
        self, mock_config_cls, mock_tavily_cls, mock_llm_cls, mock_store_cls
    ):
        mock_config = MagicMock()
        mock_config.db_path = MagicMock()
        mock_config_cls.return_value = mock_config

        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store

        mock_tavily = MagicMock()
        mock_tavily.search.return_value = [
            SearchResult(
                url="https://reuters.com/article1", title="Good news", content="Stock soars"
            ),
        ]
        mock_tavily_cls.return_value = mock_tavily

        mock_llm = MagicMock()
        mock_llm.complete.return_value = _SentimentScore(
            score=85.0,
            positive_pct=80.0,
            key_headlines=["Stock hits all-time high"],
            reasoning="Very positive",
        )
        mock_llm_cls.return_value = mock_llm

        result = check_sentiment("AAPL")

        assert result.is_bullish is True
        assert result.score == 85.0
        assert result.positive_pct == 80.0
        assert len(result.key_headlines) == 1
        # Sources should be populated from the registry
        assert len(result.sources) >= 1
        assert result.sources[0].url == "https://reuters.com/article1"

    @patch("advisor.confluence.sentiment.Store")
    @patch("advisor.confluence.sentiment.ClaudeLLM")
    @patch("advisor.confluence.sentiment.TavilyClient")
    @patch("advisor.confluence.sentiment.ResearchConfig")
    def test_bullish_llm_receives_formatted_context(
        self, mock_config_cls, mock_tavily_cls, mock_llm_cls, mock_store_cls
    ):
        """Verify the LLM receives [sN]-formatted search results."""
        mock_config = MagicMock()
        mock_config.db_path = MagicMock()
        mock_config_cls.return_value = mock_config

        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store

        mock_tavily = MagicMock()
        mock_tavily.search.return_value = [
            SearchResult(url="https://reuters.com/a", title="Article A", content="Content A"),
            SearchResult(url="https://wsj.com/b", title="Article B", content="Content B"),
        ]
        mock_tavily_cls.return_value = mock_tavily

        mock_llm = MagicMock()
        mock_llm.complete.return_value = _SentimentScore(
            score=60.0,
            positive_pct=55.0,
            key_headlines=[],
            reasoning="Mixed",
        )
        mock_llm_cls.return_value = mock_llm

        check_sentiment("AAPL")

        # Verify LLM was called with [s1]/[s2] formatted context
        call_args = mock_llm.complete.call_args
        user_prompt = call_args.kwargs.get("user_prompt") or call_args[1].get("user_prompt", "")
        if not user_prompt:
            # positional args: system_prompt, user_prompt, response_model
            user_prompt = call_args[0][1] if len(call_args[0]) > 1 else ""
        assert "[s1]" in user_prompt
        assert "[s2]" in user_prompt

    @patch("advisor.confluence.sentiment.Store")
    @patch("advisor.confluence.sentiment.ClaudeLLM")
    @patch("advisor.confluence.sentiment.TavilyClient")
    @patch("advisor.confluence.sentiment.ResearchConfig")
    def test_bearish_sentiment(
        self, mock_config_cls, mock_tavily_cls, mock_llm_cls, mock_store_cls
    ):
        mock_config = MagicMock()
        mock_config.db_path = MagicMock()
        mock_config_cls.return_value = mock_config

        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store

        mock_tavily = MagicMock()
        mock_tavily.search.return_value = [
            SearchResult(url="https://example.com", title="Bad news", content="Stock drops"),
        ]
        mock_tavily_cls.return_value = mock_tavily

        mock_llm = MagicMock()
        mock_llm.complete.return_value = _SentimentScore(
            score=30.0,
            positive_pct=20.0,
            key_headlines=["Stock plummets on earnings miss"],
            reasoning="Very negative",
        )
        mock_llm_cls.return_value = mock_llm

        result = check_sentiment("AAPL")

        assert result.is_bullish is False
        assert result.score == 30.0
        assert result.positive_pct == 20.0

    @patch("advisor.confluence.sentiment.Store")
    @patch("advisor.confluence.sentiment.ClaudeLLM")
    @patch("advisor.confluence.sentiment.TavilyClient")
    @patch("advisor.confluence.sentiment.ResearchConfig")
    def test_no_search_results_returns_neutral(
        self, mock_config_cls, mock_tavily_cls, mock_llm_cls, mock_store_cls
    ):
        mock_config = MagicMock()
        mock_config.db_path = MagicMock()
        mock_config_cls.return_value = mock_config

        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store

        mock_tavily = MagicMock()
        mock_tavily.search.return_value = []
        mock_tavily_cls.return_value = mock_tavily

        result = check_sentiment("AAPL")

        assert result.is_bullish is False
        assert result.score == 50.0
        assert result.positive_pct == 50.0
        assert result.sources == []

    @patch("advisor.confluence.sentiment.Store")
    @patch("advisor.confluence.sentiment.ClaudeLLM")
    @patch("advisor.confluence.sentiment.TavilyClient")
    @patch("advisor.confluence.sentiment.ResearchConfig")
    def test_exception_returns_neutral(
        self, mock_config_cls, mock_tavily_cls, mock_llm_cls, mock_store_cls
    ):
        mock_config = MagicMock()
        mock_config.db_path = MagicMock()
        mock_config_cls.return_value = mock_config

        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store

        mock_tavily = MagicMock()
        mock_tavily.search.side_effect = RuntimeError("API error")
        mock_tavily_cls.return_value = mock_tavily

        result = check_sentiment("AAPL")

        assert result.is_bullish is False
        assert result.score == 50.0
        assert result.sources == []

    @patch("advisor.confluence.sentiment.Store")
    @patch("advisor.confluence.sentiment.ClaudeLLM")
    @patch("advisor.confluence.sentiment.TavilyClient")
    @patch("advisor.confluence.sentiment.ResearchConfig")
    def test_source_tiers_populated(
        self, mock_config_cls, mock_tavily_cls, mock_llm_cls, mock_store_cls
    ):
        """Sources from tier-2 domains should get tier=2 classification."""
        mock_config = MagicMock()
        mock_config.db_path = MagicMock()
        mock_config_cls.return_value = mock_config

        mock_store = MagicMock()
        mock_store_cls.return_value = mock_store

        mock_tavily = MagicMock()
        mock_tavily.search.return_value = [
            SearchResult(
                url="https://reuters.com/news/1", title="Reuters article", content="Content"
            ),
            SearchResult(url="https://randomsite.com/blog", title="Blog post", content="Opinion"),
        ]
        mock_tavily_cls.return_value = mock_tavily

        mock_llm = MagicMock()
        mock_llm.complete.return_value = _SentimentScore(
            score=70.0,
            positive_pct=65.0,
            key_headlines=[],
            reasoning="Moderate",
        )
        mock_llm_cls.return_value = mock_llm

        result = check_sentiment("AAPL")

        # reuters.com is tier 2, randomsite.com is tier 3
        tiers = {s.url: s.tier for s in result.sources}
        assert tiers["https://reuters.com/news/1"] == 2
        assert tiers["https://randomsite.com/blog"] == 3
