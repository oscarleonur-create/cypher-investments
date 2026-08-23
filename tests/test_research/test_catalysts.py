"""Tests for the catalyst calendar builder (earnings + news sources)."""

from __future__ import annotations

from datetime import date, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from advisor.research.catalysts import _earnings_catalysts, _news_catalysts, build_catalysts
from advisor.research.models import CatalystItem, CatalystType

# ── Earnings (yfinance via advisor.data.yahoo.fetch_earnings_dates) ─────────


def test_earnings_catalysts_marks_near_term_within_90_days():
    today = date.today()
    near = today + timedelta(days=30)
    far = today + timedelta(days=200)

    with patch("advisor.data.yahoo.fetch_earnings_dates", return_value=[near, far]):
        items = _earnings_catalysts("AAPL")

    assert len(items) == 2
    by_date = {i.expected_date: i for i in items}
    assert by_date[near.isoformat()].is_near_term is True
    assert by_date[far.isoformat()].is_near_term is False
    for item in items:
        assert item.catalyst_type == CatalystType.EARNINGS
        assert item.source == "yfinance"
        assert item.probability == 0.5
        assert item.direction == "mixed"


def test_earnings_catalysts_empty_when_no_dates():
    with patch("advisor.data.yahoo.fetch_earnings_dates", return_value=[]):
        assert _earnings_catalysts("AAPL") == []


# ── News (Tavily search + LLM extraction) ────────────────────────────────────


def test_news_catalysts_returns_empty_without_tavily_key():
    with patch("research_agent.config.ResearchConfig") as mock_cfg:
        mock_cfg.return_value = MagicMock(tavily_api_key="")
        assert _news_catalysts("AAPL", "Apple") == []


def test_news_catalysts_returns_empty_on_no_search_results():
    with (
        patch("research_agent.config.ResearchConfig") as mock_cfg,
        patch("research_agent.store.Store"),
        patch("research_agent.search.TavilyClient") as mock_tavily_cls,
        patch("research_agent.llm.OpenRouterLLM"),
    ):
        mock_cfg.return_value = MagicMock(tavily_api_key="key", db_path=MagicMock())
        mock_tavily_cls.return_value.search.return_value = []

        assert _news_catalysts("AAPL", "Apple") == []


def test_news_catalysts_parses_llm_response():
    with (
        patch("research_agent.config.ResearchConfig") as mock_cfg,
        patch("research_agent.store.Store"),
        patch("research_agent.search.TavilyClient") as mock_tavily_cls,
        patch("research_agent.llm.OpenRouterLLM") as mock_llm_cls,
    ):
        mock_cfg.return_value = MagicMock(tavily_api_key="key", db_path=MagicMock())
        mock_tavily_cls.return_value.search.return_value = [
            SimpleNamespace(title="Apple news", url="https://x.com", content="launch coming"),
        ]
        mock_llm_cls.return_value.complete.return_value = SimpleNamespace(
            catalysts=[
                {
                    "description": "New product launch expected",
                    "catalyst_type": "PRODUCT_LAUNCH",
                    "expected_date": "2026-11-01",
                    "is_near_term": True,
                    "probability": 0.7,
                    "direction": "bullish",
                    "price_impact_pct": 5.0,
                },
                {
                    # Missing/invalid catalyst_type should be dropped, not crash.
                    "description": "bad row",
                    "catalyst_type": "NOT_A_REAL_TYPE",
                },
            ]
        )

        items = _news_catalysts("AAPL", "Apple")

    assert len(items) == 1
    item = items[0]
    assert item.description == "New product launch expected"
    assert item.catalyst_type == CatalystType.PRODUCT_LAUNCH
    assert item.source == "tavily"
    assert item.probability == 0.7
    assert item.direction == "bullish"
    assert item.price_impact_pct == 5.0


def test_news_catalysts_swallows_unexpected_exceptions():
    with patch("research_agent.config.ResearchConfig", side_effect=RuntimeError("boom")):
        assert _news_catalysts("AAPL", "Apple") == []


# ── build_catalysts (combines both sources) ──────────────────────────────────


def test_build_catalysts_combines_earnings_and_news():
    earnings_item = CatalystItem(description="earnings", catalyst_type=CatalystType.EARNINGS)
    news_item = CatalystItem(description="news", catalyst_type=CatalystType.PRODUCT_LAUNCH)

    with (
        patch("advisor.research.catalysts._earnings_catalysts", return_value=[earnings_item]),
        patch("advisor.research.catalysts._news_catalysts", return_value=[news_item]),
    ):
        result = build_catalysts("aapl", "Apple")

    assert result.symbol == "AAPL"
    assert result.catalysts == [earnings_item, news_item]
