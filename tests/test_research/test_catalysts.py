"""Tests for the catalyst calendar builder (earnings + news sources)."""

from __future__ import annotations

from datetime import date, timedelta
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from advisor.research.catalysts import (
    _earnings_catalysts,
    _news_catalysts,
    _resolve_expected_date,
    build_catalysts,
)
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
    future_date = (date.today() + timedelta(days=30)).isoformat()
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
                    "expected_date": future_date,
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


def test_news_catalysts_drops_stale_llm_dates_regardless_of_llm_claim():
    """The bug this guards against: an LLM reading an old article about an
    event that was 'upcoming' when written can still claim is_near_term=True
    for a date that has since passed. The code must override that, not trust
    it — this was reproduced live against a real AAPL search+LLM call before
    the fix (a 2026-04-30 earnings date got marked near-term from an August
    vantage point)."""
    past_date = (date.today() - timedelta(days=10)).isoformat()
    with (
        patch("research_agent.config.ResearchConfig") as mock_cfg,
        patch("research_agent.store.Store"),
        patch("research_agent.search.TavilyClient") as mock_tavily_cls,
        patch("research_agent.llm.OpenRouterLLM") as mock_llm_cls,
    ):
        mock_cfg.return_value = MagicMock(tavily_api_key="key", db_path=MagicMock())
        mock_tavily_cls.return_value.search.return_value = [
            SimpleNamespace(title="Old article", url="https://x.com", content="stale"),
        ]
        mock_llm_cls.return_value.complete.return_value = SimpleNamespace(
            catalysts=[
                {
                    "description": "Earnings release that already happened",
                    "catalyst_type": "EARNINGS",
                    "expected_date": past_date,
                    "is_near_term": True,  # LLM wrongly claims this is upcoming
                    "probability": 1.0,
                    "direction": "mixed",
                    "price_impact_pct": 7.0,
                },
            ]
        )

        items = _news_catalysts("AAPL", "Apple")

    assert items == []  # dropped entirely, not just relabeled


def test_news_catalysts_recomputes_is_near_term_from_real_date_math():
    """A future date the LLM marks is_near_term=False (or =True) should be
    recomputed from actual date math, not trusted verbatim either way."""
    far_future = (date.today() + timedelta(days=200)).isoformat()
    with (
        patch("research_agent.config.ResearchConfig") as mock_cfg,
        patch("research_agent.store.Store"),
        patch("research_agent.search.TavilyClient") as mock_tavily_cls,
        patch("research_agent.llm.OpenRouterLLM") as mock_llm_cls,
    ):
        mock_cfg.return_value = MagicMock(tavily_api_key="key", db_path=MagicMock())
        mock_tavily_cls.return_value.search.return_value = [
            SimpleNamespace(title="Article", url="https://x.com", content="text"),
        ]
        mock_llm_cls.return_value.complete.return_value = SimpleNamespace(
            catalysts=[
                {
                    "description": "Far-future event, LLM incorrectly says near-term",
                    "catalyst_type": "REGULATORY",
                    "expected_date": far_future,
                    "is_near_term": True,  # wrong -- 200 days out, code must correct this
                    "probability": 0.6,
                    "direction": "neutral",
                    "price_impact_pct": 3.0,
                },
            ]
        )

        items = _news_catalysts("AAPL", "Apple")

    assert len(items) == 1
    assert items[0].is_near_term is False


def test_news_catalysts_passes_through_unparseable_quarter_strings():
    """Quarter-label dates (e.g. 'Q3 FY2026') can't be verified against real
    date math -- best effort, so they pass through with the LLM's own
    is_near_term guess rather than being dropped outright."""
    with (
        patch("research_agent.config.ResearchConfig") as mock_cfg,
        patch("research_agent.store.Store"),
        patch("research_agent.search.TavilyClient") as mock_tavily_cls,
        patch("research_agent.llm.OpenRouterLLM") as mock_llm_cls,
    ):
        mock_cfg.return_value = MagicMock(tavily_api_key="key", db_path=MagicMock())
        mock_tavily_cls.return_value.search.return_value = [
            SimpleNamespace(title="Article", url="https://x.com", content="text"),
        ]
        mock_llm_cls.return_value.complete.return_value = SimpleNamespace(
            catalysts=[
                {
                    "description": "Product launch next quarter",
                    "catalyst_type": "PRODUCT_LAUNCH",
                    "expected_date": "Q3 FY2027",
                    "is_near_term": False,
                    "probability": 0.8,
                    "direction": "bullish",
                    "price_impact_pct": 6.0,
                },
            ]
        )

        items = _news_catalysts("AAPL", "Apple")

    assert len(items) == 1
    assert items[0].expected_date == "Q3 FY2027"
    assert items[0].is_near_term is False


def test_news_catalysts_swallows_unexpected_exceptions():
    with patch("research_agent.config.ResearchConfig", side_effect=RuntimeError("boom")):
        assert _news_catalysts("AAPL", "Apple") == []


# ── _resolve_expected_date (the date-anchoring guard itself) ────────────────


def test_resolve_expected_date_drops_past_dates():
    today = date(2026, 8, 23)
    # llm_is_near_term=True (the LLM's wrong claim) must not save a past date.
    assert _resolve_expected_date("2026-04-30", True, today) is None


def test_resolve_expected_date_keeps_and_recomputes_near_term_flag():
    today = date(2026, 8, 23)
    # Recomputed from real date math regardless of what the LLM claimed.
    assert _resolve_expected_date("2026-09-01", False, today) == ("2026-09-01", True)
    assert _resolve_expected_date("2027-01-01", True, today) == ("2027-01-01", False)


def test_resolve_expected_date_passes_through_non_iso_strings():
    today = date(2026, 8, 23)
    # Can't verify a quarter label against real date math -- trust the LLM's guess.
    assert _resolve_expected_date("Q3 FY2027", False, today) == ("Q3 FY2027", False)
    assert _resolve_expected_date("Q3 FY2027", True, today) == ("Q3 FY2027", True)


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
