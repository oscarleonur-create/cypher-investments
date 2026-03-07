"""Tests for the smart money screener (Perplexity + Claude pipeline)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from advisor.confluence.smart_money_screener import (
    CongressScore,
    InsiderScore,
    _CongressTradesResponse,
    _fetch_congress_trades,
    _fetch_insider_activity,
    _InsiderTradesResponse,
)
from research_agent.search import SearchResult

_P = "advisor.confluence.smart_money_screener"


def _setup(mock_config_cls, mock_perplexity_cls, mock_llm_cls, mock_store_cls):
    """Wire mock classes and return (search_instance, llm_instance)."""
    cfg = MagicMock()
    cfg.db_path = MagicMock()
    mock_config_cls.return_value = cfg
    mock_store_cls.return_value = MagicMock()

    search = MagicMock()
    mock_perplexity_cls.return_value = search

    llm = MagicMock()
    mock_llm_cls.return_value = llm

    return search, llm


# ---------------------------------------------------------------------------
# 1. Insider activity tests
# ---------------------------------------------------------------------------


class TestFetchInsiderActivity:
    @patch(f"{_P}._cache_set")
    @patch(f"{_P}._cache_get", return_value=None)
    @patch(f"{_P}.Store")
    @patch(f"{_P}.ClaudeLLM")
    @patch(f"{_P}.PerplexityClient")
    @patch(f"{_P}.ResearchConfig")
    def test_insider_cluster_buys_scored(
        self, mock_cfg, mock_px, mock_llm_cls, mock_store, _cg, _cs
    ):
        """3 Purchase trades -> cluster_buys=3, score includes +13."""
        search, llm = _setup(mock_cfg, mock_px, mock_llm_cls, mock_store)

        search.search_sec.return_value = [
            SearchResult(
                url="https://sec.gov/1",
                title="Form 4",
                content=(
                    "Filing Date: 2026-02-01 Alice VP Purchase 100 shares at $50.00\n"
                    "Filing Date: 2026-02-05 Bob VP Purchase 200 shares at $51.00\n"
                    "Filing Date: 2026-02-10 Carol VP Purchase 150 shares at $52.00"
                ),
            ),
        ]
        llm.complete.return_value = _InsiderTradesResponse(
            trades=[
                {
                    "filing_date": "2026-02-01",
                    "insider_name": "Alice",
                    "title": "VP",
                    "trade_type": "Purchase",
                    "price": 50.0,
                    "qty": 100,
                    "value": 5000.0,
                },
                {
                    "filing_date": "2026-02-05",
                    "insider_name": "Bob",
                    "title": "VP",
                    "trade_type": "Purchase",
                    "price": 51.0,
                    "qty": 200,
                    "value": 10200.0,
                },
                {
                    "filing_date": "2026-02-10",
                    "insider_name": "Carol",
                    "title": "VP",
                    "trade_type": "Purchase",
                    "price": 52.0,
                    "qty": 150,
                    "value": 7800.0,
                },
            ]
        )

        result = _fetch_insider_activity("AAPL")

        assert isinstance(result, InsiderScore)
        assert result.cluster_buys == 3
        # cluster >= 3 -> +13, plus +4 base for any buys = at least 17
        assert result.score >= 17.0
        assert len(result.buy_trades) == 3
        assert len(result.sell_trades) == 0

    @patch(f"{_P}._cache_set")
    @patch(f"{_P}._cache_get", return_value=None)
    @patch(f"{_P}.Store")
    @patch(f"{_P}.ClaudeLLM")
    @patch(f"{_P}.PerplexityClient")
    @patch(f"{_P}.ResearchConfig")
    def test_insider_csuite_large_buy_bonus(
        self, mock_cfg, mock_px, mock_llm_cls, mock_store, _cg, _cs
    ):
        """CEO purchase over $100K -> csuite + large buy bonuses."""
        search, llm = _setup(mock_cfg, mock_px, mock_llm_cls, mock_store)

        search.search_sec.return_value = [
            SearchResult(
                url="https://sec.gov/1",
                title="Form 4",
                content=(
                    "Filing Date: 2026-02-15 John CEO Chief Executive Officer"
                    " Purchase 1000 shares at $150.00 value $150,000"
                ),
            ),
        ]
        llm.complete.return_value = _InsiderTradesResponse(
            trades=[
                {
                    "filing_date": "2026-02-15",
                    "insider_name": "John CEO",
                    "title": "CEO",
                    "trade_type": "Purchase",
                    "price": 150.0,
                    "qty": 1000,
                    "value": 150000.0,
                },
            ]
        )

        result = _fetch_insider_activity("AAPL")

        assert result.csuite_buys == 1
        assert result.large_buys == 1
        # +4 (any buy) + 4 (large) + 4 (csuite) = 12
        assert result.score >= 12.0

    @patch(f"{_P}._cache_set")
    @patch(f"{_P}._cache_get", return_value=None)
    @patch(f"{_P}.Store")
    @patch(f"{_P}.ClaudeLLM")
    @patch(f"{_P}.PerplexityClient")
    @patch(f"{_P}.ResearchConfig")
    def test_insider_sells_negative(self, mock_cfg, mock_px, mock_llm_cls, mock_store, _cg, _cs):
        """5 Sale trades -> negative score."""
        search, llm = _setup(mock_cfg, mock_px, mock_llm_cls, mock_store)

        search.search_sec.return_value = [
            SearchResult(
                url="https://sec.gov/1",
                title="Form 4",
                content=(
                    "Filing Date: 2026-02-01\n"
                    "Seller0 VP Sale 500 shares at $100.00\n"
                    "Seller1 VP Sale 500 shares at $100.00\n"
                    "Seller2 VP Sale 500 shares at $100.00\n"
                    "Seller3 VP Sale 500 shares at $100.00\n"
                    "Seller4 VP Sale 500 shares at $100.00"
                ),
            ),
        ]
        llm.complete.return_value = _InsiderTradesResponse(
            trades=[
                {
                    "filing_date": "2026-02-01",
                    "insider_name": f"Seller{i}",
                    "title": "VP",
                    "trade_type": "Sale",
                    "price": 100.0,
                    "qty": 500,
                    "value": 50000.0,
                }
                for i in range(5)
            ]
        )

        result = _fetch_insider_activity("AAPL")

        assert result.cluster_sells == 5
        assert result.score < 0
        assert len(result.sell_trades) == 5

    @patch(f"{_P}._cache_set")
    @patch(f"{_P}._cache_get", return_value=None)
    @patch(f"{_P}.Store")
    @patch(f"{_P}.ClaudeLLM")
    @patch(f"{_P}.PerplexityClient")
    @patch(f"{_P}.ResearchConfig")
    def test_insider_no_results_empty(self, mock_cfg, mock_px, mock_llm_cls, mock_store, _cg, _cs):
        """Empty search results -> InsiderScore(score=0)."""
        search, llm = _setup(mock_cfg, mock_px, mock_llm_cls, mock_store)

        search.search_sec.return_value = []

        result = _fetch_insider_activity("AAPL")

        assert result.score == 0.0
        assert result.cluster_buys == 0
        llm.complete.assert_not_called()

    @patch(f"{_P}._cache_set")
    @patch(f"{_P}._cache_get", return_value=None)
    @patch(f"{_P}.Store")
    @patch(f"{_P}.ClaudeLLM")
    @patch(f"{_P}.PerplexityClient")
    @patch(f"{_P}.ResearchConfig")
    def test_insider_exception_returns_empty(
        self, mock_cfg, mock_px, mock_llm_cls, mock_store, _cg, mock_cache_set
    ):
        """RuntimeError during search -> empty InsiderScore cached."""
        search, llm = _setup(mock_cfg, mock_px, mock_llm_cls, mock_store)

        search.search_sec.side_effect = RuntimeError("API down")

        result = _fetch_insider_activity("AAPL")

        assert result.score == 0.0
        # Should have cached the empty result
        mock_cache_set.assert_called_once()


# ---------------------------------------------------------------------------
# 2. Congressional trading tests
# ---------------------------------------------------------------------------


class TestFetchCongressTrades:
    @patch(f"{_P}._cache_set")
    @patch(f"{_P}._cache_get", return_value=None)
    @patch(f"{_P}.Store")
    @patch(f"{_P}.ClaudeLLM")
    @patch(f"{_P}.PerplexityClient")
    @patch(f"{_P}.ResearchConfig")
    def test_congress_multiple_buys_scored(
        self, mock_cfg, mock_px, mock_llm_cls, mock_store, _cg, _cs
    ):
        """3 buys by 2 politicians -> score=20."""
        search, llm = _setup(mock_cfg, mock_px, mock_llm_cls, mock_store)

        search.search.return_value = [
            SearchResult(
                url="https://news.com/1",
                title="Congress trades",
                content=(
                    "Congressional Stock Trading Disclosures\n"
                    "Sen. Smith disclosed a Purchase of AAPL on 2026-01-15\n"
                    "Rep. Jones disclosed a Purchase of AAPL on 2026-02-01\n"
                    "Sen. Smith disclosed another Purchase of AAPL on 2026-02-10"
                ),
            ),
        ]
        llm.complete.return_value = _CongressTradesResponse(
            trades=[
                {
                    "transaction_date": "2026-01-15",
                    "transaction_type": "Purchase",
                    "politician": "Sen. Smith",
                },
                {
                    "transaction_date": "2026-02-01",
                    "transaction_type": "Purchase",
                    "politician": "Rep. Jones",
                },
                {
                    "transaction_date": "2026-02-10",
                    "transaction_type": "Purchase",
                    "politician": "Sen. Smith",
                },
            ]
        )

        result = _fetch_congress_trades("AAPL")

        assert isinstance(result, CongressScore)
        assert result.recent_buys == 3
        assert len(result.politicians) == 2
        # 3 buys -> 16, 2 politicians -> +4 = 20
        assert result.score == 20.0

    @patch(f"{_P}._cache_set")
    @patch(f"{_P}._cache_get", return_value=None)
    @patch(f"{_P}.Store")
    @patch(f"{_P}.ClaudeLLM")
    @patch(f"{_P}.PerplexityClient")
    @patch(f"{_P}.ResearchConfig")
    def test_congress_no_results_empty(self, mock_cfg, mock_px, mock_llm_cls, mock_store, _cg, _cs):
        """Empty search results -> CongressScore(score=0)."""
        search, llm = _setup(mock_cfg, mock_px, mock_llm_cls, mock_store)

        search.search.return_value = []

        result = _fetch_congress_trades("AAPL")

        assert result.score == 0.0
        assert result.recent_buys == 0
        llm.complete.assert_not_called()
