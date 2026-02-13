"""Tests for research_agent.queries (mode-dependent search queries and labels)."""

from __future__ import annotations

from research_agent.models import InputMode, ResearchInput
from research_agent.queries import step1_queries, step3_queries, subject_label

EXPECTED_STEP3_KEYS = {"earnings", "guidance", "competitive", "balance_sheet", "valuation", "bear_case"}


class TestStep1Queries:
    def test_ticker_mode_returns_two_queries(self):
        inp = ResearchInput(mode=InputMode.TICKER, value="AAPL")
        queries = step1_queries(inp)
        assert len(queries) == 2
        assert all("AAPL" in q for q in queries)

    def test_sector_mode_returns_two_queries(self):
        inp = ResearchInput(mode=InputMode.SECTOR, value="Technology")
        queries = step1_queries(inp)
        assert len(queries) == 2
        assert all("Technology" in q for q in queries)

    def test_thesis_mode_returns_two_queries(self):
        inp = ResearchInput(mode=InputMode.THESIS, value="AI infrastructure spending")
        queries = step1_queries(inp)
        assert len(queries) == 2
        assert all("AI infrastructure spending" in q for q in queries)


class TestStep3Queries:
    def test_ticker_mode_has_six_categories(self):
        inp = ResearchInput(mode=InputMode.TICKER, value="MSFT")
        cats = step3_queries(inp)
        assert set(cats.keys()) == EXPECTED_STEP3_KEYS
        assert all("MSFT" in v for v in cats.values())

    def test_sector_mode_has_six_categories(self):
        inp = ResearchInput(mode=InputMode.SECTOR, value="Energy")
        cats = step3_queries(inp)
        assert set(cats.keys()) == EXPECTED_STEP3_KEYS
        assert all("Energy" in v for v in cats.values())

    def test_thesis_mode_has_six_categories(self):
        inp = ResearchInput(mode=InputMode.THESIS, value="EV battery demand")
        cats = step3_queries(inp)
        assert set(cats.keys()) == EXPECTED_STEP3_KEYS
        assert all("EV battery demand" in v for v in cats.values())


class TestSubjectLabel:
    def test_ticker_label(self):
        inp = ResearchInput(mode=InputMode.TICKER, value="aapl")
        assert subject_label(inp) == "Ticker: AAPL"

    def test_sector_label(self):
        inp = ResearchInput(mode=InputMode.SECTOR, value="Technology")
        assert subject_label(inp) == "Sector: Technology"

    def test_thesis_label(self):
        inp = ResearchInput(mode=InputMode.THESIS, value="AI infrastructure spending")
        assert subject_label(inp) == "Thesis: AI infrastructure spending"
