"""Tests for research_agent.pipeline (end-to-end with mocks)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from research_agent.models import InputMode, ResearchInput, Verdict


def _make_config(**overrides):
    from research_agent.config import ResearchConfig

    defaults = dict(
        _env_file=None,
        tavily_api_key="test",
        anthropic_api_key="test",
        max_iterations=4,
        min_evidence_items=2,
    )
    defaults.update(overrides)
    return ResearchConfig(**defaults)


class TestPipeline:
    @patch("research_agent.pipeline.run_loop")
    @patch("research_agent.pipeline.ClaudeLLM")
    @patch("research_agent.pipeline.TavilyClient")
    @patch("research_agent.pipeline.Store")
    def test_ticker_mode_calls_run_loop(self, MockStore, MockTavily, MockLLM, mock_run_loop):
        """Ticker mode initializes clients and calls run_loop."""
        from research_agent.models import OpportunityCard
        from research_agent.pipeline import run

        inp = ResearchInput(mode=InputMode.TICKER, value="AAPL")
        mock_card = OpportunityCard(id="abc", input=inp, verdict=Verdict.BUY_THE_DIP)
        mock_run_loop.return_value = mock_card

        mock_store_instance = MagicMock()
        MockStore.return_value = mock_store_instance

        config = _make_config()
        card = run(inp, config)

        assert card.verdict == Verdict.BUY_THE_DIP
        mock_run_loop.assert_called_once()
        mock_store_instance.close.assert_called_once()

    @patch("research_agent.pipeline.run_loop")
    @patch("research_agent.pipeline.ClaudeLLM")
    @patch("research_agent.pipeline.TavilyClient")
    @patch("research_agent.pipeline.Store")
    def test_sector_mode_calls_run_loop(self, MockStore, MockTavily, MockLLM, mock_run_loop):
        """Sector mode initializes clients and calls run_loop."""
        from research_agent.models import OpportunityCard
        from research_agent.pipeline import run

        inp = ResearchInput(mode=InputMode.SECTOR, value="Technology")
        mock_card = OpportunityCard(id="sec1", input=inp, verdict=Verdict.WATCH)
        mock_run_loop.return_value = mock_card

        mock_store_instance = MagicMock()
        MockStore.return_value = mock_store_instance

        config = _make_config()
        card = run(inp, config)

        assert card.verdict == Verdict.WATCH
        mock_run_loop.assert_called_once()
        mock_store_instance.close.assert_called_once()

    @patch("research_agent.pipeline.run_loop")
    @patch("research_agent.pipeline.ClaudeLLM")
    @patch("research_agent.pipeline.TavilyClient")
    @patch("research_agent.pipeline.Store")
    def test_thesis_mode_calls_run_loop(self, MockStore, MockTavily, MockLLM, mock_run_loop):
        """Thesis mode initializes clients and calls run_loop."""
        from research_agent.models import OpportunityCard
        from research_agent.pipeline import run

        inp = ResearchInput(mode=InputMode.THESIS, value="AI infrastructure spending")
        mock_card = OpportunityCard(id="th1", input=inp, verdict=Verdict.BUY_THE_DIP)
        mock_run_loop.return_value = mock_card

        mock_store_instance = MagicMock()
        MockStore.return_value = mock_store_instance

        config = _make_config()
        card = run(inp, config)

        assert card.verdict == Verdict.BUY_THE_DIP
        mock_run_loop.assert_called_once()
        mock_store_instance.close.assert_called_once()
