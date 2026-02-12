"""Tests for research_agent.pipeline (end-to-end with mocks)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

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
    def test_ticker_mode_not_implemented_raises(self):
        """Non-ticker modes raise NotImplementedError."""
        from research_agent.pipeline import run

        inp = ResearchInput(mode=InputMode.SECTOR, value="Technology")
        with pytest.raises(NotImplementedError, match="sector"):
            run(inp, _make_config())

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

        # Mock store instance
        mock_store_instance = MagicMock()
        MockStore.return_value = mock_store_instance

        config = _make_config()
        card = run(inp, config)

        assert card.verdict == Verdict.BUY_THE_DIP
        mock_run_loop.assert_called_once()
        mock_store_instance.close.assert_called_once()
