"""Tests for research_agent.agent (agent loop with mocked search + LLM)."""

from __future__ import annotations

from unittest.mock import MagicMock

from research_agent.agent import (
    _TriggerResponse,
    _ClassificationResponse,
    _FactExtractionResponse,
    _CardSynthesisResponse,
    _EvidenceItemRaw,
    run_loop,
    step1_detect_trigger,
    step2_classify_dip,
)
from research_agent.config import ResearchConfig
from research_agent.evidence import SourceRegistry
from research_agent.models import (
    AgentState,
    DipType,
    InputMode,
    ResearchInput,
    Verdict,
)
from research_agent.search import SearchResult


def _make_config(**overrides) -> ResearchConfig:
    defaults = dict(
        _env_file=None,
        tavily_api_key="test",
        anthropic_api_key="test",
        max_iterations=4,
        min_evidence_items=2,  # Low threshold for testing
    )
    defaults.update(overrides)
    return ResearchConfig(**defaults)


def _make_state(ticker="AAPL") -> AgentState:
    return AgentState(input=ResearchInput(mode=InputMode.TICKER, value=ticker))


def _make_sector_state(sector="Technology") -> AgentState:
    return AgentState(input=ResearchInput(mode=InputMode.SECTOR, value=sector))


def _make_thesis_state(thesis="AI infrastructure spending") -> AgentState:
    return AgentState(input=ResearchInput(mode=InputMode.THESIS, value=thesis))


class TestStep1DetectTrigger:
    def test_sets_trigger_on_success(self):
        state = _make_state()
        config = _make_config()
        registry = SourceRegistry()

        mock_search = MagicMock()
        mock_search.search.return_value = [
            SearchResult(
                url="https://reuters.com/aapl",
                title="Apple drops",
                content="Apple shares fell after earnings miss",
            )
        ]

        mock_llm = MagicMock()
        mock_llm.complete.return_value = _TriggerResponse(
            found=True,
            trigger_type="earnings_miss",
            summary="Apple shares dropped after Q1 earnings miss",
            source_urls=["https://reuters.com/aapl"],
        )

        step1_detect_trigger(state, mock_search, mock_llm, registry, config)

        assert state.trigger is not None
        assert state.trigger.found is True
        assert "earnings" in state.trigger.summary.lower()

    def test_handles_no_search_results(self):
        state = _make_state()
        config = _make_config()
        registry = SourceRegistry()

        mock_search = MagicMock()
        mock_search.search.return_value = []

        mock_llm = MagicMock()

        step1_detect_trigger(state, mock_search, mock_llm, registry, config)

        assert state.trigger is not None
        assert state.trigger.found is False
        mock_llm.complete.assert_not_called()


class TestStep2ClassifyDip:
    def test_classifies_dip(self):
        state = _make_state()
        state.trigger = MagicMock(summary="Earnings miss")
        registry = SourceRegistry()

        mock_llm = MagicMock()
        mock_llm.complete.return_value = _ClassificationResponse(
            dip_type="TEMPORARY",
            confidence=0.8,
            reasoning="One-time event, guidance intact",
        )

        step2_classify_dip(state, mock_llm, registry)

        assert state.classification is not None
        assert state.classification.dip_type == DipType.TEMPORARY
        assert state.classification.confidence == 0.8


class TestRunLoop:
    def test_full_loop_produces_card(self):
        """Full agent loop with mocked externals produces an OpportunityCard."""
        state = _make_state()
        config = _make_config()
        registry = SourceRegistry()

        mock_search = MagicMock()
        mock_search.search.return_value = [
            SearchResult(
                url="https://reuters.com/test",
                title="Test",
                content="Test content",
            )
        ]

        # Mock LLM to return different responses based on call order
        trigger_resp = _TriggerResponse(
            found=True,
            trigger_type="earnings",
            summary="Earnings miss",
            source_urls=["https://reuters.com/test"],
        )
        classification_resp = _ClassificationResponse(
            dip_type="TEMPORARY",
            confidence=0.85,
            reasoning="One-time miss",
        )
        facts_resp = _FactExtractionResponse(
            earnings_highlights=[
                _EvidenceItemRaw(text="Revenue grew 5%", source_urls=["https://reuters.com/test"]),
                _EvidenceItemRaw(text="EPS beat", source_urls=["https://reuters.com/test"]),
            ],
            guidance_changes=[
                _EvidenceItemRaw(text="Guidance reiterated", source_urls=[]),
            ],
        )
        card_resp = _CardSynthesisResponse(
            verdict="BUY_THE_DIP",
            catalyst_summary="Q1 earnings missed estimates, shares fell 10%",
            catalyst_date="2025-01-30",
            bull_case=["Strong earnings", "Growing revenue"],
            bear_case=["Macro risk"],
            key_metrics={
                "revenue_growth": "5%",
                "margins": "40%",
                "fcf": "Unknown",
                "cash": "Unknown",
                "debt": "Unknown",
                "guidance_notes": "Guidance reiterated",
            },
            risks=["Market downturn"],
            invalidation=["Revenue decline"],
            validation_checklist=["Q2 results"],
            next_actions=["Monitor earnings"],
        )

        mock_llm = MagicMock()
        mock_llm.complete.side_effect = [
            trigger_resp,
            classification_resp,
            facts_resp,
            card_resp,
        ]

        card = run_loop(state, mock_search, mock_llm, registry, config)

        assert card is not None
        assert card.verdict == Verdict.BUY_THE_DIP
        assert len(card.bull_case) > 0
        assert card.id == state.input.run_id()


class TestStep1DetectTriggerSector:
    def test_uses_sector_queries(self):
        state = _make_sector_state("Technology")
        config = _make_config()
        registry = SourceRegistry()

        mock_search = MagicMock()
        mock_search.search.return_value = [
            SearchResult(
                url="https://reuters.com/tech",
                title="Tech sector selloff",
                content="Technology sector declined on rate fears",
            )
        ]

        mock_llm = MagicMock()
        mock_llm.complete.return_value = _TriggerResponse(
            found=True,
            trigger_type="macro",
            summary="Technology sector fell on rising rate concerns",
            source_urls=["https://reuters.com/tech"],
        )

        step1_detect_trigger(state, mock_search, mock_llm, registry, config)

        assert state.trigger is not None
        assert state.trigger.found is True
        # Verify search was called with sector-appropriate queries
        search_queries = [call.args[0] for call in mock_search.search.call_args_list]
        assert all("Technology" in q for q in search_queries)
        assert not any("stock price drop" in q for q in search_queries)


class TestRunLoopSector:
    def test_full_loop_with_sector_produces_card(self):
        state = _make_sector_state("Technology")
        config = _make_config()
        registry = SourceRegistry()

        mock_search = MagicMock()
        mock_search.search.return_value = [
            SearchResult(
                url="https://reuters.com/tech",
                title="Tech sector",
                content="Tech sector overview",
            )
        ]

        trigger_resp = _TriggerResponse(
            found=True,
            trigger_type="macro",
            summary="Sector decline on macro fears",
            source_urls=["https://reuters.com/tech"],
        )
        classification_resp = _ClassificationResponse(
            dip_type="TEMPORARY",
            confidence=0.75,
            reasoning="Cyclical downturn",
        )
        facts_resp = _FactExtractionResponse(
            earnings_highlights=[
                _EvidenceItemRaw(text="Sector revenue grew 8%", source_urls=[]),
                _EvidenceItemRaw(text="Margins stable", source_urls=[]),
            ],
            guidance_changes=[
                _EvidenceItemRaw(text="Mixed guidance", source_urls=[]),
            ],
        )
        card_resp = _CardSynthesisResponse(
            verdict="WATCH",
            catalyst_summary="Rate hike fears drove tech selloff",
            bull_case=["Strong fundamentals"],
            bear_case=["Rate sensitivity"],
            key_metrics={"revenue_growth": "8%"},
            risks=["Continued rate hikes"],
        )

        mock_llm = MagicMock()
        mock_llm.complete.side_effect = [
            trigger_resp,
            classification_resp,
            facts_resp,
            card_resp,
        ]

        card = run_loop(state, mock_search, mock_llm, registry, config)

        assert card is not None
        assert card.verdict == Verdict.WATCH
        assert card.input.mode == InputMode.SECTOR
