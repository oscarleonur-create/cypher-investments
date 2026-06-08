"""Tests for adaptive query follow-up in the research agent."""

from __future__ import annotations

from unittest.mock import MagicMock

from research_agent.agent import (
    _CardSynthesisResponse,
    _ClassificationResponse,
    _compute_coverage,
    _EvidenceItemRaw,
    _FactExtractionResponse,
    _GapAnalysisResponse,
    _TranscriptSummaryResponse,
    _TriggerResponse,
    run_loop,
    step3_adaptive_followup,
)
from research_agent.config import ResearchConfig
from research_agent.evidence import SourceRegistry
from research_agent.models import (
    AgentState,
    CoverageMetrics,
    EvidenceItem,
    FactPack,
    InputMode,
    ResearchInput,
    Verdict,
)
from research_agent.search import SearchResult


def _make_config(**overrides) -> ResearchConfig:
    defaults = dict(
        _env_file=None,
        tavily_api_key="test",
        openrouter_api_key="test",
        max_iterations=4,
        min_evidence_items=4,
        adaptive_queries_enabled=True,
    )
    defaults.update(overrides)
    return ResearchConfig(**defaults)


def _make_state(ticker="AAPL") -> AgentState:
    return AgentState(input=ResearchInput(mode=InputMode.TICKER, value=ticker))


class TestComputeCoverage:
    def test_empty_fact_pack(self):
        fp = FactPack()
        coverage = _compute_coverage(fp)
        assert coverage.earnings_highlights == 0
        assert coverage.guidance_changes == 0
        assert coverage.unit_economics == 0
        assert coverage.balance_sheet == 0
        assert coverage.competitive_landscape == 0
        assert coverage.valuation_comparison == 0
        assert coverage.bear_rebuttals == 0

    def test_counts_per_category(self):
        fp = FactPack(
            earnings_highlights=[
                EvidenceItem(text="Rev grew 5%"),
                EvidenceItem(text="EPS beat"),
                EvidenceItem(text="Margins up"),
            ],
            guidance_changes=[EvidenceItem(text="Guidance reiterated")],
        )
        coverage = _compute_coverage(fp)
        assert coverage.earnings_highlights == 3
        assert coverage.guidance_changes == 1
        assert coverage.competitive_landscape == 0


class TestCoverageMetrics:
    def test_sparse_categories_all_empty(self):
        metrics = CoverageMetrics()
        assert len(metrics.sparse_categories) == 7  # All categories sparse

    def test_sparse_categories_some_covered(self):
        metrics = CoverageMetrics(
            earnings_highlights=3,
            guidance_changes=2,
            competitive_landscape=0,
        )
        sparse = metrics.sparse_categories
        assert "earnings_highlights" not in sparse
        assert "guidance_changes" not in sparse
        assert "competitive_landscape" in sparse
        assert "unit_economics" in sparse

    def test_coverage_score_none_covered(self):
        metrics = CoverageMetrics()
        assert metrics.coverage_score == 0.0

    def test_coverage_score_all_covered(self):
        metrics = CoverageMetrics(
            earnings_highlights=2,
            guidance_changes=2,
            competitive_landscape=2,
            unit_economics=2,
            balance_sheet=2,
            valuation_comparison=2,
            bear_rebuttals=2,
        )
        assert metrics.coverage_score == 1.0

    def test_coverage_score_partial(self):
        metrics = CoverageMetrics(
            earnings_highlights=3,
            guidance_changes=2,
            competitive_landscape=0,
            unit_economics=0,
            balance_sheet=0,
            valuation_comparison=0,
            bear_rebuttals=0,
        )
        # 2 out of 7 well-covered
        assert abs(metrics.coverage_score - 2 / 7) < 0.01

    def test_as_dict(self):
        metrics = CoverageMetrics(earnings_highlights=3)
        d = metrics.as_dict
        assert d["earnings_highlights"] == 3
        assert d["guidance_changes"] == 0
        assert len(d) == 7


class TestAdaptiveFollowup:
    def test_generates_and_executes_queries(self):
        state = _make_state()
        state.step3_static_done = True
        state.fact_pack = FactPack(
            earnings_highlights=[EvidenceItem(text="Rev grew 5%")],
        )
        state.queries_executed = ["some previous query"]

        config = _make_config(max_queries_per_iteration=4)
        registry = SourceRegistry()

        mock_search = MagicMock()
        mock_search.search.return_value = [
            SearchResult(
                url="https://sec.gov/test", title="SEC Filing", content="Balance sheet data"
            )
        ]

        mock_llm = MagicMock()
        # First call: gap analysis
        mock_llm.complete.side_effect = [
            _GapAnalysisResponse(
                queries={
                    "balance_sheet": ["AAPL balance sheet cash debt 2025"],
                    "unit_economics": ["AAPL gross margin trend 2025"],
                }
            ),
            # Second call: fact extraction
            _FactExtractionResponse(
                balance_sheet=[
                    _EvidenceItemRaw(text="Cash: $62B", source_urls=["https://sec.gov/test"])
                ],
            ),
        ]

        step3_adaptive_followup(state, mock_search, mock_llm, registry, config)

        # Should have executed new queries
        assert "AAPL balance sheet cash debt 2025" in state.queries_executed
        assert mock_search.search.call_count >= 1

    def test_respects_budget(self):
        state = _make_state()
        state.step3_static_done = True
        state.fact_pack = FactPack()
        config = _make_config(max_queries_per_iteration=1)
        registry = SourceRegistry()

        mock_search = MagicMock()
        mock_search.search.return_value = [
            SearchResult(url="https://test.com/1", title="Test", content="Content")
        ]

        mock_llm = MagicMock()
        mock_llm.complete.side_effect = [
            _GapAnalysisResponse(
                queries={
                    "balance_sheet": ["query1"],
                    "unit_economics": ["query2"],
                    "competitive_landscape": ["query3"],
                }
            ),
            _FactExtractionResponse(),
        ]

        step3_adaptive_followup(state, mock_search, mock_llm, registry, config)

        # Budget is 1 query, so only 1 search call
        assert mock_search.search.call_count == 1

    def test_skips_already_executed(self):
        state = _make_state()
        state.step3_static_done = True
        state.fact_pack = FactPack()
        state.queries_executed = ["already ran this"]
        config = _make_config(max_queries_per_iteration=4)
        registry = SourceRegistry()

        mock_search = MagicMock()
        mock_search.search.return_value = [
            SearchResult(url="https://test.com/1", title="Test", content="Content")
        ]

        mock_llm = MagicMock()
        mock_llm.complete.side_effect = [
            _GapAnalysisResponse(
                queries={
                    "balance_sheet": ["already ran this", "new query"],
                }
            ),
            _FactExtractionResponse(),
        ]

        step3_adaptive_followup(state, mock_search, mock_llm, registry, config)

        # Only "new query" should be executed
        search_queries = [call.args[0] for call in mock_search.search.call_args_list]
        assert "already ran this" not in search_queries
        assert "new query" in search_queries

    def test_skips_when_all_covered(self):
        state = _make_state()
        state.fact_pack = FactPack(
            earnings_highlights=[EvidenceItem(text="a"), EvidenceItem(text="b")],
            guidance_changes=[EvidenceItem(text="a"), EvidenceItem(text="b")],
            competitive_landscape=[EvidenceItem(text="a"), EvidenceItem(text="b")],
            unit_economics=[EvidenceItem(text="a"), EvidenceItem(text="b")],
            balance_sheet=[EvidenceItem(text="a"), EvidenceItem(text="b")],
            valuation_comparison=[EvidenceItem(text="a"), EvidenceItem(text="b")],
            bear_rebuttals=[EvidenceItem(text="a"), EvidenceItem(text="b")],
        )
        config = _make_config()
        registry = SourceRegistry()
        mock_search = MagicMock()
        mock_llm = MagicMock()

        step3_adaptive_followup(state, mock_search, mock_llm, registry, config)

        # No LLM call should be made
        mock_llm.complete.assert_not_called()

    def test_fallback_on_llm_failure(self):
        state = _make_state()
        state.step3_static_done = True
        state.fact_pack = FactPack()
        config = _make_config(max_queries_per_iteration=4)
        registry = SourceRegistry()

        mock_search = MagicMock()
        mock_search.search.return_value = [
            SearchResult(url="https://test.com/1", title="Test", content="Content")
        ]

        mock_llm = MagicMock()
        # First call (gap analysis) fails, then fact extraction succeeds
        mock_llm.complete.side_effect = [
            Exception("LLM error"),
            _FactExtractionResponse(),
        ]

        step3_adaptive_followup(state, mock_search, mock_llm, registry, config)

        # Should still execute fallback queries
        assert mock_search.search.call_count >= 1


class TestRunLoopStaticThenAdaptive:
    def test_first_pass_static_second_adaptive(self):
        """First step 3 pass is static, subsequent passes use adaptive follow-up."""
        state = _make_state()
        # Low evidence threshold that won't be met after first static pass
        config = _make_config(
            min_evidence_items=6,
            max_iterations=5,
            adaptive_queries_enabled=True,
        )
        registry = SourceRegistry()

        mock_search = MagicMock()
        mock_search.search.return_value = [
            SearchResult(url="https://reuters.com/test", title="Test", content="Content")
        ]

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
        transcript_resp = _TranscriptSummaryResponse(management_tone="bullish")
        # First static pass yields 1 item (below threshold of 6)
        facts_resp_static = _FactExtractionResponse(
            earnings_highlights=[
                _EvidenceItemRaw(text="Rev grew 5%", source_urls=[]),
            ],
        )
        # Adaptive follow-up gap analysis
        gap_resp = _GapAnalysisResponse(queries={"guidance_changes": ["AAPL guidance update 2026"]})
        # Adaptive fact extraction yields more items
        facts_resp_adaptive = _FactExtractionResponse(
            guidance_changes=[
                _EvidenceItemRaw(text="Guidance raised", source_urls=[]),
                _EvidenceItemRaw(text="Revenue target $400B", source_urls=[]),
            ],
            balance_sheet=[
                _EvidenceItemRaw(text="Cash $62B", source_urls=[]),
                _EvidenceItemRaw(text="Debt $108B", source_urls=[]),
            ],
            competitive_landscape=[
                _EvidenceItemRaw(text="Market leader", source_urls=[]),
                _EvidenceItemRaw(text="Growing services", source_urls=[]),
            ],
        )
        card_resp = _CardSynthesisResponse(
            verdict="BUY_THE_DIP",
            catalyst_summary="Earnings miss",
            bull_case=["Strong fundamentals"],
            bear_case=["Macro risk"],
            key_metrics={"revenue_growth": "5%"},
        )

        mock_llm = MagicMock()
        mock_llm.complete.side_effect = [
            trigger_resp,  # step 1
            classification_resp,  # step 2
            transcript_resp,  # transcript summary in step 3 static
            facts_resp_static,  # step 3 static fact extraction
            gap_resp,  # step 3b adaptive gap analysis
            facts_resp_adaptive,  # step 3b adaptive fact extraction
            card_resp,  # step 4
        ]

        card = run_loop(state, mock_search, mock_llm, registry, config)

        assert card is not None
        assert card.verdict == Verdict.BUY_THE_DIP
        assert state.step3_static_done is True


class TestCardIncludesCoverageConfidence:
    def test_card_has_coverage_confidence(self):
        state = _make_state()
        config = _make_config(min_evidence_items=2)
        registry = SourceRegistry()

        mock_search = MagicMock()
        mock_search.search.return_value = [
            SearchResult(url="https://reuters.com/test", title="Test", content="Content")
        ]

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
        transcript_resp = _TranscriptSummaryResponse(management_tone="bullish")
        facts_resp = _FactExtractionResponse(
            earnings_highlights=[
                _EvidenceItemRaw(text="Rev grew 5%", source_urls=[]),
                _EvidenceItemRaw(text="EPS beat", source_urls=[]),
            ],
        )
        card_resp = _CardSynthesisResponse(
            verdict="BUY_THE_DIP",
            catalyst_summary="Q1 miss",
            bull_case=["Strong revenue"],
            bear_case=["Macro risk"],
            key_metrics={"revenue_growth": "5%"},
        )

        mock_llm = MagicMock()
        mock_llm.complete.side_effect = [
            trigger_resp,
            classification_resp,
            transcript_resp,
            facts_resp,
            card_resp,
        ]

        card = run_loop(state, mock_search, mock_llm, registry, config)

        assert card is not None
        assert isinstance(card.coverage_confidence, dict)
        assert "earnings_highlights" in card.coverage_confidence
        # earnings_highlights has 2 items -> confidence 1.0
        assert card.coverage_confidence["earnings_highlights"] == 1.0
        # Other categories are empty -> confidence 0.0
        assert card.coverage_confidence["unit_economics"] == 0.0
