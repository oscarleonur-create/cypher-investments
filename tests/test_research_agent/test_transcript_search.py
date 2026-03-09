"""Tests for earnings call transcript search integration."""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock

from research_agent.agent import (
    _format_evidence_for_llm,
    _format_transcript_summary,
    _TranscriptSummaryResponse,
    step3_research_facts,
)
from research_agent.config import ResearchConfig
from research_agent.evidence import SourceRegistry
from research_agent.llm import TRANSCRIPT_SUMMARIZATION_PROMPT
from research_agent.models import (
    AgentState,
    ClassificationResult,
    DipType,
    InputMode,
    ResearchInput,
    TranscriptSummary,
    TriggerResult,
)
from research_agent.queries import (
    TRANSCRIPT_DOMAINS,
    _infer_latest_quarter,
    step3_transcript_queries,
)
from research_agent.search import SearchResult


def _make_config(**overrides) -> ResearchConfig:
    defaults = dict(
        _env_file=None,
        perplexity_api_key="test",
        anthropic_api_key="test",
        max_iterations=4,
        max_queries_per_iteration=10,
        min_evidence_items=2,
    )
    defaults.update(overrides)
    return ResearchConfig(**defaults)


def _make_ticker_state(ticker="AAPL") -> AgentState:
    state = AgentState(input=ResearchInput(mode=InputMode.TICKER, value=ticker))
    state.trigger = TriggerResult(found=True, summary="Earnings miss")
    state.classification = ClassificationResult(
        dip_type=DipType.TEMPORARY, confidence=0.8, reasoning="One-time"
    )
    return state


def _make_sector_state(sector="Technology") -> AgentState:
    state = AgentState(input=ResearchInput(mode=InputMode.SECTOR, value=sector))
    state.trigger = TriggerResult(found=True, summary="Sector rotation")
    state.classification = ClassificationResult(
        dip_type=DipType.TEMPORARY, confidence=0.7, reasoning="Cyclical"
    )
    return state


# ── Quarter inference tests ────────────────────────────────────────────────


class TestInferLatestQuarter:
    def test_q1_returns_q4_prior_year(self):
        assert _infer_latest_quarter(date(2026, 2, 15)) == ("Q4", 2025)

    def test_march_returns_q4_prior_year(self):
        assert _infer_latest_quarter(date(2026, 3, 31)) == ("Q4", 2025)

    def test_april_returns_q1_same_year(self):
        assert _infer_latest_quarter(date(2026, 4, 1)) == ("Q1", 2026)

    def test_june_returns_q1_same_year(self):
        assert _infer_latest_quarter(date(2026, 6, 30)) == ("Q1", 2026)

    def test_july_returns_q2_same_year(self):
        assert _infer_latest_quarter(date(2026, 7, 1)) == ("Q2", 2026)

    def test_october_returns_q3_same_year(self):
        assert _infer_latest_quarter(date(2026, 10, 15)) == ("Q3", 2026)

    def test_december_returns_q3_same_year(self):
        assert _infer_latest_quarter(date(2026, 12, 31)) == ("Q3", 2026)


# ── Transcript query generation tests ──────────────────────────────────────


class TestTranscriptQueries:
    def test_ticker_mode_returns_three_queries(self):
        inp = ResearchInput(mode=InputMode.TICKER, value="AAPL")
        queries = step3_transcript_queries(inp)
        assert len(queries) == 3
        assert all("AAPL" in q for q in queries)

    def test_queries_contain_transcript_keywords(self):
        inp = ResearchInput(mode=InputMode.TICKER, value="MSFT")
        queries = step3_transcript_queries(inp)
        assert any("transcript" in q for q in queries)
        assert any("guidance" in q for q in queries)
        assert any("Q&A" in q for q in queries)

    def test_sector_mode_returns_empty(self):
        inp = ResearchInput(mode=InputMode.SECTOR, value="Technology")
        assert step3_transcript_queries(inp) == []

    def test_thesis_mode_returns_empty(self):
        inp = ResearchInput(mode=InputMode.THESIS, value="AI spending")
        assert step3_transcript_queries(inp) == []

    def test_queries_include_quarter_and_year(self):
        inp = ResearchInput(mode=InputMode.TICKER, value="AAPL")
        queries = step3_transcript_queries(inp)
        # At least one query should include a quarter label (Q1-Q4)
        assert any(any(f"Q{n}" in q for n in range(1, 5)) for q in queries)


# ── Transcript domains ─────────────────────────────────────────────────────


class TestTranscriptDomains:
    def test_domains_include_seeking_alpha(self):
        assert "seekingalpha.com" in TRANSCRIPT_DOMAINS

    def test_domains_include_motley_fool(self):
        assert "fool.com" in TRANSCRIPT_DOMAINS

    def test_domains_include_nasdaq(self):
        assert "nasdaq.com" in TRANSCRIPT_DOMAINS


# ── Integration: step3_research_facts with transcript search ───────────────


class TestStep3TranscriptSearch:
    def _make_fact_resp(self):
        from research_agent.agent import _EvidenceItemRaw, _FactExtractionResponse

        return _FactExtractionResponse(
            earnings_highlights=[
                _EvidenceItemRaw(text="Revenue grew 5%", source_urls=[]),
                _EvidenceItemRaw(text="EPS beat", source_urls=[]),
            ],
        )

    def test_transcript_results_registered_as_sources(self):
        state = _make_ticker_state()
        config = _make_config()
        registry = SourceRegistry()

        mock_search = MagicMock()
        # Standard search returns one result, transcript search returns one result
        transcript_result = SearchResult(
            url="https://seekingalpha.com/article/aapl-q4-transcript",
            title="AAPL Q4 Earnings Call Transcript",
            content="Management discussed revenue growth...",
        )
        mock_search.search.return_value = [transcript_result]

        transcript_resp = _TranscriptSummaryResponse(
            management_tone="bullish",
            revenue_discussion="Revenue up 8%",
            guidance_details="Raised full-year guidance",
        )
        mock_llm = MagicMock()
        mock_llm.complete.side_effect = [transcript_resp, self._make_fact_resp()]

        step3_research_facts(state, mock_search, mock_llm, registry, config)

        # Transcript source should be registered
        assert registry.count > 0
        sources = registry.all_sources()
        urls = [s.url for s in sources]
        assert "https://seekingalpha.com/article/aapl-q4-transcript" in urls

    def test_transcript_summary_stored_in_state(self):
        state = _make_ticker_state()
        config = _make_config()
        registry = SourceRegistry()

        mock_search = MagicMock()
        mock_search.search.return_value = [
            SearchResult(
                url="https://fool.com/aapl-transcript",
                title="AAPL Transcript",
                content="CEO discussed margins...",
            )
        ]

        transcript_resp = _TranscriptSummaryResponse(
            management_tone="cautious",
            revenue_discussion="Revenue flat",
            earnings_discussion="Margins compressed",
            guidance_details="Guidance lowered",
            qa_highlights=["Analyst pressed on margins"],
            key_quotes=["CFO: 'We expect headwinds to persist'"],
        )
        mock_llm = MagicMock()
        mock_llm.complete.side_effect = [transcript_resp, self._make_fact_resp()]

        step3_research_facts(state, mock_search, mock_llm, registry, config)

        assert state.transcript_summary is not None
        assert state.transcript_summary.management_tone == "cautious"
        assert state.transcript_summary.revenue_discussion == "Revenue flat"
        assert state.transcript_summary.guidance_details == "Guidance lowered"
        assert len(state.transcript_summary.qa_highlights) == 1
        assert len(state.transcript_summary.key_quotes) == 1

    def test_llm_called_with_transcript_prompt(self):
        state = _make_ticker_state()
        config = _make_config()
        registry = SourceRegistry()

        mock_search = MagicMock()
        mock_search.search.return_value = [
            SearchResult(
                url="https://seekingalpha.com/aapl",
                title="AAPL Transcript",
                content="Transcript content",
            )
        ]

        transcript_resp = _TranscriptSummaryResponse(management_tone="bullish")
        mock_llm = MagicMock()
        mock_llm.complete.side_effect = [transcript_resp, self._make_fact_resp()]

        step3_research_facts(state, mock_search, mock_llm, registry, config)

        # First LLM call should be transcript summarization
        first_call = mock_llm.complete.call_args_list[0]
        assert first_call.kwargs["system_prompt"] == TRANSCRIPT_SUMMARIZATION_PROMPT
        assert first_call.kwargs["response_model"] == _TranscriptSummaryResponse

    def test_transcript_search_skipped_for_sector_mode(self):
        state = _make_sector_state()
        config = _make_config()
        registry = SourceRegistry()

        mock_search = MagicMock()
        mock_search.search.return_value = [
            SearchResult(
                url="https://reuters.com/tech",
                title="Tech sector",
                content="Sector overview",
            )
        ]

        mock_llm = MagicMock()
        mock_llm.complete.return_value = self._make_fact_resp()

        step3_research_facts(state, mock_search, mock_llm, registry, config)

        # Search should NOT be called with transcript domains
        for c in mock_search.search.call_args_list:
            domains = c.kwargs.get("domains") or (c.args[1] if len(c.args) > 1 else None)
            assert domains != TRANSCRIPT_DOMAINS

        # No transcript summary
        assert state.transcript_summary is None

    def test_transcript_search_skipped_when_disabled(self):
        state = _make_ticker_state()
        config = _make_config(transcript_search_enabled=False)
        registry = SourceRegistry()

        mock_search = MagicMock()
        mock_search.search.return_value = [
            SearchResult(
                url="https://reuters.com/aapl",
                title="AAPL",
                content="Content",
            )
        ]

        mock_llm = MagicMock()
        mock_llm.complete.return_value = self._make_fact_resp()

        step3_research_facts(state, mock_search, mock_llm, registry, config)

        # Search should NOT be called with transcript domains
        for c in mock_search.search.call_args_list:
            domains = c.kwargs.get("domains") or (c.args[1] if len(c.args) > 1 else None)
            assert domains != TRANSCRIPT_DOMAINS

        assert state.transcript_summary is None

    def test_pipeline_completes_when_transcript_returns_no_results(self):
        state = _make_ticker_state()
        config = _make_config()
        registry = SourceRegistry()

        # Standard search returns results, transcript search returns empty
        call_count = [0]

        def search_side_effect(query, **kwargs):
            call_count[0] += 1
            domains = kwargs.get("domains")
            if domains == TRANSCRIPT_DOMAINS:
                return []  # No transcript results
            return [
                SearchResult(
                    url="https://reuters.com/aapl",
                    title="AAPL",
                    content="Content",
                )
            ]

        mock_search = MagicMock()
        mock_search.search.side_effect = search_side_effect

        mock_llm = MagicMock()
        mock_llm.complete.return_value = self._make_fact_resp()

        step3_research_facts(state, mock_search, mock_llm, registry, config)

        # No transcript summary since no transcript results
        assert state.transcript_summary is None
        # But fact extraction should still happen
        assert mock_llm.complete.call_count == 1  # Only fact extraction, no transcript call

    def test_transcript_summarization_error_does_not_block_pipeline(self):
        state = _make_ticker_state()
        config = _make_config()
        registry = SourceRegistry()

        mock_search = MagicMock()
        mock_search.search.return_value = [
            SearchResult(
                url="https://seekingalpha.com/aapl",
                title="AAPL Transcript",
                content="Transcript",
            )
        ]

        mock_llm = MagicMock()
        # Transcript summarization fails, fact extraction succeeds
        mock_llm.complete.side_effect = [
            Exception("LLM timeout"),
            self._make_fact_resp(),
        ]

        step3_research_facts(state, mock_search, mock_llm, registry, config)

        # Transcript summary not set due to error
        assert state.transcript_summary is None
        # Error recorded
        assert any("transcript_summarization" in e for e in state.errors)
        # Fact extraction still happened
        assert state.fact_pack.total_items > 0


# ── Format transcript summary tests ────────────────────────────────────────


class TestFormatTranscriptSummary:
    def test_formats_all_fields(self):
        summary = TranscriptSummary(
            management_tone="bullish",
            revenue_discussion="Revenue up 10%",
            earnings_discussion="EPS beat by $0.05",
            guidance_details="Raised Q2 guidance",
            qa_highlights=["Analyst asked about margins", "CEO deflected"],
            key_quotes=["CEO: 'Best quarter ever'"],
        )
        text = _format_transcript_summary(summary)
        assert "Earnings Call Highlights" in text
        assert "bullish" in text
        assert "Revenue up 10%" in text
        assert "EPS beat" in text
        assert "Raised Q2 guidance" in text
        assert "Analyst asked about margins" in text
        assert "Best quarter ever" in text

    def test_omits_empty_fields(self):
        summary = TranscriptSummary(management_tone="cautious")
        text = _format_transcript_summary(summary)
        assert "cautious" in text
        assert "Revenue Discussion" not in text
        assert "Q&A Highlights" not in text

    def test_truncates_to_five_items(self):
        summary = TranscriptSummary(
            qa_highlights=[f"highlight {i}" for i in range(10)],
            key_quotes=[f"quote {i}" for i in range(10)],
        )
        text = _format_transcript_summary(summary)
        # Should only show 5 of each
        assert text.count("highlight") == 5
        assert text.count("quote") == 5


# ── Format evidence includes transcript section ───────────────────────────


class TestFormatEvidenceWithTranscript:
    def test_includes_transcript_section_when_summary_exists(self):
        state = _make_ticker_state()
        state.transcript_summary = TranscriptSummary(
            management_tone="bullish",
            revenue_discussion="Revenue grew 12%",
        )
        registry = SourceRegistry()

        text = _format_evidence_for_llm(state, registry)
        assert "## Earnings Call Highlights" in text
        assert "bullish" in text
        assert "Revenue grew 12%" in text

    def test_no_transcript_section_when_no_summary(self):
        state = _make_ticker_state()
        registry = SourceRegistry()

        text = _format_evidence_for_llm(state, registry)
        assert "Earnings Call Highlights" not in text
