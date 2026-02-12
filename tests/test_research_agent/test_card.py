"""Tests for research_agent.card (Markdown rendering)."""

from __future__ import annotations

from research_agent.card import build_partial_card, render_markdown
from research_agent.evidence import SourceRegistry
from research_agent.models import (
    AgentState,
    Catalyst,
    DipType,
    InputMode,
    KeyMetrics,
    OpportunityCard,
    ResearchInput,
    Source,
    TriggerResult,
    Verdict,
)


def _sample_card() -> OpportunityCard:
    return OpportunityCard(
        id="test123",
        input=ResearchInput(mode=InputMode.TICKER, value="AAPL"),
        verdict=Verdict.BUY_THE_DIP,
        catalyst=Catalyst(summary="Q1 earnings miss drove selloff"),
        dip_type=DipType.TEMPORARY,
        bull_case=[
            "Revenue growth 6% YoY [s1]",
            "Services growing 15% [s2]",
            "Buyback accelerated [s1]",
        ],
        bear_case=[
            "iPhone units down 8% QoQ [s3]",
            "CapEx up 20% [s1]",
        ],
        key_metrics=KeyMetrics(
            revenue_growth="6% YoY",
            margins="44% gross",
            fcf="$28B TTM",
            cash="$62B",
            debt="$108B",
            guidance_notes="Full-year guidance reiterated",
        ),
        risks=["Macro downturn", "Regulatory risk"],
        invalidation=["iPhone decline >15% Q2"],
        validation_checklist=["Q2 inventory data", "Services growth >14%"],
        next_actions=["Monitor App Store revenue", "Track channel checks"],
        sources=[
            Source(url="https://investor.apple.com", title="Apple Q1 Release", tier=1),
            Source(url="https://reuters.com/apple", title="Reuters Article", tier=2),
        ],
    )


class TestRenderMarkdown:
    def test_contains_ticker(self):
        md = render_markdown(_sample_card())
        assert "# Opportunity Card: AAPL" in md

    def test_contains_verdict(self):
        md = render_markdown(_sample_card())
        assert "BUY_THE_DIP" in md

    def test_contains_dip_type(self):
        md = render_markdown(_sample_card())
        assert "TEMPORARY" in md

    def test_contains_what_changed(self):
        md = render_markdown(_sample_card())
        assert "## What Changed" in md
        assert "Q1 earnings miss" in md

    def test_contains_bull_case(self):
        md = render_markdown(_sample_card())
        assert "## Bull Case" in md
        assert "Revenue growth 6%" in md

    def test_contains_bear_case(self):
        md = render_markdown(_sample_card())
        assert "## Bear Case" in md
        assert "iPhone units down" in md

    def test_contains_key_metrics_table(self):
        md = render_markdown(_sample_card())
        assert "| Revenue Growth | 6% YoY |" in md
        assert "| FCF | $28B TTM |" in md

    def test_contains_guidance(self):
        md = render_markdown(_sample_card())
        assert "Full-year guidance reiterated" in md

    def test_contains_validation_checklist(self):
        md = render_markdown(_sample_card())
        assert "- [ ] Q2 inventory data" in md

    def test_contains_invalidation(self):
        md = render_markdown(_sample_card())
        assert "iPhone decline >15% Q2" in md

    def test_contains_sources(self):
        md = render_markdown(_sample_card())
        assert "## Sources" in md
        assert "Apple Q1 Release" in md
        assert "Tier 1" in md

    def test_empty_sections_omitted(self):
        card = OpportunityCard(
            id="min",
            input=ResearchInput(mode=InputMode.TICKER, value="XYZ"),
        )
        md = render_markdown(card)
        assert "## Bull Case" not in md
        assert "## Bear Case" not in md


class TestBuildPartialCard:
    def test_builds_with_minimal_state(self):
        state = AgentState(
            input=ResearchInput(mode=InputMode.TICKER, value="AAPL"),
        )
        registry = SourceRegistry()
        card = build_partial_card(state, registry)
        assert card.verdict == Verdict.WATCH
        assert "manual review" in card.risks[0].lower()

    def test_builds_with_trigger(self):
        state = AgentState(
            input=ResearchInput(mode=InputMode.TICKER, value="AAPL"),
            trigger=TriggerResult(found=True, summary="Earnings miss"),
        )
        registry = SourceRegistry()
        card = build_partial_card(state, registry)
        assert card.catalyst.summary == "Earnings miss"
