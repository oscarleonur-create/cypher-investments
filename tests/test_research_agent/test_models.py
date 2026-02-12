"""Tests for research_agent.models."""

from __future__ import annotations

from research_agent.models import (
    AgentState,
    ClassificationResult,
    DipType,
    EvidenceItem,
    FactPack,
    InputMode,
    KeyMetrics,
    OpportunityCard,
    ResearchInput,
    Source,
    TriggerResult,
    Verdict,
)


def test_research_input_run_id_deterministic():
    """run_id is deterministic for same input on same day."""
    inp = ResearchInput(mode=InputMode.TICKER, value="AAPL")
    assert inp.run_id() == inp.run_id()
    assert len(inp.run_id()) == 12


def test_research_input_different_values():
    """Different values produce different run IDs."""
    a = ResearchInput(mode=InputMode.TICKER, value="AAPL")
    b = ResearchInput(mode=InputMode.TICKER, value="MSFT")
    assert a.run_id() != b.run_id()


def test_enums():
    assert Verdict.BUY_THE_DIP == "BUY_THE_DIP"
    assert DipType.TEMPORARY == "TEMPORARY"
    assert InputMode.TICKER == "ticker"


def test_source_defaults():
    src = Source(url="https://example.com")
    assert src.tier == 3
    assert src.title == ""
    assert src.publisher == ""


def test_evidence_item():
    item = EvidenceItem(text="Revenue grew 10%", source_ids=["s1", "s2"])
    assert len(item.source_ids) == 2


def test_fact_pack_total_items():
    fp = FactPack(
        earnings_highlights=[EvidenceItem(text="a"), EvidenceItem(text="b")],
        guidance_changes=[EvidenceItem(text="c")],
    )
    assert fp.total_items == 3


def test_fact_pack_empty():
    fp = FactPack()
    assert fp.total_items == 0


def test_key_metrics_all_none():
    km = KeyMetrics()
    assert km.revenue_growth is None
    assert km.debt is None


def test_opportunity_card_construction():
    inp = ResearchInput(mode=InputMode.TICKER, value="AAPL")
    card = OpportunityCard(id="abc123", input=inp)
    assert card.verdict == Verdict.WATCH
    assert card.dip_type == DipType.UNCLEAR
    assert card.bull_case == []
    assert card.sources == []


def test_trigger_result_defaults():
    tr = TriggerResult()
    assert tr.found is False
    assert tr.summary == ""


def test_classification_result():
    cr = ClassificationResult(
        dip_type=DipType.TEMPORARY,
        confidence=0.85,
        reasoning="One-time event",
    )
    assert cr.confidence == 0.85


def test_agent_state_initial():
    inp = ResearchInput(mode=InputMode.TICKER, value="TSLA")
    state = AgentState(input=inp)
    assert state.iteration == 0
    assert state.trigger is None
    assert state.classification is None
    assert state.card is None
    assert state.fact_pack.total_items == 0


def test_opportunity_card_serialization():
    """Card can round-trip through JSON."""
    inp = ResearchInput(mode=InputMode.TICKER, value="AAPL")
    card = OpportunityCard(
        id="abc123",
        input=inp,
        verdict=Verdict.BUY_THE_DIP,
        dip_type=DipType.TEMPORARY,
        bull_case=["Strong earnings"],
        bear_case=["Macro risk"],
    )
    json_str = card.model_dump_json()
    restored = OpportunityCard.model_validate_json(json_str)
    assert restored.id == "abc123"
    assert restored.verdict == Verdict.BUY_THE_DIP
    assert restored.bull_case == ["Strong earnings"]
