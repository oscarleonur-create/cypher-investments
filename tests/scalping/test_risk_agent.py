"""Tests for the Risk agent's advisory narrowing layer.

The invariant that matters most, since this drives real position sizing:
the LLM structurally cannot raise the quantity the deterministic gate
computed, or flip a blocked signal to approved -- even when adversarial.
"""

from __future__ import annotations

from advisor.agent.llm import AgentLLM
from advisor.risk.agent import RiskAgentVerdict, review_signal, review_signals
from advisor.risk.gate import assess_signal

from tests.scalping.test_risk_gate import make_signal


def test_risk_agent_verdict_schema_cannot_express_approval():
    fields = set(RiskAgentVerdict.model_fields)
    assert not ({"approved", "risk_approved", "quantity", "risk_quantity"} & fields)


def test_llm_unavailable_falls_back_to_untouched_ceiling(monkeypatch):
    sig = make_signal(entry=100.0, stop=99.0)
    monkeypatch.setattr(AgentLLM, "configured", property(lambda self: False))
    out = review_signal(sig, net_liq=100_000.0)
    ceiling = assess_signal(sig, net_liq=100_000.0)
    assert out.risk_approved == ceiling.risk_approved
    assert out.risk_quantity == ceiling.risk_quantity


def test_already_blocked_signal_skips_the_llm_call_entirely(monkeypatch):
    sig = make_signal(entry=100.0, stop=99.0, target=100.5)  # R:R too low, blocked by gate.py
    called = []
    monkeypatch.setattr(AgentLLM, "configured", property(lambda self: True))
    monkeypatch.setattr(
        "advisor.risk.agent._narrow_with_llm",
        lambda *a, **k: called.append(1) or RiskAgentVerdict(),
    )
    out = review_signal(sig, net_liq=100_000.0)
    assert out.risk_approved is False
    assert called == []  # no LLM call was made for an already-blocked signal


def test_llm_can_never_raise_quantity_or_flip_approval(monkeypatch):
    sig = make_signal(entry=100.0, stop=99.0)  # gate.py alone: approved, quantity=100
    ceiling = assess_signal(sig, net_liq=100_000.0)
    assert ceiling.risk_approved is True
    assert ceiling.risk_quantity == 100

    monkeypatch.setattr(AgentLLM, "configured", property(lambda self: True))
    monkeypatch.setattr(
        "advisor.risk.agent._narrow_with_llm",
        lambda *a, **k: RiskAgentVerdict(narrow_quantity=99_999, veto=False, reasoning="yolo"),
    )

    out = review_signal(sig, net_liq=100_000.0)
    assert out.risk_quantity == 100  # clamped to the ceiling, never raised
    assert out.risk_approved is True


def test_llm_can_narrow_quantity_down(monkeypatch):
    sig = make_signal(entry=100.0, stop=99.0)
    monkeypatch.setattr(AgentLLM, "configured", property(lambda self: True))
    monkeypatch.setattr(
        "advisor.risk.agent._narrow_with_llm",
        lambda *a, **k: RiskAgentVerdict(narrow_quantity=20, reasoning="earnings in 2 days"),
    )
    out = review_signal(sig, net_liq=100_000.0)
    assert out.risk_quantity == 20
    assert out.risk_approved is True
    assert "narrowed" in out.risk_note
    assert "earnings in 2 days" in out.risk_note


def test_llm_veto_blocks_an_otherwise_approved_signal(monkeypatch):
    sig = make_signal(entry=100.0, stop=99.0)
    monkeypatch.setattr(AgentLLM, "configured", property(lambda self: True))
    monkeypatch.setattr(
        "advisor.risk.agent._narrow_with_llm",
        lambda *a, **k: RiskAgentVerdict(veto=True, reasoning="fresh negative headline"),
    )
    out = review_signal(sig, net_liq=100_000.0)
    assert out.risk_approved is False
    assert out.risk_quantity == 0
    assert "veto" in out.risk_note


def test_negative_narrow_quantity_clamped_to_zero_not_negative(monkeypatch):
    sig = make_signal(entry=100.0, stop=99.0)
    monkeypatch.setattr(AgentLLM, "configured", property(lambda self: True))
    monkeypatch.setattr(
        "advisor.risk.agent._narrow_with_llm",
        lambda *a, **k: RiskAgentVerdict(narrow_quantity=-5, reasoning="malformed"),
    )
    out = review_signal(sig, net_liq=100_000.0)
    assert out.risk_quantity == 0
    assert out.risk_approved is False


def test_narrowing_failure_falls_back_to_untouched_ceiling(monkeypatch):
    sig = make_signal(entry=100.0, stop=99.0)
    ceiling = assess_signal(sig, net_liq=100_000.0)
    monkeypatch.setattr(AgentLLM, "configured", property(lambda self: True))

    def _boom(*a, **k):
        raise RuntimeError("LLM call failed")

    monkeypatch.setattr("advisor.risk.agent._narrow_with_llm", _boom)
    out = review_signal(sig, net_liq=100_000.0)
    assert out.risk_approved == ceiling.risk_approved
    assert out.risk_quantity == ceiling.risk_quantity


def test_review_signals_use_llm_false_matches_gate_signals_exactly(monkeypatch):
    sigs = [make_signal(symbol="AAA", entry=100.0, stop=99.0)]
    monkeypatch.setattr(
        "advisor.risk.agent._narrow_with_llm",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not be called")),
    )
    out = review_signals(sigs, net_liq=100_000.0, use_llm=False)
    assert out[0].risk_quantity == 100  # identical to gate_signals() alone


def test_review_signals_use_llm_true_applies_narrowing_per_signal(monkeypatch):
    sigs = [
        make_signal(symbol="AAA", entry=100.0, stop=99.0),
        make_signal(symbol="BBB", entry=100.0, stop=99.0),
    ]
    monkeypatch.setattr(AgentLLM, "configured", property(lambda self: True))
    monkeypatch.setattr(
        "advisor.risk.agent._narrow_with_llm",
        lambda *a, **k: RiskAgentVerdict(narrow_quantity=10, reasoning="cap"),
    )
    out = review_signals(sigs, net_liq=100_000.0, use_llm=True)
    assert {s.risk_quantity for s in out} == {10}
