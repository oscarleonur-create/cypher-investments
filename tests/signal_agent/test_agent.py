"""Tests for the Signal agent's tool-calling loop and result assembly."""

from __future__ import annotations

import json
from datetime import datetime
from types import SimpleNamespace

from advisor.agent.llm import AgentLLM
from advisor.scalping.models import ScalpAction, ScalpScanResult, ScalpSignal
from advisor.signal_agent.agent import run_signal_agent


def _tool_call_response(name: str, args: dict):
    tc = SimpleNamespace(
        id="call_1", function=SimpleNamespace(name=name, arguments=json.dumps(args))
    )
    message = SimpleNamespace(content="", tool_calls=[tc])
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _final_response(text: str):
    message = SimpleNamespace(content=text, tool_calls=None)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _signal(symbol: str) -> ScalpSignal:
    return ScalpSignal(
        symbol=symbol,
        strategy="vwap_reversion",
        action=ScalpAction.LONG,
        reason="test",
        price=100.0,
        entry=100.0,
        stop=99.0,
        target=103.0,
        score=80.0,
        bar_time=datetime(2026, 6, 15, 9, 30),
    )


def _mute_persistence(monkeypatch):
    """Signal-agent runs try to persist an audit row; keep tests offline."""
    fake_store = SimpleNamespace(save_agent_run=lambda *a, **k: "run123", close=lambda: None)
    monkeypatch.setattr("advisor.research.store.ResearchStore", lambda db_path: fake_store)


def test_llm_unavailable_returns_empty_result_with_note(monkeypatch):
    monkeypatch.setattr(AgentLLM, "configured", property(lambda self: False))
    result = run_signal_agent("find scalp setups")
    assert result.signals == []
    assert "unavailable" in result.notes.lower()


def test_signal_agent_returns_scan_output_not_llm_transcription(monkeypatch):
    _mute_persistence(monkeypatch)
    monkeypatch.setattr(AgentLLM, "configured", property(lambda self: True))

    calls = iter(
        [
            _tool_call_response("run_scalp_scan", {"symbols": ["AAA"]}),
            _final_response(
                json.dumps({"rationale": "AAA gapped on volume", "focus_symbols": ["AAA"]})
            ),
        ]
    )
    monkeypatch.setattr(AgentLLM, "chat_with_tools", lambda self, *a, **k: next(calls))
    monkeypatch.setattr(
        "advisor.scalping.scanner.ScalpScanner.scan",
        lambda self, *a, **k: ScalpScanResult(source="yfinance", signals=[_signal("AAA")]),
    )

    result = run_signal_agent("find scalp setups", universe="custom", net_liq=100_000.0)

    assert result.rationale == "AAA gapped on volume"
    assert result.focus_symbols == ["AAA"]
    assert any(s.symbol == "AAA" for s in result.signals)
    # The signal went through the real deterministic gate, not a pass-through.
    assert result.signals[0].risk_approved is True
    assert result.run_id == "run123"


def test_signal_agent_handles_unparseable_final_verdict_gracefully(monkeypatch):
    _mute_persistence(monkeypatch)
    monkeypatch.setattr(AgentLLM, "configured", property(lambda self: True))
    monkeypatch.setattr(
        AgentLLM, "chat_with_tools", lambda self, *a, **k: _final_response("not valid json")
    )
    result = run_signal_agent("find scalp setups")
    assert result.signals == []
    assert result.rationale == "not valid json"  # falls back to raw text, doesn't crash


def test_signal_agent_marks_budget_exhausted(monkeypatch):
    monkeypatch.setattr(AgentLLM, "configured", property(lambda self: True))
    # Every call returns a tool_call, never a final answer -> exhausts the loop.
    monkeypatch.setattr(
        AgentLLM,
        "chat_with_tools",
        lambda self, *a, **k: _tool_call_response("get_watchlist", {}),
    )
    monkeypatch.setattr(
        "advisor.research.store.ResearchStore",
        lambda db_path: SimpleNamespace(
            load_watchlist=lambda: [], save_agent_run=lambda *a, **k: "run1", close=lambda: None
        ),
    )
    result = run_signal_agent("find scalp setups")
    assert result.budget_exhausted is True
