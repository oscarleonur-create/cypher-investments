"""Unit tests for the Signal agent's tool registry (offline, no network)."""

from __future__ import annotations

from datetime import datetime

from advisor.scalping.models import ScalpAction, ScalpScanResult, ScalpSignal
from advisor.signal_agent.tools import SignalToolContext, make_dispatch, tool_specs


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


def test_tool_specs_returns_all_registered_tools():
    names = {s["function"]["name"] for s in tool_specs()}
    assert names == {
        "list_universe",
        "get_watchlist",
        "get_catalysts",
        "run_scalp_scan",
        "run_swing_scan",
    }


def test_dispatch_unknown_tool_returns_error():
    ctx = SignalToolContext()
    dispatch = make_dispatch(ctx)
    result = dispatch("not_a_real_tool", {})
    assert "error" in result


def test_list_universe_custom_dedupes_and_caps(monkeypatch):
    ctx = SignalToolContext(universe_cap=2)
    dispatch = make_dispatch(ctx)
    result = dispatch(
        "list_universe", {"universe": "custom", "symbols": ["aaa", "AAA", "bbb", "ccc"]}
    )
    assert result["symbols"] == ["AAA", "BBB"]  # deduped, capped at 2


def test_run_scalp_scan_collects_signals_on_context(monkeypatch):
    fake_result = ScalpScanResult(source="yfinance", signals=[_signal("AAA")])
    monkeypatch.setattr(
        "advisor.scalping.scanner.ScalpScanner.scan", lambda self, *a, **k: fake_result
    )
    ctx = SignalToolContext()
    dispatch = make_dispatch(ctx)
    result = dispatch("run_scalp_scan", {"symbols": ["AAA"], "strategy": "vwap_reversion"})
    assert result["source"] == "yfinance"
    assert len(ctx.collected_scalp) == 1
    assert ctx.collected_scalp[0].symbol == "AAA"


def test_run_scalp_scan_rejects_unknown_strategy():
    ctx = SignalToolContext()
    dispatch = make_dispatch(ctx)
    result = dispatch("run_scalp_scan", {"symbols": ["AAA"], "strategy": "not_a_strategy"})
    assert "error" in result


def test_run_swing_scan_collects_signals_on_context(monkeypatch):
    from types import SimpleNamespace

    fake_confluence_result = SimpleNamespace(
        verdict=SimpleNamespace(value="ENTER"),
        reasoning="synthetic",
        suggested_hold_days=5,
    )
    monkeypatch.setattr(
        "advisor.confluence.orchestrator.run_confluence", lambda sym, **k: fake_confluence_result
    )
    ctx = SignalToolContext()
    dispatch = make_dispatch(ctx)
    result = dispatch("run_swing_scan", {"symbols": ["AAA"]})
    assert result["signals"][0]["symbol"] == "AAA"
    assert len(ctx.collected_swing) == 1


def test_run_swing_scan_records_errors_without_crashing(monkeypatch):
    def _boom(sym, **k):
        raise RuntimeError("confluence failed")

    monkeypatch.setattr("advisor.confluence.orchestrator.run_confluence", _boom)
    ctx = SignalToolContext()
    dispatch = make_dispatch(ctx)
    result = dispatch("run_swing_scan", {"symbols": ["AAA"]})
    assert result["signals"] == []
    assert "AAA" in result["errors"][0]


def test_get_watchlist_wraps_research_store(monkeypatch):
    from types import SimpleNamespace

    fake_store = SimpleNamespace(
        load_watchlist=lambda: [{"symbol": "AAA", "note": "watching"}], close=lambda: None
    )
    monkeypatch.setattr("advisor.research.store.ResearchStore", lambda db_path: fake_store)
    ctx = SignalToolContext()
    dispatch = make_dispatch(ctx)
    result = dispatch("get_watchlist", {})
    assert result["watchlist"] == [{"symbol": "AAA", "note": "watching"}]
