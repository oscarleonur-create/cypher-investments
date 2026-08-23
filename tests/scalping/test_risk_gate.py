"""Unit tests for the risk gate (offline, deterministic)."""

from __future__ import annotations

from datetime import datetime

from advisor.risk.gate import RiskLimits, assess_signal, gate_signals
from advisor.scalping.models import ScalpAction, ScalpSignal


def make_signal(
    symbol: str = "TEST",
    entry: float = 100.0,
    stop: float = 99.0,
    target: float = 103.0,
    score: float = 80.0,
) -> ScalpSignal:
    return ScalpSignal(
        symbol=symbol,
        strategy="vwap_reversion",
        action=ScalpAction.LONG,
        reason="test",
        price=entry,
        entry=entry,
        stop=stop,
        target=target,
        score=score,
        bar_time=datetime(2026, 6, 15, 9, 30),
    )


def test_approves_and_sizes_within_risk_budget():
    sig = make_signal(entry=100.0, stop=99.0)  # $1 risk/share, R:R = 3
    limits = RiskLimits(max_symbol_exposure_pct=1.0, max_position_notional_pct=1.0)
    out = assess_signal(sig, net_liq=100_000.0, limits=limits)
    assert out.risk_approved is True
    # 0.5% of 100k = $500 risk budget / $1 per share = 500 shares, uncapped by exposure/notional
    assert out.risk_quantity == 500


def test_default_limits_cap_below_raw_risk_budget():
    sig = make_signal(entry=100.0, stop=99.0)  # would size to 500 shares on risk budget alone
    out = assess_signal(sig, net_liq=100_000.0)
    assert out.risk_approved is True
    # default max_symbol_exposure_pct=0.10 binds first: $10k / $100 = 100 shares
    assert out.risk_quantity == 100
    assert "max symbol exposure" in out.risk_note


def test_blocks_on_zero_net_liq():
    sig = make_signal()
    out = assess_signal(sig, net_liq=0.0)
    assert out.risk_approved is False
    assert out.risk_quantity == 0
    assert "no account data" in out.risk_note


def test_blocks_below_min_risk_reward():
    sig = make_signal(entry=100.0, stop=99.0, target=100.5)  # R:R = 0.5
    out = assess_signal(sig, net_liq=100_000.0)
    assert out.risk_approved is False
    assert "R:R" in out.risk_note


def test_caps_by_existing_symbol_exposure():
    sig = make_signal(entry=100.0, stop=99.0)
    limits = RiskLimits(max_symbol_exposure_pct=0.10)
    # Already $9,900 of $10,000 (10% of 100k) symbol cap used up.
    out = assess_signal(sig, net_liq=100_000.0, existing_symbol_notional=9_900.0, limits=limits)
    assert out.risk_quantity == 1  # only $100 of room left at $100/share
    assert "max symbol exposure" in out.risk_note


def test_gate_signals_sorts_approved_first():
    approved = make_signal(symbol="GOOD", entry=100.0, stop=99.0, score=10.0)
    blocked = make_signal(symbol="BAD", entry=100.0, stop=99.0, target=100.2, score=99.0)
    out = gate_signals([blocked, approved], net_liq=100_000.0)
    assert [s.symbol for s in out] == ["GOOD", "BAD"]
