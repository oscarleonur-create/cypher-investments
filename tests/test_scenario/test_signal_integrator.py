"""Tests for scenario signal integrator."""

from __future__ import annotations

import pytest
from advisor.scenario.models import BUILTIN_SCENARIOS, SignalContext
from advisor.scenario.signal_integrator import adjust_scenario_weights


class TestAdjustScenarioWeights:
    """Tests for scenario probability adjustment."""

    def _scenarios(self):
        return list(BUILTIN_SCENARIOS.values())

    def test_neutral_no_change(self):
        """NEUTRAL signal leaves base probabilities unchanged."""
        ctx = SignalContext(alpha_signal="NEUTRAL")
        weights = adjust_scenario_weights(self._scenarios(), ctx)

        assert weights["bull"] == pytest.approx(0.25, abs=0.001)
        assert weights["sideways"] == pytest.approx(0.45, abs=0.001)
        assert weights["bear"] == pytest.approx(0.20, abs=0.001)
        assert weights["crash"] == pytest.approx(0.10, abs=0.001)

    def test_no_signal_no_change(self):
        """No alpha signal leaves base probabilities unchanged."""
        ctx = SignalContext()
        weights = adjust_scenario_weights(self._scenarios(), ctx)

        assert weights["bull"] == pytest.approx(0.25, abs=0.001)
        assert sum(weights.values()) == pytest.approx(1.0, abs=0.001)

    def test_strong_buy_shifts_to_bull(self):
        """STRONG_BUY increases bull probability."""
        ctx = SignalContext(alpha_signal="STRONG_BUY")
        weights = adjust_scenario_weights(self._scenarios(), ctx)

        assert weights["bull"] > 0.25
        assert weights["bear"] < 0.20
        assert weights["crash"] < 0.10

    def test_avoid_shifts_to_bear(self):
        """AVOID increases bear/crash probability."""
        ctx = SignalContext(alpha_signal="AVOID")
        weights = adjust_scenario_weights(self._scenarios(), ctx)

        assert weights["bull"] < 0.25
        assert weights["bear"] > 0.20
        assert weights["crash"] > 0.10

    def test_weights_sum_to_one(self):
        """Adjusted weights always sum to 1.0 regardless of signal."""
        for signal in ["STRONG_BUY", "BUY", "LEAN_BUY", "NEUTRAL", "LEAN_SELL", "AVOID"]:
            ctx = SignalContext(alpha_signal=signal)
            weights = adjust_scenario_weights(self._scenarios(), ctx)
            assert sum(weights.values()) == pytest.approx(
                1.0, abs=0.001
            ), f"Weights don't sum to 1.0 for signal {signal}: {weights}"

    def test_weights_all_positive(self):
        """All adjusted weights are positive (no negative probabilities)."""
        for signal in ["STRONG_BUY", "AVOID"]:
            ctx = SignalContext(alpha_signal=signal)
            weights = adjust_scenario_weights(self._scenarios(), ctx)
            for name, w in weights.items():
                assert w > 0, f"Weight for {name} is {w} with signal {signal}"

    def test_confluence_enter_boosts_bull(self):
        """ENTER verdict further boosts bull probability."""
        ctx_no_conf = SignalContext(alpha_signal="STRONG_BUY")
        ctx_with_conf = SignalContext(alpha_signal="STRONG_BUY", confluence_verdict="ENTER")

        w1 = adjust_scenario_weights(self._scenarios(), ctx_no_conf)
        w2 = adjust_scenario_weights(self._scenarios(), ctx_with_conf)

        assert w2["bull"] > w1["bull"]

    def test_confluence_pass_reduces_bull(self):
        """PASS verdict further reduces bull probability."""
        ctx_no_conf = SignalContext(alpha_signal="AVOID")
        ctx_with_conf = SignalContext(alpha_signal="AVOID", confluence_verdict="PASS")

        w1 = adjust_scenario_weights(self._scenarios(), ctx_no_conf)
        w2 = adjust_scenario_weights(self._scenarios(), ctx_with_conf)

        assert w2["bull"] < w1["bull"]
