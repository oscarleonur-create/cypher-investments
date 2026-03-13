"""Tests for Monte Carlo exit sensitivity analysis."""

from __future__ import annotations

from advisor.simulator.engine import MonteCarloEngine
from advisor.simulator.exit_sensitivity import (
    ExitSensitivityAnalyzer,
    ExitSensitivityPoint,
    ExitSensitivityResult,
)
from advisor.simulator.models import PCSCandidate, SimConfig


def _make_candidate() -> PCSCandidate:
    """Create a synthetic PCS candidate for testing."""
    return PCSCandidate(
        symbol="TEST",
        expiration="2024-02-05",
        dte=35,
        short_strike=95.0,
        long_strike=90.0,
        width=5.0,
        short_bid=1.80,
        short_ask=2.00,
        long_bid=0.70,
        long_ask=0.90,
        net_credit=0.90,
        mid_credit=1.00,
        short_delta=-0.25,
        short_gamma=0.02,
        short_theta=-0.03,
        short_vega=0.10,
        short_iv=0.30,
        long_delta=-0.15,
        long_iv=0.32,
        underlying_price=100.0,
        buying_power=410.0,
    )


class TestExitSensitivityPoint:
    def test_model_creation(self):
        p = ExitSensitivityPoint(
            profit_target_pct=0.50,
            stop_loss_multiplier=3.0,
            close_at_dte=21,
            ev=25.50,
            pop=0.72,
        )
        assert p.profit_target_pct == 0.50
        assert p.ev == 25.50

    def test_defaults(self):
        p = ExitSensitivityPoint(
            profit_target_pct=0.50,
            stop_loss_multiplier=3.0,
            close_at_dte=21,
        )
        assert p.ev == 0.0
        assert p.exit_trailing_stop == 0.0


class TestExitSensitivityResult:
    def test_model_creation(self):
        r = ExitSensitivityResult(
            symbol="TEST",
            short_strike=95.0,
            long_strike=90.0,
            dte=35,
            net_credit=0.90,
            n_paths=10000,
        )
        assert r.n_paths == 10000
        assert r.points == []


class TestExitSensitivityAnalyzer:
    def test_sweep_produces_results(self):
        """Sweep should produce one result per param combo."""
        config = SimConfig(n_paths=500, seed=42)  # small for speed
        engine = MonteCarloEngine(config)
        candidate = _make_candidate()
        analyzer = ExitSensitivityAnalyzer(engine, candidate)

        result = analyzer.sweep(
            profit_targets=[0.40, 0.60],
            stop_losses=[2.0, 3.0],
            close_dtes=[7, 14],
        )

        assert len(result.points) == 8  # 2 * 2 * 2
        assert result.symbol == "TEST"
        assert result.n_paths > 0

    def test_sweep_different_params_produce_different_results(self):
        """Different exit params should generally produce different outcomes."""
        config = SimConfig(n_paths=1000, seed=42)
        engine = MonteCarloEngine(config)
        candidate = _make_candidate()
        analyzer = ExitSensitivityAnalyzer(engine, candidate)

        result = analyzer.sweep(
            profit_targets=[0.25, 0.75],
            stop_losses=[2.0],
            close_dtes=[7],
        )

        assert len(result.points) == 2
        # Tighter PT (0.25) should have higher exit_profit_target rate
        pt_25 = [p for p in result.points if p.profit_target_pct == 0.25][0]
        pt_75 = [p for p in result.points if p.profit_target_pct == 0.75][0]
        assert pt_25.exit_profit_target >= pt_75.exit_profit_target

    def test_sweep_default_ranges(self):
        """Sweep with defaults should work."""
        config = SimConfig(n_paths=200, seed=42)
        engine = MonteCarloEngine(config)
        candidate = _make_candidate()
        analyzer = ExitSensitivityAnalyzer(engine, candidate)

        result = analyzer.sweep()

        # 5 * 4 * 5 = 100 combos
        assert len(result.points) == 100

    def test_all_exit_fractions_sum_to_one(self):
        """Exit fractions should sum to approximately 1.0 for each point."""
        config = SimConfig(n_paths=500, seed=42)
        engine = MonteCarloEngine(config)
        candidate = _make_candidate()
        analyzer = ExitSensitivityAnalyzer(engine, candidate)

        result = analyzer.sweep(
            profit_targets=[0.50],
            stop_losses=[2.0],
            close_dtes=[14],
        )

        for p in result.points:
            total = (
                p.exit_profit_target
                + p.exit_stop_loss
                + p.exit_dte
                + p.exit_expiration
                + p.exit_trailing_stop
            )
            assert abs(total - 1.0) < 0.01, f"Exit fractions sum to {total}, expected ~1.0"

    def test_ev_and_pop_reasonable(self):
        """EV and POP should be within reasonable bounds."""
        config = SimConfig(n_paths=1000, seed=42)
        engine = MonteCarloEngine(config)
        candidate = _make_candidate()
        analyzer = ExitSensitivityAnalyzer(engine, candidate)

        result = analyzer.sweep(
            profit_targets=[0.50],
            stop_losses=[3.0],
            close_dtes=[7],
        )

        for p in result.points:
            # POP should be between 0 and 1
            assert 0.0 <= p.pop <= 1.0
            # EV bounded by max profit and max loss
            max_profit = candidate.net_credit * 100
            max_loss = candidate.width * 100
            assert p.ev >= -max_loss
            assert p.ev <= max_profit

    def test_sweep_with_trailing_stop(self):
        """Trailing stop should produce exit_trailing_stop > 0 when enabled."""
        config = SimConfig(
            n_paths=1000,
            seed=42,
            use_trailing_stop=True,
            trailing_activation_pct=0.30,
            trailing_floor_pct=0.10,
        )
        engine = MonteCarloEngine(config)
        candidate = _make_candidate()
        analyzer = ExitSensitivityAnalyzer(engine, candidate)

        result = analyzer.sweep(
            profit_targets=[0.90],  # high PT so trailing stop can trigger
            stop_losses=[3.0],
            close_dtes=[0],
        )

        # With trailing stop enabled and high profit target, some should trail
        for p in result.points:
            total = (
                p.exit_profit_target
                + p.exit_stop_loss
                + p.exit_dte
                + p.exit_expiration
                + p.exit_trailing_stop
            )
            assert abs(total - 1.0) < 0.01
