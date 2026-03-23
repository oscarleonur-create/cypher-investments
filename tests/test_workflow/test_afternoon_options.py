"""Tests for options afternoon workflow — decision gate logic."""

from __future__ import annotations

from advisor.workflow.afternoon_options import OptionsAfternoonWorkflow
from advisor.workflow.models import (
    BrierScoreSummary,
    PositionAction,
    WatchCandidate,
)


class TestOptionsAfternoonPositionGates:
    """Test afternoon options position management gates."""

    def test_close_profit_target_hit(self):
        workflow = OptionsAfternoonWorkflow(skip_account=True, skip_validation=True)
        workflow._parse_dte = lambda sym: 30
        workflow._days_to_earnings = lambda sym: None

        positions = [
            {
                "symbol": "PLTR 250515P00023000",
                "underlying_symbol": "PLTR",
                "quantity": 1,
                "quantity_direction": "Short",
                "average_open_price": 1.00,
                "close_price": 0.40,  # 60% profit
            }
        ]
        results = workflow._check_positions(positions, [])
        assert len(results) == 1
        assert results[0].action == PositionAction.CLOSE
        assert results[0].limit_price is not None
        assert results[0].limit_price == 0.50  # 50% of credit

    def test_roll_low_dte(self):
        workflow = OptionsAfternoonWorkflow(skip_account=True, skip_validation=True)
        workflow._parse_dte = lambda sym: 5
        workflow._days_to_earnings = lambda sym: None

        positions = [
            {
                "symbol": "AMD 250321P00155000",
                "underlying_symbol": "AMD",
                "quantity": 1,
                "quantity_direction": "Short",
                "average_open_price": 0.85,
                "close_price": 0.60,  # 29% profit, not 50% target
            }
        ]
        results = workflow._check_positions(positions, [])
        assert len(results) == 1
        assert results[0].action == PositionAction.ROLL
        assert "roll" in results[0].reason.lower() or "DTE" in results[0].reason

    def test_close_before_earnings(self):
        workflow = OptionsAfternoonWorkflow(skip_account=True, skip_validation=True)
        workflow._parse_dte = lambda sym: 30
        workflow._days_to_earnings = lambda sym: 2

        positions = [
            {
                "symbol": "SOFI 250515P00008000",
                "underlying_symbol": "SOFI",
                "quantity": 1,
                "quantity_direction": "Short",
                "average_open_price": 0.50,
                "close_price": 0.45,  # small profit
            }
        ]
        results = workflow._check_positions(positions, [])
        assert len(results) == 1
        assert results[0].action == PositionAction.CLOSE
        assert "earnings" in results[0].reason.lower()

    def test_hold_normal_option(self):
        workflow = OptionsAfternoonWorkflow(skip_account=True, skip_validation=True)
        workflow._parse_dte = lambda sym: 35
        workflow._days_to_earnings = lambda sym: None

        positions = [
            {
                "symbol": "MARA 250515P00020000",
                "underlying_symbol": "MARA",
                "quantity": 1,
                "quantity_direction": "Short",
                "average_open_price": 1.20,
                "close_price": 0.90,  # 25% profit
            }
        ]
        results = workflow._check_positions(positions, [])
        assert len(results) == 1
        assert results[0].action == PositionAction.HOLD


class TestBrierScoreGate:
    """Test prediction validation Brier score gate."""

    def test_good_calibration_no_warning(self):
        summary = BrierScoreSummary(
            pop_brier=0.15,
            n_samples=50,
        )
        assert summary.warning is None

    def test_poor_calibration_warning(self):
        summary = BrierScoreSummary(
            pop_brier=0.30,
            n_samples=50,
            warning="MODEL CALIBRATION WARNING — POP Brier 0.300 > 0.25, consider retraining",
        )
        assert summary.warning is not None
        assert "CALIBRATION" in summary.warning

    def test_empty_samples(self):
        summary = BrierScoreSummary(n_samples=0)
        assert summary.pop_brier is None


class TestOptionsTomorrowWatchlist:
    """Test options-specific watchlist building."""

    def test_iv_rank_candidates_promoted(self):
        workflow = OptionsAfternoonWorkflow(skip_account=True, skip_validation=True)
        watch_updates = [
            WatchCandidate(symbol="MARA", status="IV rank 82 — good sell candidate"),
            WatchCandidate(symbol="SOFI", status="No change"),
            WatchCandidate(symbol="AMD", score_change=8.0, status="Score improved +8"),
        ]
        current = ["PLTR", "RIVN"]
        result = workflow._build_tomorrow_watchlist(watch_updates, current)
        assert result[0] == "MARA"  # IV rank promoted first
        assert "AMD" in result  # score improved
        assert "PLTR" in result
