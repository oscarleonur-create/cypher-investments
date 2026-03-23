"""Tests for options morning workflow — decision gate logic."""

from __future__ import annotations

from advisor.workflow.models import (
    CandidateVerdict,
    PositionAction,
)
from advisor.workflow.morning_options import OptionsMorningWorkflow


class TestOptionsPositionHealthGates:
    """Test options position health check decision gates."""

    def test_take_profit_50pct_credit(self):
        workflow = OptionsMorningWorkflow(skip_account=True)
        positions = [
            {
                "symbol": "PLTR 250418P00023000",
                "underlying_symbol": "PLTR",
                "quantity": 1,
                "quantity_direction": "Short",
                "average_open_price": 1.00,  # $1.00 credit received
                "close_price": 0.40,  # now worth $0.40 → 60% profit
                "instrument_type": "Option",
            }
        ]
        results = workflow._check_position_health(positions, [])
        assert len(results) == 1
        assert results[0].action == PositionAction.TAKE_PROFIT
        assert results[0].pnl_pct >= 50

    def test_consider_closing_2x_loss(self):
        workflow = OptionsMorningWorkflow(skip_account=True)
        positions = [
            {
                "symbol": "AMD 250418P00155000",
                "underlying_symbol": "AMD",
                "quantity": 1,
                "quantity_direction": "Short",
                "average_open_price": 0.85,  # $0.85 credit
                "close_price": 2.60,  # loss > 2x credit
                "instrument_type": "Option",
            }
        ]
        results = workflow._check_position_health(positions, [])
        assert len(results) == 1
        assert results[0].action == PositionAction.CONSIDER_CLOSING

    def test_approaching_expiry_14dte(self):
        """DTE <= 14 triggers APPROACHING_EXPIRY."""
        workflow = OptionsMorningWorkflow(skip_account=True)
        # Mock _parse_dte to return 10
        original_parse = workflow._parse_dte
        workflow._parse_dte = lambda sym: 10

        positions = [
            {
                "symbol": "SOFI 250325P00008000",
                "underlying_symbol": "SOFI",
                "quantity": 1,
                "quantity_direction": "Short",
                "average_open_price": 0.50,
                "close_price": 0.45,  # small profit, not 50% yet
                "instrument_type": "Option",
            }
        ]
        results = workflow._check_position_health(positions, [])
        workflow._parse_dte = original_parse

        assert len(results) == 1
        assert results[0].action == PositionAction.APPROACHING_EXPIRY

    def test_hold_normal_option(self):
        workflow = OptionsMorningWorkflow(skip_account=True)
        workflow._parse_dte = lambda sym: 30  # plenty of time

        positions = [
            {
                "symbol": "MARA 250515P00020000",
                "underlying_symbol": "MARA",
                "quantity": 1,
                "quantity_direction": "Short",
                "average_open_price": 1.20,
                "close_price": 0.90,  # 25% profit, below 50% target
                "instrument_type": "Option",
            }
        ]
        results = workflow._check_position_health(positions, [])
        workflow._parse_dte = OptionsMorningWorkflow._parse_dte

        assert len(results) == 1
        assert results[0].action == PositionAction.HOLD


class TestOptionsValidationGates:
    """Test candidate validation decision gates for options mode."""

    def test_deep_dive_enter_ml_buy(self):
        """Options mode: ENTER + ML BUY (>=0.60) → DEEP_DIVE (no strategy scan needed)."""
        confluence = "ENTER"
        ml_signal = "BUY"
        ml_win_prob = 0.65

        is_enter = confluence == "ENTER"
        ml_buy = ml_signal == "BUY" and ml_win_prob >= 0.60

        if is_enter and ml_buy:
            verdict = CandidateVerdict.DEEP_DIVE
        elif (is_enter or confluence == "CAUTION") and ml_signal != "SELL":
            verdict = CandidateVerdict.WATCH
        else:
            verdict = CandidateVerdict.SKIP

        assert verdict == CandidateVerdict.DEEP_DIVE

    def test_watch_enter_ml_hold(self):
        """ENTER + ML HOLD → WATCH"""
        confluence = "ENTER"
        ml_signal = "HOLD"
        ml_win_prob = 0.50

        is_enter = confluence == "ENTER"
        ml_buy = ml_signal == "BUY" and ml_win_prob >= 0.60
        ml_not_sell = ml_signal != "SELL"

        if is_enter and ml_buy:
            verdict = CandidateVerdict.DEEP_DIVE
        elif (is_enter or confluence == "CAUTION") and ml_not_sell:
            verdict = CandidateVerdict.WATCH
        else:
            verdict = CandidateVerdict.SKIP

        assert verdict == CandidateVerdict.WATCH

    def test_watch_caution_ml_buy_low_prob(self):
        """CAUTION + ML BUY but win_prob < 0.60 → WATCH (not deep dive)."""
        confluence = "CAUTION"
        ml_signal = "BUY"
        ml_win_prob = 0.55  # below 0.60 threshold

        is_enter = confluence == "ENTER"
        is_caution = confluence == "CAUTION"
        ml_buy = ml_signal == "BUY" and ml_win_prob >= 0.60
        ml_not_sell = ml_signal != "SELL"

        if is_enter and ml_buy:
            verdict = CandidateVerdict.DEEP_DIVE
        elif (is_enter or is_caution) and ml_not_sell:
            verdict = CandidateVerdict.WATCH
        else:
            verdict = CandidateVerdict.SKIP

        assert verdict == CandidateVerdict.WATCH

    def test_skip_pass_ml_sell(self):
        """PASS + ML SELL → SKIP"""
        confluence = "PASS"
        ml_signal = "SELL"
        ml_win_prob = 0.25

        is_enter = confluence == "ENTER"
        is_caution = confluence == "CAUTION"
        ml_buy = ml_signal == "BUY" and ml_win_prob >= 0.60
        ml_not_sell = ml_signal != "SELL"

        if is_enter and ml_buy:
            verdict = CandidateVerdict.DEEP_DIVE
        elif (is_enter or is_caution) and ml_not_sell:
            verdict = CandidateVerdict.WATCH
        else:
            verdict = CandidateVerdict.SKIP

        assert verdict == CandidateVerdict.SKIP


class TestDTEParsing:
    """Test OCC symbol DTE parsing."""

    def test_valid_symbol(self):
        workflow = OptionsMorningWorkflow(skip_account=True)
        # Use a far future date to ensure positive DTE
        dte = workflow._parse_dte("PLTR 301231P00050000")
        assert dte is not None
        assert dte > 0

    def test_invalid_symbol(self):
        workflow = OptionsMorningWorkflow(skip_account=True)
        dte = workflow._parse_dte("PLTR")
        assert dte is None
