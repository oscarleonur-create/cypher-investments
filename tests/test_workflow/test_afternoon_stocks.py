"""Tests for stocks afternoon workflow — decision gate logic."""

from __future__ import annotations

from datetime import date, timedelta

from advisor.workflow.afternoon_stocks import StocksAfternoonWorkflow
from advisor.workflow.models import (
    EarningsAlert,
    PositionAction,
    WatchCandidate,
    WorkflowState,
)


class TestAfternoonPositionGates:
    """Test afternoon equity position management gates."""

    def test_earnings_warning_within_7_days(self):
        """Positions with earnings within 7 days get EARNINGS_WARNING."""
        workflow = StocksAfternoonWorkflow(skip_account=True)
        # Override earnings check to return 3 days
        workflow._days_to_earnings = lambda sym: 3

        positions = [
            {
                "symbol": "AAPL",
                "underlying_symbol": "AAPL",
                "quantity": 10,
                "average_open_price": 180.0,
                "close_price": 185.0,
            }
        ]

        # Patch yfinance to avoid network
        from unittest.mock import MagicMock, patch

        with patch("yfinance.Ticker") as mock_ticker:
            mock_hist = MagicMock()
            mock_hist.__len__ = lambda s: 0
            mock_ticker.return_value.history.return_value = mock_hist
            results = workflow._check_positions(positions, [])

        assert len(results) == 1
        assert results[0].action == PositionAction.EARNINGS_WARNING
        assert "3 days" in results[0].reason


class TestTomorrowWatchlist:
    """Test watchlist building for next day."""

    def test_promoted_candidates_first(self):
        workflow = StocksAfternoonWorkflow(skip_account=True)
        watch_updates = [
            WatchCandidate(symbol="AMD", new_buy_signals=["momentum_breakout"], status="NEW BUY"),
            WatchCandidate(symbol="SOFI", new_buy_signals=[], status="No change"),
        ]
        current = ["NVDA", "PLTR"]
        result = workflow._build_tomorrow_watchlist(watch_updates, current)
        assert result[0] == "AMD"  # promoted first
        assert "NVDA" in result
        assert "PLTR" in result

    def test_max_20_items(self):
        workflow = StocksAfternoonWorkflow(skip_account=True)
        watch_updates = []
        current = [f"SYM{i}" for i in range(30)]
        result = workflow._build_tomorrow_watchlist(watch_updates, current)
        assert len(result) <= 20


class TestEarningsAlerts:
    """Test earnings alert generation."""

    def test_earnings_alert_with_position(self):
        alert = EarningsAlert(
            symbol="AAPL",
            earnings_date=str(date.today() + timedelta(days=2)),
            days_until=2,
            has_position=True,
        )
        assert alert.has_position is True
        assert alert.days_until == 2

    def test_earnings_alert_no_position(self):
        alert = EarningsAlert(
            symbol="MSFT",
            earnings_date=str(date.today() + timedelta(days=5)),
            days_until=5,
            has_position=False,
        )
        assert alert.has_position is False


class TestWorkflowState:
    """Test workflow state persistence."""

    def test_roundtrip(self, tmp_path):
        """State saves and loads correctly."""

        import advisor.workflow.models as m

        original_path = m._STATE_PATH
        m._STATE_PATH = tmp_path / "workflow_state.json"

        try:
            state = WorkflowState(
                mode="stocks",
                watchlist=["AMD", "NVDA"],
                watch_candidates=["SOFI"],
                morning_alpha_scores={"AMD": 75.0, "NVDA": 82.0},
            )
            state.save()

            loaded = WorkflowState.load()
            assert loaded.mode == "stocks"
            assert loaded.watchlist == ["AMD", "NVDA"]
            assert loaded.watch_candidates == ["SOFI"]
            assert loaded.morning_alpha_scores["AMD"] == 75.0
        finally:
            m._STATE_PATH = original_path

    def test_load_missing_file(self, tmp_path):
        """Missing file returns default state."""
        import advisor.workflow.models as m

        original_path = m._STATE_PATH
        m._STATE_PATH = tmp_path / "nonexistent.json"

        try:
            state = WorkflowState.load()
            assert state.mode == ""
            assert state.watchlist == []
        finally:
            m._STATE_PATH = original_path
