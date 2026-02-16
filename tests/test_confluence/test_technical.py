"""Tests for the technical confluence agent."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

from advisor.confluence.technical import check_technical
from advisor.engine.signals import ScanResult, SignalAction, StrategySignal

_INDICATORS = {"price": 155.0, "sma_20": 148.50, "volume_ratio": 1.85}


class TestCheckTechnical:
    def _make_scan_result(self, action: SignalAction, price: float = 150.0) -> ScanResult:
        return ScanResult(
            symbol="AAPL",
            scanned_at=datetime.now(),
            signals=[
                StrategySignal(
                    strategy_name="momentum_breakout",
                    symbol="AAPL",
                    action=action,
                    reason="test",
                    timestamp=datetime.now(),
                    price=price,
                )
            ],
        )

    @patch("advisor.confluence.technical._compute_indicators", return_value=_INDICATORS)
    @patch("advisor.confluence.technical.SignalScanner")
    def test_buy_signal_is_bullish(self, mock_scanner_cls, mock_indicators):
        mock_scanner = MagicMock()
        mock_scanner.scan.return_value = self._make_scan_result(SignalAction.BUY, 155.0)
        mock_scanner_cls.return_value = mock_scanner

        result = check_technical("AAPL")

        assert result.is_bullish is True
        assert result.signal == "BUY"
        assert result.price == 155.0
        assert result.sma_20 == 148.50
        assert result.volume_ratio == 1.85
        mock_scanner.scan.assert_called_once_with("AAPL", strategy_names=["momentum_breakout"])

    @patch("advisor.confluence.technical._compute_indicators", return_value=_INDICATORS)
    @patch("advisor.confluence.technical.SignalScanner")
    def test_hold_signal_not_bullish(self, mock_scanner_cls, mock_indicators):
        mock_scanner = MagicMock()
        mock_scanner.scan.return_value = self._make_scan_result(SignalAction.HOLD)
        mock_scanner_cls.return_value = mock_scanner

        result = check_technical("AAPL")

        assert result.is_bullish is False
        assert result.signal == "HOLD"

    @patch("advisor.confluence.technical._compute_indicators", return_value=_INDICATORS)
    @patch("advisor.confluence.technical.SignalScanner")
    def test_neutral_signal_not_bullish(self, mock_scanner_cls, mock_indicators):
        mock_scanner = MagicMock()
        mock_scanner.scan.return_value = self._make_scan_result(SignalAction.NEUTRAL)
        mock_scanner_cls.return_value = mock_scanner

        result = check_technical("AAPL")

        assert result.is_bullish is False
        assert result.signal == "NEUTRAL"

    @patch("advisor.confluence.technical._compute_indicators", return_value=_INDICATORS)
    @patch("advisor.confluence.technical.SignalScanner")
    def test_sell_signal_not_bullish(self, mock_scanner_cls, mock_indicators):
        mock_scanner = MagicMock()
        mock_scanner.scan.return_value = self._make_scan_result(SignalAction.SELL)
        mock_scanner_cls.return_value = mock_scanner

        result = check_technical("AAPL")

        assert result.is_bullish is False
        assert result.signal == "SELL"

    @patch("advisor.confluence.technical.SignalScanner")
    def test_empty_signals_not_bullish(self, mock_scanner_cls):
        mock_scanner = MagicMock()
        mock_scanner.scan.return_value = ScanResult(
            symbol="AAPL", scanned_at=datetime.now(), signals=[]
        )
        mock_scanner_cls.return_value = mock_scanner

        result = check_technical("AAPL")

        assert result.is_bullish is False
        assert result.signal == "NEUTRAL"
        assert result.price == 0.0

    @patch(
        "advisor.confluence.technical._compute_indicators",
        return_value={"price": 0.0, "sma_20": 0.0, "volume_ratio": 0.0},
    )
    @patch("advisor.confluence.technical.SignalScanner")
    def test_indicator_failure_returns_zeros(self, mock_scanner_cls, mock_indicators):
        """If indicator computation fails, we still get the signal."""
        mock_scanner = MagicMock()
        mock_scanner.scan.return_value = self._make_scan_result(SignalAction.BUY, 150.0)
        mock_scanner_cls.return_value = mock_scanner

        result = check_technical("AAPL")

        assert result.is_bullish is True
        assert result.sma_20 == 0.0
        assert result.volume_ratio == 0.0

    @patch("advisor.confluence.technical._compute_indicators", return_value=_INDICATORS)
    @patch("advisor.confluence.technical.SignalScanner")
    def test_check_technical_with_custom_strategy(self, mock_scanner_cls, mock_indicators):
        """Strategy name is forwarded to the scanner."""
        mock_scanner = MagicMock()
        mock_scanner.scan.return_value = self._make_scan_result(SignalAction.BUY, 155.0)
        mock_scanner_cls.return_value = mock_scanner

        check_technical("AAPL", strategy_name="sma_crossover")

        mock_scanner.scan.assert_called_once_with("AAPL", strategy_names=["sma_crossover"])
