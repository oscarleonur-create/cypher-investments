"""Tests for the live signal scanner."""

from __future__ import annotations

from datetime import date
from typing import ClassVar

import numpy as np
import pandas as pd
from advisor.core.enums import StrategyType
from advisor.engine.scanner import SignalScanner
from advisor.engine.signals import SignalAction
from advisor.strategies.base import StrategyBase
from advisor.strategies.registry import StrategyRegistry

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_ohlcv(n: int = 100, start_price: float = 100.0, trend: str = "flat") -> pd.DataFrame:
    """Generate synthetic OHLCV data."""
    dates = pd.bdate_range(end=date.today(), periods=n)
    if trend == "up":
        close = start_price + np.linspace(0, 50, n)
    elif trend == "down":
        close = start_price - np.linspace(0, 50, n)
    else:
        close = np.full(n, start_price)

    # Add small noise
    rng = np.random.default_rng(42)
    noise = rng.normal(0, 0.5, n)
    close = close + noise

    df = pd.DataFrame(
        {
            "Open": close - 0.5,
            "High": close + 1.0,
            "Low": close - 1.0,
            "Close": close,
            "Volume": np.full(n, 1_000_000),
        },
        index=dates,
    )
    return df


def _mock_provider(df: pd.DataFrame):
    """Create a mock provider that returns the given DataFrame."""

    class FakeProvider:
        def get_stock_history(self, symbol, start, end, interval="1d"):
            return df.copy()

    return FakeProvider()


# ---------------------------------------------------------------------------
# Test strategies (registered per-test via the autouse reset_registry fixture)
# ---------------------------------------------------------------------------


def _register_always_buy():
    """Register a strategy that buys on the first bar and holds."""

    @StrategyRegistry.register
    class AlwaysBuy(StrategyBase):
        strategy_name: ClassVar[str] = "always_buy"
        strategy_type: ClassVar[StrategyType] = StrategyType.EQUITY
        description: ClassVar[str] = "Buys immediately and holds"

        def __init__(self):
            super().__init__()
            self.order = None
            self.bought = False

        def next(self):
            if self.order:
                return
            if not self.bought:
                self.order = self.buy(size=10)
                self.bought = True

        def notify_order(self, order):
            if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
                self.order = None

    return AlwaysBuy


def _register_never_buy():
    """Register a strategy that never trades."""

    @StrategyRegistry.register
    class NeverBuy(StrategyBase):
        strategy_name: ClassVar[str] = "never_buy"
        strategy_type: ClassVar[StrategyType] = StrategyType.EQUITY
        description: ClassVar[str] = "Never enters a position"

        def next(self):
            pass

    return NeverBuy


def _register_buy_last_bar():
    """Register a strategy that buys on the second-to-last bar.

    The buy order placed on bar N-2 fills on bar N-1 (the last bar),
    since backtrader uses next-bar execution.
    """

    @StrategyRegistry.register
    class BuyLastBar(StrategyBase):
        strategy_name: ClassVar[str] = "buy_last_bar"
        strategy_type: ClassVar[StrategyType] = StrategyType.EQUITY
        description: ClassVar[str] = "Buys near the end"

        def __init__(self):
            super().__init__()
            self.order = None

        def next(self):
            if self.order:
                return
            # Buy on second-to-last bar → fills on last bar
            if len(self.data) == self.data.buflen() - 1:
                self.order = self.buy(size=10)

        def notify_order(self, order):
            if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
                self.order = None

    return BuyLastBar


def _register_sell_last_bar():
    """Register a strategy that buys early and sells near the end.

    The sell order placed on bar N-2 fills on bar N-1 (the last bar),
    since backtrader uses next-bar execution.
    """

    @StrategyRegistry.register
    class SellLastBar(StrategyBase):
        strategy_name: ClassVar[str] = "sell_last_bar"
        strategy_type: ClassVar[StrategyType] = StrategyType.EQUITY
        description: ClassVar[str] = "Buys early, sells near the end"

        def __init__(self):
            super().__init__()
            self.order = None
            self.bought = False

        def next(self):
            if self.order:
                return
            if not self.bought:
                self.order = self.buy(size=10)
                self.bought = True
            elif len(self.data) == self.data.buflen() - 1 and self.position:
                # Sell on second-to-last bar → fills on last bar
                self.order = self.close()

        def notify_order(self, order):
            if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
                self.order = None

    return SellLastBar


def _register_options_strategy():
    """Register an OPTIONS-type strategy."""

    @StrategyRegistry.register
    class FakeOptions(StrategyBase):
        strategy_name: ClassVar[str] = "fake_options"
        strategy_type: ClassVar[StrategyType] = StrategyType.OPTIONS
        description: ClassVar[str] = "Fake options strategy"

        def next(self):
            pass

    return FakeOptions


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestSignalInference:
    """Test the static _infer_signal logic."""

    def test_buy_on_last_bar(self):
        action, reason = SignalScanner._infer_signal(
            has_position=True, last_action="buy", last_action_bar=99, total_bars=100
        )
        assert action == SignalAction.BUY
        assert "last bar" in reason.lower()

    def test_sell_on_last_bar(self):
        action, reason = SignalScanner._infer_signal(
            has_position=False, last_action="sell", last_action_bar=99, total_bars=100
        )
        assert action == SignalAction.SELL
        assert "last bar" in reason.lower()

    def test_in_position_hold(self):
        action, _ = SignalScanner._infer_signal(
            has_position=True, last_action="buy", last_action_bar=50, total_bars=100
        )
        assert action == SignalAction.HOLD

    def test_no_position_neutral(self):
        action, _ = SignalScanner._infer_signal(
            has_position=False, last_action=None, last_action_bar=-1, total_bars=100
        )
        assert action == SignalAction.NEUTRAL

    def test_no_position_old_sell_neutral(self):
        action, _ = SignalScanner._infer_signal(
            has_position=False, last_action="sell", last_action_bar=50, total_bars=100
        )
        assert action == SignalAction.NEUTRAL


class TestScannerIntegration:
    """Integration tests using synthetic data and mock strategies."""

    def test_hold_signal(self):
        """Strategy that buys early → HOLD."""
        _register_always_buy()
        df = _make_ohlcv(100)
        scanner = SignalScanner(provider=_mock_provider(df))
        result = scanner.scan("TEST", strategy_names=["always_buy"])

        assert len(result.signals) == 1
        assert result.signals[0].action == SignalAction.HOLD
        assert result.symbol == "TEST"

    def test_neutral_signal(self):
        """Strategy that never buys → NEUTRAL."""
        _register_never_buy()
        df = _make_ohlcv(100)
        scanner = SignalScanner(provider=_mock_provider(df))
        result = scanner.scan("TEST", strategy_names=["never_buy"])

        assert len(result.signals) == 1
        assert result.signals[0].action == SignalAction.NEUTRAL

    def test_buy_last_bar_signal(self):
        """Strategy that buys on the last bar → BUY."""
        _register_buy_last_bar()
        df = _make_ohlcv(100)
        scanner = SignalScanner(provider=_mock_provider(df))
        result = scanner.scan("TEST", strategy_names=["buy_last_bar"])

        assert len(result.signals) == 1
        assert result.signals[0].action == SignalAction.BUY

    def test_sell_last_bar_signal(self):
        """Strategy that sells on the last bar → SELL."""
        _register_sell_last_bar()
        df = _make_ohlcv(100)
        scanner = SignalScanner(provider=_mock_provider(df))
        result = scanner.scan("TEST", strategy_names=["sell_last_bar"])

        assert len(result.signals) == 1
        assert result.signals[0].action == SignalAction.SELL

    def test_equity_only_default(self):
        """Default scan excludes OPTIONS strategies."""
        _register_always_buy()
        _register_options_strategy()
        df = _make_ohlcv(100)
        scanner = SignalScanner(provider=_mock_provider(df))
        result = scanner.scan("TEST")

        strategy_names = [s.strategy_name for s in result.signals]
        assert "always_buy" in strategy_names
        assert "fake_options" not in strategy_names

    def test_explicit_strategy_override(self):
        """Explicit --strategy can target any strategy type."""
        _register_options_strategy()
        df = _make_ohlcv(100)
        scanner = SignalScanner(provider=_mock_provider(df))
        result = scanner.scan("TEST", strategy_names=["fake_options"])

        assert len(result.signals) == 1
        assert result.signals[0].strategy_name == "fake_options"

    def test_scan_result_summary(self):
        """ScanResult.summary aggregates counts correctly."""
        _register_always_buy()
        _register_never_buy()
        df = _make_ohlcv(100)
        scanner = SignalScanner(provider=_mock_provider(df))
        result = scanner.scan("TEST", strategy_names=["always_buy", "never_buy"])

        assert "1 HOLD" in result.summary
        assert "1 NEUTRAL" in result.summary

    def test_signal_has_price(self):
        """Each signal should have a non-zero price."""
        _register_always_buy()
        df = _make_ohlcv(100, start_price=150.0)
        scanner = SignalScanner(provider=_mock_provider(df))
        result = scanner.scan("TEST", strategy_names=["always_buy"])

        assert result.signals[0].price > 0
