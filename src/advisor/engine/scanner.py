"""Live signal scanner - runs strategies on recent data to produce signals."""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta

import backtrader as bt

from advisor.core.enums import StrategyType
from advisor.data.feeds import create_feed
from advisor.data.yahoo import YahooDataProvider
from advisor.engine.signals import ScanResult, SignalAction, StrategySignal
from advisor.strategies.registry import StrategyRegistry

logger = logging.getLogger(__name__)

# Number of calendar days to fetch (gives ~200 trading days after weekends/holidays)
_LOOKBACK_DAYS = 300


class _SignalTracker(bt.Analyzer):
    """Lightweight analyzer that tracks the last order action and bar count."""

    def start(self):
        self.last_action = None  # "buy" or "sell"
        self.last_action_bar = -1
        self.total_bars = 0

    def notify_order(self, order):
        if order.status != order.Completed:
            return
        if order.isbuy():
            self.last_action = "buy"
        else:
            self.last_action = "sell"
        self.last_action_bar = self.total_bars

    def next(self):
        self.total_bars += 1

    def get_analysis(self):
        return {
            "last_action": self.last_action,
            "last_action_bar": self.last_action_bar,
            "total_bars": self.total_bars,
        }


class SignalScanner:
    """Scans symbols against registered strategies to produce live signals."""

    def __init__(self, provider: YahooDataProvider | None = None):
        self.provider = provider or YahooDataProvider(cache=None)

    def scan(
        self,
        symbol: str,
        strategy_names: list[str] | None = None,
    ) -> ScanResult:
        """Run strategies on recent data and return signals.

        Args:
            symbol: Ticker symbol to scan.
            strategy_names: Specific strategies to run. Defaults to all EQUITY strategies.
        """
        registry = StrategyRegistry()
        registry.discover()

        if strategy_names:
            strat_classes = [
                (name, registry.get_strategy(name)) for name in strategy_names
            ]
        else:
            # Default: equity strategies only
            strat_classes = [
                (name, registry.get_strategy(name))
                for name in registry.names
                if registry.get_strategy(name).strategy_type == StrategyType.EQUITY
            ]

        signals: list[StrategySignal] = []
        scanned_at = datetime.now()

        for name, strat_cls in strat_classes:
            signal = self._run_strategy(symbol, name, strat_cls, scanned_at)
            signals.append(signal)

        return ScanResult(symbol=symbol, scanned_at=scanned_at, signals=signals)

    def _run_strategy(
        self,
        symbol: str,
        strategy_name: str,
        strategy_cls: type,
        scanned_at: datetime,
    ) -> StrategySignal:
        """Run a single strategy through a mini-backtest and infer the signal."""
        end = date.today() + timedelta(days=1)
        start = end - timedelta(days=_LOOKBACK_DAYS)

        feed = create_feed(symbol, start, end, provider=self.provider)

        cerebro = bt.Cerebro()
        cerebro.adddata(feed, name=symbol)
        cerebro.addstrategy(strategy_cls)
        cerebro.addanalyzer(_SignalTracker, _name="signal_tracker")
        cerebro.broker.setcash(100_000.0)

        results = cerebro.run()
        strat = results[0]

        tracker = strat.analyzers.signal_tracker.get_analysis()
        has_position = bool(strat.position)
        last_action = tracker["last_action"]
        last_action_bar = tracker["last_action_bar"]
        total_bars = tracker["total_bars"]

        price = strat.data.close[0]

        action, reason = self._infer_signal(
            has_position, last_action, last_action_bar, total_bars
        )

        return StrategySignal(
            strategy_name=strategy_name,
            symbol=symbol,
            action=action,
            reason=reason,
            timestamp=scanned_at,
            price=price,
        )

    @staticmethod
    def _infer_signal(
        has_position: bool,
        last_action: str | None,
        last_action_bar: int,
        total_bars: int,
    ) -> tuple[SignalAction, str]:
        """Determine signal from strategy final state."""
        is_last_bar = last_action_bar == total_bars - 1

        if is_last_bar and last_action == "buy":
            return SignalAction.BUY, "Entered long position on last bar"
        if is_last_bar and last_action == "sell":
            return SignalAction.SELL, "Exited position on last bar"
        if has_position:
            return SignalAction.HOLD, "In long position"
        return SignalAction.NEUTRAL, "No position"
