"""Backtest runner - wraps Backtrader's Cerebro."""

from __future__ import annotations

import logging
import uuid
from datetime import date
from typing import Any

import backtrader as bt

from advisor.data.feeds import create_feed
from advisor.data.yahoo import YahooDataProvider
from advisor.engine.analyzers import TradeRecorder
from advisor.engine.results import BacktestResult, TradeRecord
from advisor.strategies.base import StrategyBase
from advisor.strategies.registry import StrategyRegistry

logger = logging.getLogger(__name__)


class BacktestRunner:
    """Runs backtests using Backtrader."""

    def __init__(
        self,
        initial_cash: float = 100_000.0,
        commission: float = 0.001,
        provider: YahooDataProvider | None = None,
    ):
        self.initial_cash = initial_cash
        self.commission = commission
        self.provider = provider or YahooDataProvider()

    def run(
        self,
        strategy_name: str,
        symbol: str,
        start: date,
        end: date,
        params: dict[str, Any] | None = None,
        interval: str = "1d",
    ) -> BacktestResult:
        """Execute a backtest and return results."""
        registry = StrategyRegistry()
        strategy_cls = registry.get_strategy(strategy_name)

        cerebro = bt.Cerebro()

        # Add data feed
        feed = create_feed(symbol, start, end, provider=self.provider, interval=interval)
        cerebro.adddata(feed, name=symbol)

        # Add strategy with params
        kwargs = params or {}
        cerebro.addstrategy(strategy_cls, **kwargs)

        # Configure broker
        cerebro.broker.setcash(self.initial_cash)
        cerebro.broker.setcommission(commission=self.commission)

        # Add analyzers
        cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name="sharpe", riskfreerate=0.04)
        cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
        cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
        cerebro.addanalyzer(TradeRecorder, _name="trade_recorder")

        # Run
        logger.info(f"Running backtest: {strategy_name} on {symbol} ({start} to {end})")
        results = cerebro.run()
        strat = results[0]

        # Extract results
        return self._build_result(
            strat=strat,
            strategy_name=strategy_name,
            symbol=symbol,
            start=start,
            end=end,
            params=kwargs,
            interval=interval,
        )

    def _build_result(
        self,
        strat: StrategyBase,
        strategy_name: str,
        symbol: str,
        start: date,
        end: date,
        params: dict[str, Any],
        interval: str = "1d",
    ) -> BacktestResult:
        """Extract analyzer data into a BacktestResult."""
        final_value = strat.broker.getvalue()

        # Sharpe ratio
        sharpe_analysis = strat.analyzers.sharpe.get_analysis()
        sharpe = sharpe_analysis.get("sharperatio")

        # Drawdown
        dd_analysis = strat.analyzers.drawdown.get_analysis()
        max_dd = dd_analysis.get("max", {}).get("moneydown", 0.0)
        max_dd_pct = dd_analysis.get("max", {}).get("drawdown", 0.0)

        # Trade stats
        trade_analysis = strat.analyzers.trades.get_analysis()
        total_trades = trade_analysis.get("total", {}).get("closed", 0)
        won = trade_analysis.get("won", {}).get("total", 0)
        lost = trade_analysis.get("lost", {}).get("total", 0)

        avg_pnl = None
        if total_trades > 0:
            total_pnl = trade_analysis.get("pnl", {}).get("net", {}).get("total", 0.0)
            avg_pnl = total_pnl / total_trades

        # Trade records
        recorder_analysis = strat.analyzers.trade_recorder.get_analysis()
        trade_records = [
            TradeRecord(**t) for t in recorder_analysis.get("trades", [])
        ]

        result = BacktestResult(
            run_id=str(uuid.uuid4())[:8],
            strategy_name=strategy_name,
            symbol=symbol,
            start_date=str(start),
            end_date=str(end),
            initial_cash=self.initial_cash,
            final_value=final_value,
            sharpe_ratio=sharpe,
            max_drawdown=max_dd,
            max_drawdown_pct=max_dd_pct,
            total_trades=total_trades,
            winning_trades=won,
            losing_trades=lost,
            avg_trade_pnl=avg_pnl,
            trades=trade_records,
            interval=interval,
            params=params,
        )
        result.compute_derived()
        return result
