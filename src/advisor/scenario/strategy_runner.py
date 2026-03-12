"""Strategy runner — executes Backtrader strategies on synthetic OHLCV feeds."""

from __future__ import annotations

import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
from typing import Any

import backtrader as bt
import numpy as np
import pandas as pd

from advisor.scenario.models import (
    PathStrategyResult,
    ScenarioConfig,
    StrategyScenarioResult,
)

logger = logging.getLogger(__name__)


class _SnapshotValue(bt.Analyzer):
    """Captures portfolio value at a specific bar index (start of forward period)."""

    params = (("snapshot_bar", 0),)

    def __init__(self):
        super().__init__()
        self._bar = 0
        self.snapshot = None
        self.max_dd_from_snapshot = 0.0
        self._peak = None

    def next(self):
        if self._bar == self.p.snapshot_bar:
            self.snapshot = self.strategy.broker.getvalue()
            self._peak = self.snapshot
        elif self._bar > self.p.snapshot_bar and self.snapshot is not None:
            val = self.strategy.broker.getvalue()
            if val > self._peak:
                self._peak = val
            dd = (self._peak - val) / self._peak * 100 if self._peak > 0 else 0.0
            if dd > self.max_dd_from_snapshot:
                self.max_dd_from_snapshot = dd
        self._bar += 1

    def get_analysis(self):
        return {
            "snapshot_value": self.snapshot,
            "max_dd_from_snapshot": self.max_dd_from_snapshot,
        }


def _run_single_path(
    df: pd.DataFrame,
    strategy_cls_name: str,
    initial_cash: float,
    strategy_params: dict[str, Any] | None = None,
    warmup_bars: int = 0,
) -> PathStrategyResult:
    """Run a single Backtrader strategy on one OHLCV DataFrame.

    warmup_bars: number of historical bars prepended for indicator warmup.
    Returns are measured from the start of the forward period (after warmup).
    """
    import logging as _logging

    from advisor.data.feeds import PandasFeed
    from advisor.engine.analyzers import TradeRecorder
    from advisor.strategies.registry import StrategyRegistry

    # Suppress noisy registry warnings in subprocesses
    _logging.getLogger("advisor.strategies.registry").setLevel(_logging.ERROR)
    registry = StrategyRegistry()
    registry.discover()
    strategy_cls = registry.get_strategy(strategy_cls_name)

    cerebro = bt.Cerebro()

    feed = PandasFeed(dataname=df)
    cerebro.adddata(feed, name="SIM")

    kwargs = strategy_params or {}
    cerebro.addstrategy(strategy_cls, **kwargs)

    cerebro.broker.setcash(initial_cash)
    cerebro.broker.setcommission(commission=0.001)
    cerebro.broker.set_slippage_perc(perc=0.001, slip_open=True, slip_match=True)

    cerebro.addanalyzer(bt.analyzers.DrawDown, _name="drawdown")
    cerebro.addanalyzer(bt.analyzers.TradeAnalyzer, _name="trades")
    cerebro.addanalyzer(TradeRecorder, _name="trade_recorder")
    cerebro.addanalyzer(_SnapshotValue, _name="snapshot", snapshot_bar=warmup_bars)

    try:
        results = cerebro.run()
    except Exception as e:
        logger.warning("Strategy %s failed on path: %s", strategy_cls_name, e)
        return PathStrategyResult(final_value=initial_cash)

    strat = results[0]
    final_value = strat.broker.getvalue()

    # Snapshot at start of forward period
    snap = strat.analyzers.snapshot.get_analysis()
    snapshot_value = snap.get("snapshot_value") or initial_cash
    max_dd_fwd = snap.get("max_dd_from_snapshot", 0.0)

    # Return measured from forward-period start, not from initial cash
    total_return_pct = ((final_value - snapshot_value) / snapshot_value) * 100

    # Trade stats
    trade_analysis = strat.analyzers.trades.get_analysis()
    total_trades = trade_analysis.get("total", {}).get("closed", 0)
    won = trade_analysis.get("won", {}).get("total", 0)
    win_rate = (won / total_trades * 100) if total_trades > 0 else None

    return PathStrategyResult(
        total_return_pct=total_return_pct,
        max_drawdown_pct=max_dd_fwd,
        total_trades=total_trades,
        win_rate=win_rate,
        final_value=final_value,
    )


def _is_streamlit() -> bool:
    """Detect if running inside Streamlit (multiprocessing is unsafe there)."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx

        return get_script_run_ctx() is not None
    except Exception:
        return False


def run_strategy_on_paths(
    feeds: list[pd.DataFrame],
    strategy_name: str,
    config: ScenarioConfig,
    strategy_params: dict[str, Any] | None = None,
    max_workers: int | None = None,
) -> list[PathStrategyResult]:
    """Run a strategy across multiple path feeds.

    Uses sequential execution in Streamlit (multiprocessing is unsafe there).
    Uses ProcessPoolExecutor with spawn context otherwise.
    """
    args = [
        (df, strategy_name, config.initial_cash, strategy_params, config.warmup_bars)
        for df in feeds
    ]

    workers = max_workers or min(4, len(feeds))

    # Force sequential in Streamlit — multiprocessing deadlocks/fails silently
    if workers <= 1 or len(feeds) <= 2 or _is_streamlit():
        return [_run_single_path(*a) for a in args]

    # Use "spawn" context to avoid fork() issues on macOS
    ctx = mp.get_context("spawn")
    results = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=ctx) as executor:
        futures = [executor.submit(_run_single_path, *a) for a in args]
        for f in futures:
            try:
                results.append(f.result())
            except Exception as e:
                logger.warning("Path execution failed: %s", e)
                results.append(PathStrategyResult(final_value=config.initial_cash))

    return results


def aggregate_path_results(
    path_results: list[PathStrategyResult],
    strategy_name: str,
    scenario_name: str,
) -> StrategyScenarioResult:
    """Aggregate per-path results into scenario-level statistics."""
    if not path_results:
        return StrategyScenarioResult(strategy_name=strategy_name, scenario_name=scenario_name)

    returns = np.array([r.total_return_pct for r in path_results])
    drawdowns = np.array([r.max_drawdown_pct for r in path_results])
    trades = np.array([r.total_trades for r in path_results])
    win_rates = [r.win_rate for r in path_results if r.win_rate is not None]

    return StrategyScenarioResult(
        strategy_name=strategy_name,
        scenario_name=scenario_name,
        n_paths=len(path_results),
        mean_return_pct=float(np.mean(returns)),
        median_return_pct=float(np.median(returns)),
        p5_return_pct=float(np.percentile(returns, 5)),
        p25_return_pct=float(np.percentile(returns, 25)),
        p75_return_pct=float(np.percentile(returns, 75)),
        p95_return_pct=float(np.percentile(returns, 95)),
        prob_positive=float(np.mean(returns > 0)),
        mean_max_dd_pct=float(np.mean(drawdowns)),
        median_max_dd_pct=float(np.median(drawdowns)),
        avg_trades=float(np.mean(trades)),
        avg_win_rate=float(np.mean(win_rates)) if win_rates else None,
    )
