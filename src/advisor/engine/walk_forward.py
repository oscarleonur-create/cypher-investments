"""Walk-forward testing — rolling train/test window analysis."""

from __future__ import annotations

import logging
import uuid
from datetime import date, timedelta
from typing import Any

from pydantic import BaseModel, Field

from advisor.engine.results import BacktestResult
from advisor.engine.runner import BacktestRunner

logger = logging.getLogger(__name__)


class WindowResult(BaseModel):
    window_index: int
    train_start: str
    train_end: str
    test_start: str
    test_end: str
    in_sample: BacktestResult
    out_of_sample: BacktestResult


class WalkForwardResult(BaseModel):
    run_id: str
    strategy: str
    symbol: str
    start_date: str
    end_date: str
    n_windows: int
    train_pct: float
    windows: list[WindowResult] = Field(default_factory=list)

    # Aggregate OOS metrics
    oos_avg_return_pct: float = 0.0
    oos_avg_sharpe: float | None = None
    oos_avg_max_dd_pct: float = 0.0

    # Overfitting indicator
    is_avg_return_pct: float = 0.0
    is_vs_oos_gap: float = 0.0

    def compute_aggregates(self) -> None:
        """Average OOS and IS metrics across windows, compute gap."""
        if not self.windows:
            return

        n = len(self.windows)

        oos_returns = [w.out_of_sample.total_return_pct for w in self.windows]
        self.oos_avg_return_pct = sum(oos_returns) / n

        oos_sharpes = [
            w.out_of_sample.sharpe_ratio
            for w in self.windows
            if w.out_of_sample.sharpe_ratio is not None
        ]
        self.oos_avg_sharpe = sum(oos_sharpes) / len(oos_sharpes) if oos_sharpes else None

        oos_dds = [
            w.out_of_sample.max_drawdown_pct or 0.0
            for w in self.windows
        ]
        self.oos_avg_max_dd_pct = sum(oos_dds) / n

        is_returns = [w.in_sample.total_return_pct for w in self.windows]
        self.is_avg_return_pct = sum(is_returns) / n

        self.is_vs_oos_gap = self.is_avg_return_pct - self.oos_avg_return_pct


class WalkForwardRunner:
    """Splits a date range into rolling train/test windows and runs backtests."""

    def __init__(self, runner: BacktestRunner):
        self.runner = runner

    def run(
        self,
        strategy_name: str,
        symbol: str,
        start: date,
        end: date,
        n_windows: int = 3,
        train_pct: float = 0.7,
        params: dict[str, Any] | None = None,
        interval: str = "1d",
    ) -> WalkForwardResult:
        total_days = (end - start).days
        window_days = total_days / n_windows
        train_days = int(window_days * train_pct)
        test_days = int(window_days * (1 - train_pct))

        result = WalkForwardResult(
            run_id=str(uuid.uuid4())[:8],
            strategy=strategy_name,
            symbol=symbol,
            start_date=str(start),
            end_date=str(end),
            n_windows=n_windows,
            train_pct=train_pct,
        )

        for i in range(n_windows):
            w_start = start + timedelta(days=int(i * window_days))
            train_end = w_start + timedelta(days=train_days)
            test_start = train_end + timedelta(days=1)
            test_end = w_start + timedelta(days=int(window_days))

            # Clamp last window to overall end date
            if i == n_windows - 1:
                test_end = end

            logger.info(
                f"Walk-forward window {i + 1}/{n_windows}: "
                f"train {w_start}-{train_end}, test {test_start}-{test_end}"
            )

            is_result = self.runner.run(
                strategy_name=strategy_name,
                symbol=symbol,
                start=w_start,
                end=train_end,
                params=params,
                interval=interval,
            )

            oos_result = self.runner.run(
                strategy_name=strategy_name,
                symbol=symbol,
                start=test_start,
                end=test_end,
                params=params,
                interval=interval,
            )

            result.windows.append(
                WindowResult(
                    window_index=i,
                    train_start=str(w_start),
                    train_end=str(train_end),
                    test_start=str(test_start),
                    test_end=str(test_end),
                    in_sample=is_result,
                    out_of_sample=oos_result,
                )
            )

        result.compute_aggregates()
        return result
