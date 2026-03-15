"""Exit parameter optimizer — grid search over BacktestConfig exit rules.

Runs the existing backtester N times with different exit parameters and ranks
results by a chosen metric to find optimal exit settings empirically.
"""

from __future__ import annotations

import itertools
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from advisor.backtesting.options_backtester import BacktestConfig

from pydantic import BaseModel, Field
from rich.console import Console
from rich.table import Table

logger = logging.getLogger(__name__)


# ── Grid definition ──────────────────────────────────────────────────────────


@dataclass
class ExitParameterGrid:
    """Defines the parameter space for exit rule optimization."""

    profit_target_pcts: list[float] = None
    stop_loss_multipliers: list[float] = None
    close_at_dtes: list[int] = None

    def __post_init__(self):
        if self.profit_target_pcts is None:
            self.profit_target_pcts = [0.25, 0.50, 0.75]
        if self.stop_loss_multipliers is None:
            self.stop_loss_multipliers = [1.5, 2.0, 3.0, 4.0]
        if self.close_at_dtes is None:
            self.close_at_dtes = [7, 14, 21, 28]

    def combos(self) -> list[tuple[float, float, int]]:
        """All (profit_target, stop_loss, dte) combinations."""
        return list(
            itertools.product(
                self.profit_target_pcts,
                self.stop_loss_multipliers,
                self.close_at_dtes,
            )
        )

    @property
    def n_combos(self) -> int:
        return (
            len(self.profit_target_pcts) * len(self.stop_loss_multipliers) * len(self.close_at_dtes)
        )


# ── Result model ─────────────────────────────────────────────────────────────


class GridResult(BaseModel):
    """Outcome of one parameter combination."""

    profit_target_pct: float
    stop_loss_multiplier: float
    close_at_dte: int

    total_pnl: float = 0.0
    win_rate_pct: float = 0.0
    sharpe_ratio: float = 0.0
    profit_factor: float = Field(default=0.0, description="Sum(wins) / abs(sum(losses))")
    max_drawdown_pct: float = 0.0
    avg_hold_days: float = 0.0
    num_trades: int = 0
    exit_reasons: dict[str, int] = Field(default_factory=dict)


# ── Worker function (top-level for pickling) ─────────────────────────────────


def _run_single_combo(
    symbol: str,
    start: str,
    end: str,
    cash: float,
    strategy: str,
    profit_target: float,
    stop_loss: float,
    close_dte: int,
    base_config_dict: dict,
) -> GridResult:
    """Run a single backtest combo. Designed for ProcessPoolExecutor."""
    from advisor.backtesting.options_backtester import BacktestConfig, Backtester

    config = BacktestConfig(**base_config_dict)
    config.profit_target_pct = profit_target
    config.stop_loss_multiplier = stop_loss
    config.close_at_dte = close_dte

    bt = Backtester(symbol, start, end, cash, config=config)
    result = bt.run(strategy)

    pnls = [t.get("pnl", 0) or 0 for t in result.get("trades", [])]
    wins_sum = sum(p for p in pnls if p > 0)
    losses_sum = abs(sum(p for p in pnls if p <= 0))
    profit_factor = (
        wins_sum / losses_sum if losses_sum > 0 else float("inf") if wins_sum > 0 else 0.0
    )

    # Average hold days
    hold_days = []
    for t in result.get("trades", []):
        if t.get("entry_date") and t.get("exit_date"):
            import pandas as pd

            d = (pd.Timestamp(t["exit_date"]) - pd.Timestamp(t["entry_date"])).days
            hold_days.append(d)

    return GridResult(
        profit_target_pct=profit_target,
        stop_loss_multiplier=stop_loss,
        close_at_dte=close_dte,
        total_pnl=result.get("total_pnl", 0.0),
        win_rate_pct=result.get("win_rate_pct", 0.0),
        sharpe_ratio=result.get("sharpe_ratio", 0.0),
        profit_factor=round(profit_factor, 2),
        max_drawdown_pct=result.get("max_drawdown_pct", 0.0),
        avg_hold_days=round(sum(hold_days) / len(hold_days), 1) if hold_days else 0.0,
        num_trades=result.get("num_trades", 0),
        exit_reasons=result.get("exit_reasons", {}),
    )


# ── Optimizer ────────────────────────────────────────────────────────────────


class ExitOptimizer:
    """Grid-search optimizer for exit parameters."""

    def __init__(
        self,
        symbol: str,
        start: str,
        end: str,
        strategy: str,
        cash: float = 100_000.0,
        base_config: "BacktestConfig | None" = None,
    ):
        from advisor.backtesting.options_backtester import BacktestConfig

        self.symbol = symbol
        self.start = start
        self.end = end
        self.strategy = strategy
        self.cash = cash
        self.base_config = base_config or BacktestConfig()

    def optimize(
        self,
        grid: ExitParameterGrid,
        metric: str = "sharpe_ratio",
        max_workers: int | None = None,
        progress_callback: callable | None = None,
    ) -> list[GridResult]:
        """Run grid search and return results sorted by metric (descending).

        Uses ProcessPoolExecutor to parallelize across combos.
        Each worker creates its own Backtester (loads data independently).
        """
        from dataclasses import asdict

        combos = grid.combos()
        n_total = len(combos)
        logger.info(
            "Running %d exit parameter combos for %s %s", n_total, self.strategy, self.symbol
        )

        base_dict = asdict(self.base_config)
        results: list[GridResult] = []

        workers = max_workers or min(4, n_total)

        with ProcessPoolExecutor(max_workers=workers) as executor:
            future_to_combo = {}
            for pt, sl, dte in combos:
                future = executor.submit(
                    _run_single_combo,
                    self.symbol,
                    self.start,
                    self.end,
                    self.cash,
                    self.strategy,
                    pt,
                    sl,
                    dte,
                    base_dict,
                )
                future_to_combo[future] = (pt, sl, dte)

            completed = 0
            for future in as_completed(future_to_combo):
                completed += 1
                combo = future_to_combo[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.warning("Combo %s failed: %s", combo, e)

                if progress_callback:
                    progress_callback(completed, n_total)

        # Sort by chosen metric (descending; for drawdown, more negative is worse)
        reverse = metric != "max_drawdown_pct"
        results.sort(key=lambda r: getattr(r, metric, 0), reverse=reverse)
        return results


# ── Display ──────────────────────────────────────────────────────────────────


def render_ranking_table(results: list[GridResult], top_n: int = 10) -> None:
    """Print a Rich table of top grid search results."""
    console = Console()
    table = Table(title=f"Top {min(top_n, len(results))} Exit Parameter Combos", show_lines=True)

    table.add_column("#", justify="right", style="dim")
    table.add_column("Profit Target", justify="right")
    table.add_column("Stop Loss", justify="right")
    table.add_column("DTE Exit", justify="right")
    table.add_column("Total P&L", justify="right")
    table.add_column("Win Rate", justify="right")
    table.add_column("Sharpe", justify="right")
    table.add_column("Profit Factor", justify="right")
    table.add_column("Max DD", justify="right")
    table.add_column("Avg Hold", justify="right")
    table.add_column("Trades", justify="right")

    for i, r in enumerate(results[:top_n], 1):
        pnl_style = "green" if r.total_pnl >= 0 else "red"
        table.add_row(
            str(i),
            f"{r.profit_target_pct:.0%}",
            f"{r.stop_loss_multiplier:.1f}x",
            str(r.close_at_dte),
            f"[{pnl_style}]${r.total_pnl:,.2f}[/]",
            f"{r.win_rate_pct:.1f}%",
            f"{r.sharpe_ratio:.2f}",
            f"{r.profit_factor:.2f}",
            f"{r.max_drawdown_pct:.1f}%",
            f"{r.avg_hold_days:.0f}d",
            str(r.num_trades),
        )

    console.print(table)
