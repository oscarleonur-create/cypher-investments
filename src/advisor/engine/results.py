"""Backtest result models."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field


class TradeRecord(BaseModel):
    ref: int
    direction: str  # "long" or "short"
    symbol: str = ""
    open_date: str = ""
    close_date: str = ""
    open_price: float = 0.0
    close_price: float = 0.0
    pnl: float = 0.0
    pnl_pct: float = 0.0
    size: float = 0.0
    commission: float = 0.0


class BacktestResult(BaseModel):
    run_id: str
    strategy_name: str
    symbol: str
    start_date: str
    end_date: str
    created_at: datetime = Field(default_factory=datetime.now)

    # Portfolio metrics
    initial_cash: float
    final_value: float
    total_return: float = 0.0
    total_return_pct: float = 0.0

    # Risk metrics
    sharpe_ratio: float | None = None
    max_drawdown: float | None = None
    max_drawdown_pct: float | None = None

    # Trade stats
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    win_rate: float | None = None
    avg_trade_pnl: float | None = None

    # Detailed records
    trades: list[TradeRecord] = Field(default_factory=list)

    # Strategy params used
    params: dict[str, Any] = Field(default_factory=dict)

    def compute_derived(self) -> None:
        """Compute derived metrics from base values."""
        self.total_return = self.final_value - self.initial_cash
        if self.initial_cash > 0:
            self.total_return_pct = (self.total_return / self.initial_cash) * 100

        if self.total_trades > 0:
            self.win_rate = (self.winning_trades / self.total_trades) * 100
