"""Put credit spread strategy — BS backtester + Monte Carlo simulator.

Supports both the legacy Black-Scholes backtester and the new Monte Carlo
simulator with real TastyTrade market data (--monte-carlo flag).
"""

from __future__ import annotations

from typing import ClassVar

from advisor.core.enums import StrategyType
from advisor.strategies.base import StrategyBase
from advisor.strategies.registry import StrategyRegistry


@StrategyRegistry.register
class PutCreditSpread(StrategyBase):
    strategy_name: ClassVar[str] = "put_credit_spread"
    strategy_type: ClassVar[StrategyType] = StrategyType.OPTIONS
    description: ClassVar[str] = (
        "Sell put credit spreads on oversold stocks. "
        "Supports BS backtester and Monte Carlo simulator with real market data."
    )
    version: ClassVar[str] = "3.0.0"
    params: ClassVar[tuple] = ()

    def __init__(self):
        super().__init__()

    def next(self):
        pass
