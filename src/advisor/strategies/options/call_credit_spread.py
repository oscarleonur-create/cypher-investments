"""Call credit spread strategy — bearish premium selling via BS backtester."""

from __future__ import annotations

from typing import ClassVar

from advisor.core.enums import StrategyType
from advisor.strategies.base import StrategyBase
from advisor.strategies.registry import StrategyRegistry


@StrategyRegistry.register
class CallCreditSpread(StrategyBase):
    strategy_name: ClassVar[str] = "call_credit_spread"
    strategy_type: ClassVar[StrategyType] = StrategyType.OPTIONS
    description: ClassVar[str] = (
        "Sell call credit spreads on overbought stocks. "
        "Bearish counterpart to put credit spreads with defined risk."
    )
    version: ClassVar[str] = "1.0.0"
    params: ClassVar[tuple] = ()

    def __init__(self):
        super().__init__()

    def next(self):
        pass
