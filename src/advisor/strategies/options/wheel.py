"""Wheel strategy — thin adapter over the BS backtester.

DEPRECATED: See advisor.backtesting.options_backtester.Backtester.
"""

from __future__ import annotations

from typing import ClassVar

from advisor.core.enums import StrategyType
from advisor.strategies.base import StrategyBase
from advisor.strategies.registry import StrategyRegistry


@StrategyRegistry.register
class Wheel(StrategyBase):
    strategy_name: ClassVar[str] = "wheel"
    strategy_type: ClassVar[StrategyType] = StrategyType.OPTIONS
    description: ClassVar[str] = (
        "The Wheel: sell puts until assigned, then sell covered calls."
        " Uses Black-Scholes backtester."
    )
    version: ClassVar[str] = "2.0.0"
    params: ClassVar[tuple] = ()

    def __init__(self):
        super().__init__()

    def next(self):
        pass
