"""Iron condor strategy — neutral defined-risk premium selling via BS backtester."""

from __future__ import annotations

from typing import ClassVar

from advisor.core.enums import StrategyType
from advisor.strategies.base import StrategyBase
from advisor.strategies.registry import StrategyRegistry


@StrategyRegistry.register
class IronCondor(StrategyBase):
    strategy_name: ClassVar[str] = "iron_condor"
    strategy_type: ClassVar[StrategyType] = StrategyType.OPTIONS
    description: ClassVar[str] = (
        "Sell iron condors (put spread + call spread) on range-bound stocks. "
        "Neutral strategy with defined risk on both sides."
    )
    version: ClassVar[str] = "1.0.0"
    params: ClassVar[tuple] = ()

    def __init__(self):
        super().__init__()

    def next(self):
        pass
