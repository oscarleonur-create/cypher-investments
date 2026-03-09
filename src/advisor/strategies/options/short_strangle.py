"""Short strangle strategy — neutral undefined-risk premium selling via BS backtester."""

from __future__ import annotations

from typing import ClassVar

from advisor.core.enums import StrategyType
from advisor.strategies.base import StrategyBase
from advisor.strategies.registry import StrategyRegistry


@StrategyRegistry.register
class ShortStrangle(StrategyBase):
    strategy_name: ClassVar[str] = "short_strangle"
    strategy_type: ClassVar[StrategyType] = StrategyType.OPTIONS
    description: ClassVar[str] = (
        "Sell short strangles (naked put + naked call) on range-bound stocks. "
        "Neutral strategy with undefined risk — conservative sizing required."
    )
    version: ClassVar[str] = "1.0.0"
    params: ClassVar[tuple] = ()

    def __init__(self):
        super().__init__()

    def next(self):
        pass
