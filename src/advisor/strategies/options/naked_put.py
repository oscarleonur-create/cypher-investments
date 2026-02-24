"""Naked put strategy — thin adapter over the BS backtester.

DEPRECATED: The old Backtrader-based stub with hardcoded placeholder math has been
removed. Options strategies now use the Black-Scholes backtester at
``advisor.backtesting.options_backtester.Backtester``.

This module keeps the registry entry so ``advisor strategy list`` still shows the
strategy, but ``advisor backtest run naked_put`` is routed to the real backtester
in backtest_cmds.py.
"""

from __future__ import annotations

import logging
from typing import ClassVar

from advisor.core.enums import StrategyType
from advisor.strategies.base import StrategyBase
from advisor.strategies.registry import StrategyRegistry

logger = logging.getLogger(__name__)


@StrategyRegistry.register
class NakedPut(StrategyBase):
    strategy_name: ClassVar[str] = "naked_put"
    strategy_type: ClassVar[StrategyType] = StrategyType.OPTIONS
    description: ClassVar[str] = (
        "Sell OTM puts on red days when RSI < 40. Uses Black-Scholes backtester."
    )
    version: ClassVar[str] = "2.0.0"

    params: ClassVar[tuple] = ()

    def __init__(self):
        super().__init__()

    def next(self):
        # No-op: options strategies are handled by the BS backtester.
        pass
