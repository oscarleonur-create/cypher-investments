"""Base strategy class for the advisor system."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, ClassVar

if TYPE_CHECKING:
    from advisor.confluence.models import ConfluenceResult

import backtrader as bt

from advisor.core.enums import StrategyType

logger = logging.getLogger(__name__)


class StrategyParams(bt.MetaParams):
    """Metaclass that merges params from StrategyBase and subclasses."""

    pass


class StrategyBase(bt.Strategy):
    """Abstract base class for all advisor strategies.

    Strategies extend this to be both Backtrader-compatible and registry-aware.
    Subclasses must define class-level metadata and implement next().
    """

    # Registry metadata - subclasses must override
    strategy_name: ClassVar[str] = ""
    strategy_type: ClassVar[StrategyType] = StrategyType.EQUITY
    description: ClassVar[str] = ""
    version: ClassVar[str] = "1.0.0"
    force_all_confluence: ClassVar[bool] = False

    params: ClassVar[tuple] = (
        ("use_sizer", False),
        ("stop_loss_pct", -0.10),  # -10% hard stop; 0 = disabled
        ("trailing_stop_pct", 0.0),  # 0 = disabled; e.g. 0.08 for 8% trail
    )

    def __init__(self):
        super().__init__()
        self.order = None
        self._risk_entry_price: float | None = None
        self._risk_peak_price: float | None = None
        self._circuit_breaker_tripped = False
        self._validate_params()

    def _validate_params(self):
        """Override in subclasses to validate strategy parameters."""
        pass

    def _check_risk_exits(self) -> bool:
        """Check stop-loss and trailing-stop. Returns True if exit order issued.

        Call at the top of next() when in position:
            if self.position and self._check_risk_exits():
                return
        """
        if not self.position or self._risk_entry_price is None:
            return False

        # Circuit breaker — close everything, stop trading
        if self._circuit_breaker_tripped:
            self.order = self.close()
            return True

        price = self.data.close[0]

        # Update peak price for trailing stop
        if self._risk_peak_price is None or price > self._risk_peak_price:
            self._risk_peak_price = price

        # Hard stop-loss
        if self.p.stop_loss_pct < 0:
            pct_change = (price - self._risk_entry_price) / self._risk_entry_price
            if pct_change <= self.p.stop_loss_pct:
                logger.info(
                    "Stop-loss triggered at %.2f (entry %.2f, loss %.1f%%)",
                    price,
                    self._risk_entry_price,
                    pct_change * 100,
                )
                self.order = self.close()
                return True

        # Trailing stop
        if self.p.trailing_stop_pct > 0 and self._risk_peak_price is not None:
            drawdown = (price - self._risk_peak_price) / self._risk_peak_price
            if drawdown <= -self.p.trailing_stop_pct:
                logger.info(
                    "Trailing stop triggered at %.2f (peak %.2f, drawdown %.1f%%)",
                    price,
                    self._risk_peak_price,
                    drawdown * 100,
                )
                self.order = self.close()
                return True

        return False

    def notify_order(self, order):
        """Track entry prices for risk exits and clear pending order reference."""
        if order.status == order.Completed:
            if order.isbuy():
                self._risk_entry_price = order.executed.price
                self._risk_peak_price = order.executed.price
            elif order.issell() and not self.position:
                self._risk_entry_price = None
                self._risk_peak_price = None
        if order.status in [order.Completed, order.Canceled, order.Margin, order.Rejected]:
            self.order = None

    def get_info(self) -> dict[str, Any]:
        """Return strategy metadata for the registry."""
        return {
            "name": self.strategy_name,
            "type": self.strategy_type.value,
            "description": self.description,
            "version": self.version,
            "params": {
                name: getattr(self.params, name, default)
                for name, default in self._get_param_defaults().items()
            },
        }

    @classmethod
    def _get_param_defaults(cls) -> dict[str, Any]:
        """Extract parameter names and defaults from backtrader's params system."""
        result = {}
        # Backtrader transforms params tuples into AutoInfoClass objects
        params_cls = cls.params
        if hasattr(params_cls, "_getkeys"):
            keys = params_cls._getkeys()
            defaults = params_cls._getdefaults()
            for key, default in zip(keys, defaults):
                result[key] = default
        return result

    @classmethod
    def get_metadata(cls) -> dict[str, Any]:
        """Return class-level metadata without instantiation."""
        return {
            "name": cls.strategy_name,
            "type": cls.strategy_type.value,
            "description": cls.description,
            "version": cls.version,
            "params": cls._get_param_defaults(),
        }

    @classmethod
    def scan(cls, symbol: str) -> "ConfluenceResult":
        """Run the full confluence pipeline for this strategy.

        Returns:
            ConfluenceResult with verdict, all agent results, and reasoning.
        """
        from advisor.confluence.orchestrator import run_confluence

        return run_confluence(
            symbol, strategy_name=cls.strategy_name, force_all=cls.force_all_confluence
        )
