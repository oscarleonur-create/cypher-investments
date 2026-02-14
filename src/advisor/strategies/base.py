"""Base strategy class for the advisor system."""

from __future__ import annotations

from typing import Any, ClassVar

import backtrader as bt

from advisor.core.enums import StrategyType


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

    params: ClassVar[tuple] = (
        ("use_sizer", False),
    )

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

        return run_confluence(symbol, strategy_name=cls.strategy_name)
