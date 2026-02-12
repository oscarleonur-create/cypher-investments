"""Strategy registry with decorator-based registration and auto-discovery."""

from __future__ import annotations

import importlib
import logging
import pkgutil
from typing import Any

from advisor.strategies.base import StrategyBase

logger = logging.getLogger(__name__)


class StrategyRegistry:
    """Singleton registry for strategy classes."""

    _instance: StrategyRegistry | None = None
    _strategies: dict[str, type[StrategyBase]]

    def __new__(cls) -> StrategyRegistry:
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._strategies = {}
        return cls._instance

    @classmethod
    def register(cls, strategy_cls: type[StrategyBase]) -> type[StrategyBase]:
        """Decorator to register a strategy class.

        Usage:
            @StrategyRegistry.register
            class MyStrategy(StrategyBase):
                strategy_name = "my_strategy"
                ...
        """
        instance = cls()
        name = strategy_cls.strategy_name
        if not name:
            raise ValueError(
                f"Strategy {strategy_cls.__name__} must define 'strategy_name'"
            )
        if name in instance._strategies:
            logger.warning(f"Overwriting existing strategy: {name}")
        instance._strategies[name] = strategy_cls
        logger.debug(f"Registered strategy: {name}")
        return strategy_cls

    def get(self, name: str) -> type[StrategyBase] | None:
        return self._strategies.get(name)

    def list_strategies(self) -> list[dict[str, Any]]:
        return [cls.get_metadata() for cls in self._strategies.values()]

    def get_strategy(self, name: str) -> type[StrategyBase]:
        cls = self._strategies.get(name)
        if cls is None:
            available = ", ".join(sorted(self._strategies.keys()))
            raise KeyError(
                f"Strategy '{name}' not found. Available: {available}"
            )
        return cls

    @property
    def names(self) -> list[str]:
        return sorted(self._strategies.keys())

    def discover(self) -> None:
        """Auto-discover strategies in advisor.strategies subpackages."""
        import advisor.strategies as pkg

        for _importer, modname, _ispkg in pkgutil.walk_packages(
            pkg.__path__, prefix="advisor.strategies."
        ):
            try:
                importlib.import_module(modname)
            except Exception as e:
                logger.warning(f"Failed to import {modname}: {e}")

    def discover_user_strategies(self, path: str) -> None:
        """Discover strategies from a user-provided directory."""
        import sys
        from pathlib import Path

        strategies_dir = Path(path)
        if not strategies_dir.is_dir():
            return

        if str(strategies_dir.parent) not in sys.path:
            sys.path.insert(0, str(strategies_dir.parent))

        for _importer, modname, _ispkg in pkgutil.walk_packages(
            [str(strategies_dir)], prefix=f"{strategies_dir.name}."
        ):
            try:
                importlib.import_module(modname)
            except Exception as e:
                logger.warning(f"Failed to import user strategy {modname}: {e}")

    @classmethod
    def reset(cls) -> None:
        """Reset the registry (mainly for testing)."""
        if cls._instance is not None:
            cls._instance._strategies = {}
