"""Shared exit rule configuration for options strategies.

Provides a single source of truth for exit parameters used by both the
options backtester (scalar, per-bar evaluation) and the Monte Carlo
simulator (vectorized, per-path evaluation).
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ExitRuleConfig:
    """Unified exit rule defaults for options strategies.

    Both the backtester (BacktestConfig) and simulator (SimConfig) should
    derive their exit-rule defaults from this class to prevent drift.
    """

    profit_target_pct: float = 0.50  # close at 50% of credit collected
    stop_loss_multiplier: float = 3.0  # close at 3x credit loss
    close_at_dte: int = 21  # close when DTE <= this value


# Canonical defaults — import this instead of hardcoding values
EXIT_RULE_DEFAULTS = ExitRuleConfig()
