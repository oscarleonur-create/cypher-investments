"""Adaptive exit policy — adjusts exit parameters based on vol regime and conviction.

Uses HV20-tercile regime detection from drawdown_analysis to shift exit
parameters toward wider targets in high-vol environments and tighter stops
in low-vol environments.

| Parameter          | Low Vol | Normal | High Vol |
|--------------------|---------|--------|----------|
| profit_target_pct  |   0.40  |  0.50  |   0.65   |
| stop_loss_mult     |   2.0   |  3.0   |   4.0    |
| close_at_dte       |   14    |  21    |   28     |
"""

from __future__ import annotations

import copy
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from advisor.backtesting.options_backtester import BacktestConfig

logger = logging.getLogger(__name__)

# Regime parameter lookup table
REGIME_PARAMS: dict[str, dict[str, float | int]] = {
    "low": {"profit_target_pct": 0.40, "stop_loss_multiplier": 2.0, "close_at_dte": 14},
    "mid": {"profit_target_pct": 0.50, "stop_loss_multiplier": 3.0, "close_at_dte": 21},
    "high": {"profit_target_pct": 0.65, "stop_loss_multiplier": 4.0, "close_at_dte": 28},
}


class AdaptiveExitPolicy:
    """Adjusts BacktestConfig exit parameters based on vol regime + conviction."""

    def __init__(
        self,
        base_config: "BacktestConfig",
        vol_regime: str,
        conviction_score: float | None = None,
    ):
        """
        Args:
            base_config: Starting BacktestConfig to adapt.
            vol_regime: One of "low", "mid", "high" from compute_vol_regime_labels().
            conviction_score: Optional 0-100 conviction score. >70 widens profit
                target +10%; <30 tightens stop loss -20%.
        """

        self.base_config = base_config
        self.vol_regime = vol_regime if vol_regime in REGIME_PARAMS else "mid"
        self.conviction_score = conviction_score

    def adapt(self) -> "BacktestConfig":
        """Return a new BacktestConfig with regime-adjusted exit parameters."""

        config = copy.copy(self.base_config)
        params = REGIME_PARAMS[self.vol_regime]

        config.profit_target_pct = params["profit_target_pct"]
        config.stop_loss_multiplier = params["stop_loss_multiplier"]
        config.close_at_dte = params["close_at_dte"]

        # Conviction adjustments
        if self.conviction_score is not None:
            if self.conviction_score > 70:
                config.profit_target_pct *= 1.10  # widen by 10%
            elif self.conviction_score < 30:
                config.stop_loss_multiplier *= 0.80  # tighten by 20%

        logger.info(
            "Adaptive exits: regime=%s, conviction=%s -> PT=%.2f, SL=%.1fx, DTE=%d",
            self.vol_regime,
            self.conviction_score,
            config.profit_target_pct,
            config.stop_loss_multiplier,
            config.close_at_dte,
        )
        return config
