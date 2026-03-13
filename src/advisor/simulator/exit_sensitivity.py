"""Exit sensitivity analysis — sweep exit parameters over shared MC paths.

Generates paths ONCE, then replays exit logic with different params for a fair
apples-to-apples comparison across the same market scenarios.
"""

from __future__ import annotations

import itertools
import logging

import numpy as np
from pydantic import BaseModel, Field

from advisor.simulator.engine import MonteCarloEngine, bsm_put_price_vec
from advisor.simulator.models import PCSCandidate, SimConfig

logger = logging.getLogger(__name__)


# ── Models ───────────────────────────────────────────────────────────────────


class ExitSensitivityPoint(BaseModel):
    """Results for one exit parameter combination."""

    profit_target_pct: float
    stop_loss_multiplier: float
    close_at_dte: int

    ev: float = 0.0
    pop: float = 0.0
    cvar_95: float = 0.0
    avg_hold_days: float = 0.0
    sharpe_approx: float = Field(default=0.0, description="EV / StdDev of P&L")

    exit_profit_target: float = 0.0
    exit_stop_loss: float = 0.0
    exit_dte: float = 0.0
    exit_expiration: float = 0.0
    exit_trailing_stop: float = 0.0


class ExitSensitivityResult(BaseModel):
    """Full sensitivity sweep output."""

    symbol: str
    short_strike: float
    long_strike: float
    dte: int
    net_credit: float
    n_paths: int
    points: list[ExitSensitivityPoint] = Field(default_factory=list)


# ── Analyzer ─────────────────────────────────────────────────────────────────


class ExitSensitivityAnalyzer:
    """Sweep exit parameters over pre-generated MC paths."""

    def __init__(self, engine: MonteCarloEngine, candidate: PCSCandidate):
        self.engine = engine
        self.candidate = candidate

    def sweep(
        self,
        profit_targets: list[float] | None = None,
        stop_losses: list[float] | None = None,
        close_dtes: list[int] | None = None,
    ) -> ExitSensitivityResult:
        """Run sensitivity sweep. Generates paths once, replays per combo."""
        if profit_targets is None:
            profit_targets = [0.25, 0.40, 0.50, 0.60, 0.75]
        if stop_losses is None:
            stop_losses = [1.0, 1.5, 2.0, 3.0]
        if close_dtes is None:
            close_dtes = [0, 5, 7, 14, 21]

        candidate = self.candidate
        cfg = self.engine.config
        atm_iv = cfg.vol_mean_level if cfg.vol_mean_level > 0 else candidate.short_iv

        # Generate paths ONCE
        prices, ivs = self.engine._generate_paths(
            S0=candidate.underlying_price,
            iv0=atm_iv,
            dte=candidate.dte,
        )
        n_paths = prices.shape[0]
        logger.info(
            "Sweeping %d combos over %d paths",
            len(profit_targets) * len(stop_losses) * len(close_dtes),
            n_paths,
        )

        points = []
        for pt, sl, dte_exit in itertools.product(profit_targets, stop_losses, close_dtes):
            point = self._replay_exits(prices, ivs, candidate, cfg, pt, sl, dte_exit)
            points.append(point)

        return ExitSensitivityResult(
            symbol=candidate.symbol,
            short_strike=candidate.short_strike,
            long_strike=candidate.long_strike,
            dte=candidate.dte,
            net_credit=candidate.net_credit,
            n_paths=n_paths,
            points=points,
        )

    def _replay_exits(
        self,
        prices: np.ndarray,
        ivs: np.ndarray,
        candidate: PCSCandidate,
        cfg: SimConfig,
        profit_target_pct: float,
        stop_loss_multiplier: float,
        close_at_dte: int,
    ) -> ExitSensitivityPoint:
        """Replay exit logic on pre-generated paths with given params."""
        n_paths, n_days = prices.shape
        dte = n_days - 1

        credit = candidate.net_credit * 100
        max_loss = (candidate.width - candidate.net_credit) * 100

        short_iv_ratio = candidate.short_iv / max(candidate.short_iv, 1e-8)
        long_iv_ratio = candidate.long_iv / max(candidate.short_iv, 1e-8)

        # Initial spread value
        S0 = prices[:, 0]
        iv0 = ivs[:, 0]
        T0 = dte / 252.0
        short_put_0 = bsm_put_price_vec(
            S0, candidate.short_strike, T0, cfg.risk_free_rate, iv0 * short_iv_ratio
        )
        long_put_0 = bsm_put_price_vec(
            S0, candidate.long_strike, T0, cfg.risk_free_rate, iv0 * long_iv_ratio
        )
        entry_spread_value = (short_put_0 - long_put_0) * 100

        profit_threshold = credit * profit_target_pct
        stop_threshold = credit * stop_loss_multiplier

        pnl = np.full(n_paths, np.nan)
        hold_days = np.full(n_paths, dte, dtype=np.float64)
        exit_reason = np.zeros(n_paths, dtype=np.int32)

        # Trailing stop tracking
        max_upnl = np.zeros(n_paths, dtype=np.float64)
        trailing_activation = credit * cfg.trailing_activation_pct if cfg.use_trailing_stop else 0.0

        active = np.ones(n_paths, dtype=bool)
        slippage = cfg.slippage_pct * candidate.width * 100

        for day in range(1, n_days):
            if not np.any(active):
                break

            remaining_dte = dte - day
            T = remaining_dte / 252.0
            S = prices[active, day]
            iv = ivs[active, day]

            short_iv = iv * short_iv_ratio
            long_iv = iv * long_iv_ratio
            short_put = bsm_put_price_vec(
                S, candidate.short_strike, T, cfg.risk_free_rate, short_iv
            )
            long_put = bsm_put_price_vec(S, candidate.long_strike, T, cfg.risk_free_rate, long_iv)
            current_spread = (short_put - long_put) * 100

            entry_vals = entry_spread_value[active]
            unrealized_pnl = entry_vals - current_spread
            change = current_spread - entry_vals
            mtm_pnl = credit - change - slippage

            # DTE exit
            if close_at_dte > 0 and remaining_dte <= close_at_dte and remaining_dte > 0:
                active_indices = np.where(active)[0]
                pnl[active_indices] = mtm_pnl
                hold_days[active_indices] = day
                exit_reason[active_indices] = 3
                active[active_indices] = False
                continue

            # Profit target
            profit_mask = unrealized_pnl >= profit_threshold
            if np.any(profit_mask):
                active_indices = np.where(active)[0]
                profit_indices = active_indices[profit_mask]
                pnl[profit_indices] = mtm_pnl[profit_mask]
                hold_days[profit_indices] = day
                exit_reason[profit_indices] = 1
                active[profit_indices] = False

            if not np.any(active):
                continue

            # Trailing stop
            if cfg.use_trailing_stop:
                current_upnl = entry_spread_value[active] - (
                    (
                        bsm_put_price_vec(
                            prices[active, day],
                            candidate.short_strike,
                            T,
                            cfg.risk_free_rate,
                            ivs[active, day] * short_iv_ratio,
                        )
                        - bsm_put_price_vec(
                            prices[active, day],
                            candidate.long_strike,
                            T,
                            cfg.risk_free_rate,
                            ivs[active, day] * long_iv_ratio,
                        )
                    )
                    * 100
                )
                active_indices = np.where(active)[0]
                max_upnl[active_indices] = np.maximum(max_upnl[active_indices], current_upnl)
                activated = max_upnl[active_indices] >= trailing_activation
                below_floor = current_upnl < (cfg.trailing_floor_pct * max_upnl[active_indices])
                trail_mask = activated & below_floor
                if np.any(trail_mask):
                    trail_indices = active_indices[trail_mask]
                    pnl[trail_indices] = credit + current_upnl[trail_mask] - slippage
                    hold_days[trail_indices] = day
                    exit_reason[trail_indices] = 4
                    active[trail_indices] = False

            if not np.any(active):
                continue

            # Stop loss
            remaining_upnl = entry_spread_value[active] - (
                (
                    bsm_put_price_vec(
                        prices[active, day],
                        candidate.short_strike,
                        T,
                        cfg.risk_free_rate,
                        ivs[active, day] * short_iv_ratio,
                    )
                    - bsm_put_price_vec(
                        prices[active, day],
                        candidate.long_strike,
                        T,
                        cfg.risk_free_rate,
                        ivs[active, day] * long_iv_ratio,
                    )
                )
                * 100
            )
            stop_mask = remaining_upnl <= -stop_threshold
            if np.any(stop_mask):
                active_indices = np.where(active)[0]
                stop_indices = active_indices[stop_mask]
                stop_change = -remaining_upnl[stop_mask]
                pnl[stop_indices] = credit - stop_change - slippage
                hold_days[stop_indices] = day
                exit_reason[stop_indices] = 2
                active[stop_indices] = False

        # Remaining: settle at expiration
        if np.any(active):
            active_indices = np.where(active)[0]
            S_final = prices[active, -1]
            short_intrinsic = np.maximum(candidate.short_strike - S_final, 0.0) * 100
            long_intrinsic = np.maximum(candidate.long_strike - S_final, 0.0) * 100
            spread_intrinsic = short_intrinsic - long_intrinsic
            entry_vals = entry_spread_value[active]
            pnl[active_indices] = credit - (spread_intrinsic - entry_vals)
            exit_reason[active_indices] = 0

        pnl = np.maximum(pnl, -max_loss)

        # Statistics
        valid_pnl = pnl[~np.isnan(pnl)]
        ev = float(np.mean(valid_pnl))
        pop = float(np.mean(valid_pnl > 0))
        sorted_pnl = np.sort(valid_pnl)
        n_tail = max(int(len(sorted_pnl) * 0.05), 1)
        cvar_95 = float(np.mean(sorted_pnl[:n_tail]))
        avg_hold = float(np.nanmean(hold_days))
        std_pnl = float(np.std(valid_pnl))
        sharpe_approx = ev / std_pnl if std_pnl > 0 else 0.0

        n_total = len(pnl)
        return ExitSensitivityPoint(
            profit_target_pct=profit_target_pct,
            stop_loss_multiplier=stop_loss_multiplier,
            close_at_dte=close_at_dte,
            ev=round(ev, 2),
            pop=round(pop, 4),
            cvar_95=round(cvar_95, 2),
            avg_hold_days=round(avg_hold, 1),
            sharpe_approx=round(sharpe_approx, 4),
            exit_profit_target=round(float(np.sum(exit_reason == 1) / n_total), 4),
            exit_stop_loss=round(float(np.sum(exit_reason == 2) / n_total), 4),
            exit_dte=round(float(np.sum(exit_reason == 3) / n_total), 4),
            exit_expiration=round(float(np.sum(exit_reason == 0) / n_total), 4),
            exit_trailing_stop=round(float(np.sum(exit_reason == 4) / n_total), 4),
        )
