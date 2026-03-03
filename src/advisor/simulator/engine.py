"""Monte Carlo engine — vectorized simulation of put credit spread outcomes."""

from __future__ import annotations

import logging

import numpy as np
from scipy.stats import norm
from scipy.stats import t as student_t

from advisor.simulator.models import PCSCandidate, SimConfig, SimResult

logger = logging.getLogger(__name__)


def bsm_put_price_vec(
    S: np.ndarray,
    K: float,
    T: float | np.ndarray,
    r: float,
    sigma: np.ndarray,
) -> np.ndarray:
    """Vectorized Black-Scholes put price for arrays of S and sigma.

    ~1000x faster than scalar bsm_price() for batch pricing.
    """
    # Handle expiration
    if np.isscalar(T):
        T = np.full_like(S, T, dtype=np.float64)

    result = np.zeros_like(S, dtype=np.float64)
    expired = T <= 0
    live = ~expired

    # Expired: intrinsic value
    result[expired] = np.maximum(K - S[expired], 0.0)

    if not np.any(live):
        return result

    S_live = S[live]
    T_live = T[live]
    sigma_live = sigma[live]

    # Avoid division by zero
    sigma_safe = np.maximum(sigma_live, 1e-8)
    T_safe = np.maximum(T_live, 1e-8)

    sqrt_T = np.sqrt(T_safe)
    d1 = (np.log(S_live / K) + (r + 0.5 * sigma_safe**2) * T_safe) / (sigma_safe * sqrt_T)
    d2 = d1 - sigma_safe * sqrt_T

    result[live] = K * np.exp(-r * T_safe) * norm.cdf(-d2) - S_live * norm.cdf(-d1)
    return result


class MonteCarloEngine:
    """Monte Carlo simulator for put credit spreads."""

    def __init__(self, config: SimConfig | None = None):
        self.config = config or SimConfig()

    def _generate_paths(
        self,
        S0: float,
        iv0: float,
        dte: int,
        n_paths: int | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Generate correlated Student-t price + mean-reverting IV paths.

        Returns (prices, ivs) arrays of shape (n_paths, dte+1).

        Supports antithetic variates (pair Z with -Z) and stratified sampling
        (divide uniform CDF into strata for full distribution coverage).
        """
        cfg = self.config
        n = n_paths or cfg.n_paths
        dt = 1 / 252  # Daily steps

        # Seeded RNG for reproducibility
        rng = np.random.default_rng(cfg.seed)

        # Determine base draw count for antithetic
        if cfg.use_antithetic:
            n_base = n // 2
        else:
            n_base = n

        # Generate base innovations
        if cfg.use_stratified:
            # Stratified sampling: divide [0,1] into n_base strata per timestep
            z_price = self._stratified_student_t(n_base, dte, rng)
            z_vol_base = self._stratified_normal(n_base, dte, rng)
        else:
            z_price = student_t.rvs(df=cfg.student_t_df, size=(n_base, dte), random_state=rng)
            z_vol_base = rng.standard_normal((n_base, dte))

        # Apply antithetic: mirror base draws
        if cfg.use_antithetic:
            z_price = np.concatenate([z_price, -z_price], axis=0)
            z_vol_base = np.concatenate([z_vol_base, -z_vol_base], axis=0)

        actual_n = z_price.shape[0]

        # Correlate vol innovations with price innovations via leverage effect
        z_vol = cfg.leverage_effect * z_price + np.sqrt(1 - cfg.leverage_effect**2) * z_vol_base

        prices = np.zeros((actual_n, dte + 1))
        ivs = np.zeros((actual_n, dte + 1))
        prices[:, 0] = S0
        ivs[:, 0] = iv0

        kappa = cfg.vol_mean_revert_speed
        theta = cfg.vol_mean_level
        xi = cfg.vol_of_vol  # vol-of-vol parameter

        for t in range(dte):
            vol = ivs[:, t]
            vol_safe = np.maximum(vol, 0.01)  # Floor at 1%

            # Price evolution: GBM with fat-tailed innovations
            drift = (cfg.risk_free_rate - 0.5 * vol_safe**2) * dt
            prices[:, t + 1] = prices[:, t] * np.exp(drift + vol_safe * np.sqrt(dt) * z_price[:, t])

            # IV evolution: OU process with configurable vol-of-vol
            iv_change = kappa * (theta - vol) * dt + xi * vol_safe * np.sqrt(dt) * z_vol[:, t]
            ivs[:, t + 1] = np.maximum(vol + iv_change, 0.01)

        return prices, ivs

    def _stratified_student_t(self, n: int, dte: int, rng: np.random.Generator) -> np.ndarray:
        """Stratified Student-t draws: divide [0,1] into n strata, apply inverse CDF."""
        strata = np.arange(n, dtype=np.float64)
        result = np.empty((n, dte))
        for t in range(dte):
            u = (strata + rng.uniform(size=n)) / n
            u = np.clip(u, 1e-10, 1 - 1e-10)
            result[:, t] = student_t.ppf(u, df=self.config.student_t_df)
            rng.shuffle(result[:, t])  # Break ordering correlation across timesteps
        return result

    def _stratified_normal(self, n: int, dte: int, rng: np.random.Generator) -> np.ndarray:
        """Stratified normal draws: divide [0,1] into n strata, apply inverse CDF."""
        strata = np.arange(n, dtype=np.float64)
        result = np.empty((n, dte))
        for t in range(dte):
            u = (strata + rng.uniform(size=n)) / n
            u = np.clip(u, 1e-10, 1 - 1e-10)
            result[:, t] = norm.ppf(u)
            rng.shuffle(result[:, t])
        return result

    def _simulate_pcs(
        self,
        candidate: PCSCandidate,
        prices: np.ndarray,
        ivs: np.ndarray,
        return_raw_pnl: bool = False,
    ) -> SimResult | tuple[SimResult, np.ndarray]:
        """Simulate a put credit spread across all paths.

        Uses mark-to-market approach: compute initial BSM spread value at t=0,
        then track changes from that baseline. Exit rules (profit target, stop
        loss) are based on the mark-to-market P&L relative to credit collected.

        When return_raw_pnl=True, returns (SimResult, pnl_array) for use by
        importance sampling and control variate computations.
        """
        cfg = self.config
        n_paths, n_days = prices.shape
        dte = n_days - 1

        credit = candidate.net_credit * 100  # Per contract in dollars
        max_loss = (candidate.width - candidate.net_credit) * 100

        # IV skew ratio between short and long strikes (preserved throughout sim)
        # This models differential skew: lower strikes carry higher IV
        short_iv_ratio = candidate.short_iv / max(candidate.short_iv, 1e-8)  # 1.0
        long_iv_ratio = candidate.long_iv / max(candidate.short_iv, 1e-8)

        # Compute initial BSM spread value at entry (t=0)
        S0 = prices[:, 0]
        iv0 = ivs[:, 0]
        T0 = dte / 252.0
        short_iv_0 = iv0 * short_iv_ratio
        long_iv_0 = iv0 * long_iv_ratio
        short_put_0 = bsm_put_price_vec(
            S0, candidate.short_strike, T0, cfg.risk_free_rate, short_iv_0
        )
        long_put_0 = bsm_put_price_vec(S0, candidate.long_strike, T0, cfg.risk_free_rate, long_iv_0)
        entry_spread_value = (short_put_0 - long_put_0) * 100

        # Profit target: close when unrealized P&L >= profit_target_pct * credit
        profit_threshold = credit * cfg.profit_target_pct
        # Stop loss: close when unrealized loss >= stop_loss_multiplier * credit
        stop_threshold = credit * cfg.stop_loss_multiplier

        # Track per-path outcomes
        pnl = np.full(n_paths, np.nan)
        hold_days = np.full(n_paths, dte, dtype=np.float64)
        exit_reason = np.zeros(n_paths, dtype=np.int32)  # 0=expiry, 1=profit, 2=stop, 3=dte
        touched_short = np.zeros(n_paths, dtype=bool)

        active = np.ones(n_paths, dtype=bool)
        slippage = cfg.slippage_pct * candidate.width * 100

        for day in range(1, n_days):
            if not np.any(active):
                break

            remaining_dte = dte - day
            T = remaining_dte / 252.0
            S = prices[active, day]
            iv = ivs[active, day]

            # Check if price touched short strike
            touched_short[active] |= S <= candidate.short_strike

            # Reprice with per-strike IV (preserving skew ratio)
            short_iv = iv * short_iv_ratio
            long_iv = iv * long_iv_ratio
            short_put = bsm_put_price_vec(
                S, candidate.short_strike, T, cfg.risk_free_rate, short_iv
            )
            long_put = bsm_put_price_vec(S, candidate.long_strike, T, cfg.risk_free_rate, long_iv)
            current_spread = (short_put - long_put) * 100

            # Unrealized P&L = entry_value - current_value (positive = profit)
            entry_vals = entry_spread_value[active]
            unrealized_pnl = entry_vals - current_spread

            # Realized P&L = credit - change_in_spread - slippage
            change = current_spread - entry_vals
            mtm_pnl = credit - change - slippage

            # Check DTE exit first
            if remaining_dte <= cfg.close_at_dte and remaining_dte > 0:
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

            # Stop loss — reuse unrealized_pnl for remaining active paths
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

        # Remaining paths: settle at expiration using MTM-consistent formula
        if np.any(active):
            active_indices = np.where(active)[0]
            S_final = prices[active, -1]
            short_intrinsic = np.maximum(candidate.short_strike - S_final, 0.0) * 100
            long_intrinsic = np.maximum(candidate.long_strike - S_final, 0.0) * 100
            spread_intrinsic = short_intrinsic - long_intrinsic
            # Consistent with MTM: pnl = credit - (final_value - entry_value)
            entry_vals = entry_spread_value[active]
            pnl[active_indices] = credit - (spread_intrinsic - entry_vals)
            exit_reason[active_indices] = 0

        # Cap losses at max_loss
        pnl = np.maximum(pnl, -max_loss)

        # ── Control variate adjustment ────────────────────────────────────
        raw_pnl = pnl.copy()
        variance_reduction_factor = 1.0

        if cfg.use_control_variate:
            pnl, variance_reduction_factor = self._apply_control_variate(
                pnl, candidate, prices, ivs
            )

        # ── Compute statistics ────────────────────────────────────────────
        valid_pnl = pnl[~np.isnan(pnl)]
        ev = float(np.mean(valid_pnl))
        pop = float(np.nanmean(pnl > 0))
        touch_prob = float(np.mean(touched_short))

        # MC standard error
        mc_std_err = float(np.std(valid_pnl) / np.sqrt(len(valid_pnl)))

        # CVaR 95: average of worst 5%
        sorted_pnl = np.sort(valid_pnl)
        n_tail = max(int(len(sorted_pnl) * 0.05), 1)
        cvar_95 = float(np.mean(sorted_pnl[:n_tail]))

        stop_prob = float(np.mean(exit_reason == 2))
        avg_hold = float(np.nanmean(hold_days))
        buying_power = candidate.buying_power
        ev_per_bp = ev / buying_power if buying_power > 0 else 0.0

        percentiles = np.nanpercentile(pnl, [5, 25, 50, 75, 95])
        n_total = len(pnl)

        result = SimResult(
            symbol=candidate.symbol,
            short_strike=candidate.short_strike,
            long_strike=candidate.long_strike,
            dte=candidate.dte,
            net_credit=candidate.net_credit,
            ev=round(ev, 2),
            pop=round(pop, 4),
            touch_prob=round(touch_prob, 4),
            cvar_95=round(cvar_95, 2),
            stop_prob=round(stop_prob, 4),
            avg_hold_days=round(avg_hold, 1),
            ev_per_bp=round(ev_per_bp, 6),
            mc_std_err=round(mc_std_err, 4),
            variance_reduction_factor=round(variance_reduction_factor, 2),
            pnl_p5=round(float(percentiles[0]), 2),
            pnl_p25=round(float(percentiles[1]), 2),
            pnl_p50=round(float(percentiles[2]), 2),
            pnl_p75=round(float(percentiles[3]), 2),
            pnl_p95=round(float(percentiles[4]), 2),
            exit_profit_target=round(float(np.sum(exit_reason == 1) / n_total), 4),
            exit_stop_loss=round(float(np.sum(exit_reason == 2) / n_total), 4),
            exit_dte=round(float(np.sum(exit_reason == 3) / n_total), 4),
            exit_expiration=round(float(np.sum(exit_reason == 0) / n_total), 4),
        )

        if return_raw_pnl:
            return result, raw_pnl
        return result

    def _apply_control_variate(
        self,
        pnl: np.ndarray,
        candidate: PCSCandidate,
        prices: np.ndarray,
        ivs: np.ndarray,
    ) -> tuple[np.ndarray, float]:
        """Apply BSM European spread price as a control variate.

        Uses the analytical BSM spread value at expiration as a correlated
        control. When exit rules dominate (high early-exit rate), beta
        auto-shrinks toward 0 so the adjustment does no harm.

        Returns (adjusted_pnl, variance_reduction_factor).
        """
        cfg = self.config

        # European PCS payoff at expiration (no early exit, no slippage)
        S_final = prices[:, -1]
        credit = candidate.net_credit * 100
        short_intrinsic = np.maximum(candidate.short_strike - S_final, 0.0) * 100
        long_intrinsic = np.maximum(candidate.long_strike - S_final, 0.0) * 100
        european_pnl = credit - (short_intrinsic - long_intrinsic)

        # Analytical BSM European spread value (scalar)
        T = candidate.dte / 252.0
        iv0 = ivs[0, 0]
        short_iv_ratio = candidate.short_iv / max(candidate.short_iv, 1e-8)
        long_iv_ratio = candidate.long_iv / max(candidate.short_iv, 1e-8)
        S0 = candidate.underlying_price
        S0_arr = np.array([S0])
        iv0_arr = np.array([iv0])
        short_put_analytical = bsm_put_price_vec(
            S0_arr, candidate.short_strike, T, cfg.risk_free_rate, iv0_arr * short_iv_ratio
        )[0]
        long_put_analytical = bsm_put_price_vec(
            S0_arr, candidate.long_strike, T, cfg.risk_free_rate, iv0_arr * long_iv_ratio
        )[0]
        analytical_spread = (short_put_analytical - long_put_analytical) * 100
        analytical_value = credit - analytical_spread  # Analytical expected European P&L approx

        # Compute control variate beta
        valid = ~np.isnan(pnl)
        if np.sum(valid) < 10:
            return pnl, 1.0

        cov_matrix = np.cov(pnl[valid], european_pnl[valid])
        var_control = cov_matrix[1, 1]
        if var_control < 1e-12:
            return pnl, 1.0

        beta = cov_matrix[0, 1] / var_control

        # Adjust P&L
        pnl_adjusted = pnl.copy()
        pnl_adjusted[valid] = pnl[valid] - beta * (european_pnl[valid] - analytical_value)

        # Variance reduction factor
        var_raw = np.var(pnl[valid])
        var_adj = np.var(pnl_adjusted[valid])
        vrf = var_raw / var_adj if var_adj > 1e-12 else 1.0

        return pnl_adjusted, vrf

    def simulate(
        self,
        candidate: PCSCandidate,
        n_paths: int | None = None,
    ) -> SimResult:
        """Run full MC simulation for a single PCS candidate.

        Uses the calibrated vol_mean_level for path generation (ATM-like vol),
        not the per-strike IV which inflates deep OTM dynamics.
        """
        # Use calibrated ATM vol for path generation, not per-strike skew IV
        atm_iv = (
            self.config.vol_mean_level if self.config.vol_mean_level > 0 else candidate.short_iv
        )

        prices, ivs = self._generate_paths(
            S0=candidate.underlying_price,
            iv0=atm_iv,
            dte=candidate.dte,
            n_paths=n_paths,
        )
        result = self._simulate_pcs(candidate, prices, ivs)

        # Importance sampling for tail risk (optional)
        if self.config.use_importance_sampling:
            cvar_is, cvar_se = self._estimate_cvar_is(candidate, n_paths)
            result.cvar_95_is = round(cvar_is, 2)
            result.cvar_95_se = round(cvar_se, 4)

        return result

    def _estimate_cvar_is(
        self,
        candidate: PCSCandidate,
        n_paths: int | None = None,
    ) -> tuple[float, float]:
        """Estimate CVaR95 via importance sampling with drift-tilted paths.

        Shifts the price drift downward toward the short strike so more paths
        land in the loss tail. Likelihood ratios correct for the bias.
        Returns (cvar_95_is, standard_error).
        """
        cfg = self.config
        n = n_paths or cfg.n_paths
        dte = candidate.dte
        dt = 1 / 252.0
        atm_iv = cfg.vol_mean_level if cfg.vol_mean_level > 0 else candidate.short_iv
        S0 = candidate.underlying_price

        # Compute drift tilt: shift mean toward short strike
        target_log_return = np.log(candidate.short_strike / S0)
        natural_drift = (cfg.risk_free_rate - 0.5 * atm_iv**2) * dte * dt
        mu_tilt = (target_log_return - natural_drift) / (dte * dt) if dte > 0 else 0.0

        # Generate tilted paths (no antithetic/stratified — clean IS estimate)
        rng = np.random.default_rng(cfg.seed + 1_000_000 if cfg.seed is not None else None)
        z_price = student_t.rvs(df=cfg.student_t_df, size=(n, dte), random_state=rng)
        z_vol = rng.standard_normal((n, dte))
        z_vol = cfg.leverage_effect * z_price + np.sqrt(1 - cfg.leverage_effect**2) * z_vol

        prices = np.zeros((n, dte + 1))
        ivs = np.zeros((n, dte + 1))
        prices[:, 0] = S0
        ivs[:, 0] = atm_iv
        log_lr = np.zeros(n)  # Log likelihood ratios

        kappa = cfg.vol_mean_revert_speed
        theta = cfg.vol_mean_level
        xi = cfg.vol_of_vol

        for t in range(dte):
            vol = ivs[:, t]
            vol_safe = np.maximum(vol, 0.01)

            # Tilted drift: add mu_tilt * dt to the natural drift
            natural = (cfg.risk_free_rate - 0.5 * vol_safe**2) * dt
            tilted_drift = natural + mu_tilt * dt
            prices[:, t + 1] = prices[:, t] * np.exp(
                tilted_drift + vol_safe * np.sqrt(dt) * z_price[:, t]
            )

            # Accumulate log likelihood ratio (Girsanov-like correction)
            # LR = exp(-mu_tilt * z * sqrt(dt) / vol - 0.5 * mu_tilt^2 * dt / vol^2)
            # approximation for Student-t: use Gaussian LR as proxy
            log_lr -= mu_tilt * np.sqrt(dt) * z_price[:, t] / vol_safe
            log_lr -= 0.5 * (mu_tilt**2) * dt / (vol_safe**2)

            iv_change = kappa * (theta - vol) * dt + xi * vol_safe * np.sqrt(dt) * z_vol[:, t]
            ivs[:, t + 1] = np.maximum(vol + iv_change, 0.01)

        # Simulate PCS on tilted paths to get raw P&L
        _, raw_pnl = self._simulate_pcs(candidate, prices, ivs, return_raw_pnl=True)

        # Compute likelihood ratios
        lr = np.exp(np.clip(log_lr, -50, 50))  # Clip to avoid overflow

        # Weighted CVaR95: find the 5th percentile of the IS-weighted distribution
        valid = ~np.isnan(raw_pnl)
        pnl_valid = raw_pnl[valid]
        lr_valid = lr[valid]

        if len(pnl_valid) == 0:
            return 0.0, 0.0

        # Sort by P&L, accumulate weights to find 5th percentile
        sort_idx = np.argsort(pnl_valid)
        pnl_sorted = pnl_valid[sort_idx]
        lr_sorted = lr_valid[sort_idx]
        weights = lr_sorted / lr_sorted.sum()
        cum_weights = np.cumsum(weights)

        # Find tail: everything up to the is_tail_quantile
        tail_mask = cum_weights <= cfg.is_tail_quantile
        if not np.any(tail_mask):
            tail_mask[0] = True  # At least one point

        tail_pnl = pnl_sorted[tail_mask]
        tail_lr = lr_sorted[tail_mask]
        tail_weights = tail_lr / tail_lr.sum()
        cvar_is = float(np.sum(tail_pnl * tail_weights))

        # Standard error via effective sample size
        ess = (lr_valid.sum() ** 2) / (lr_valid**2).sum() if (lr_valid**2).sum() > 0 else 1.0
        cvar_se = float(np.std(pnl_valid * lr_valid) / np.sqrt(max(ess, 1.0)))

        return cvar_is, cvar_se
