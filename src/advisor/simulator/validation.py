"""Validation module — resolve MC predictions against historical prices."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Callable

import numpy as np
import pandas as pd

from advisor.simulator.engine import bsm_put_price_vec
from advisor.simulator.models import SimConfig

logger = logging.getLogger(__name__)


@dataclass
class ResolvedOutcome:
    """Result of resolving a single prediction against historical prices."""

    candidate_id: str
    symbol: str
    actual_profit: float  # 1.0 if profitable, 0.0 if not
    actual_touch: float  # 1.0 if price touched short strike, 0.0 if not
    actual_stop: float  # 1.0 if stopped out, 0.0 if not
    actual_pnl: float  # Realized P&L in dollars
    exit_reason: str  # "profit_target", "stop_loss", "dte_close", "expiration"
    exit_day: int  # Day of exit (0-based from entry)


@dataclass
class BacktestValidationResult:
    """Aggregate result from historical replay validation."""

    n_predictions: int = 0
    n_resolved: int = 0
    pop_brier: float | None = None
    touch_brier: float | None = None
    stop_brier: float | None = None
    ev_mae: float | None = None
    ev_correlation: float | None = None
    calibration_buckets: list[dict] = field(default_factory=list)
    per_trade: list[dict] = field(default_factory=list)


def resolve_single_outcome(
    symbol: str,
    entry_date: date,
    expiration: str,
    short_strike: float,
    long_strike: float,
    net_credit: float,
    width: float,
    short_iv: float,
    long_iv: float,
    config: SimConfig | None = None,
    data_provider=None,
) -> ResolvedOutcome:
    """Walk historical OHLCV day-by-day to resolve a PCS prediction.

    Fetches price data from (entry_date - 45 days) through expiration,
    computes rolling 30-day HV as IV proxy, and simulates the daily exit logic
    matching the MC engine exactly.
    """
    from advisor.data.yahoo import YahooDataProvider

    cfg = config or SimConfig()
    provider = data_provider or YahooDataProvider()

    exp_date = date.fromisoformat(expiration) if isinstance(expiration, str) else expiration

    # Fetch OHLCV with extra buffer for rolling HV window
    fetch_start = entry_date - timedelta(days=60)
    fetch_end = exp_date + timedelta(days=1)

    df = provider.get_stock_history(symbol, fetch_start, fetch_end)

    # Compute rolling 30-day HV as IV proxy (same as options_backtester)
    log_returns = np.log(df["Close"] / df["Close"].shift(1))
    rolling_hv = log_returns.rolling(30).std() * np.sqrt(252)
    df = df.copy()
    df["hv"] = rolling_hv.values

    # Trim to entry_date through expiration
    entry_idx = df.index.searchsorted(pd.Timestamp(entry_date))
    exp_idx = df.index.searchsorted(pd.Timestamp(exp_date), side="right")
    trade_df = df.iloc[entry_idx:exp_idx]

    if len(trade_df) < 2:
        # Not enough data — return expired at entry
        return ResolvedOutcome(
            candidate_id="",
            symbol=symbol,
            actual_profit=0.0,
            actual_touch=0.0,
            actual_stop=0.0,
            actual_pnl=0.0,
            exit_reason="insufficient_data",
            exit_day=0,
        )

    # Preserve IV skew ratio from original candidate
    if short_iv > 0:
        short_iv_ratio = 1.0
        long_iv_ratio = long_iv / short_iv
    else:
        short_iv_ratio = 1.0
        long_iv_ratio = 1.0

    credit = net_credit * 100  # Per contract
    max_loss = (width - net_credit) * 100
    slippage = cfg.slippage_pct * width * 100
    profit_threshold = credit * cfg.profit_target_pct
    stop_threshold = credit * cfg.stop_loss_multiplier

    total_dte = (exp_date - entry_date).days

    # Compute entry spread value at t=0 using BSM
    row0 = trade_df.iloc[0]
    S0 = float(row0["Close"])
    hv0 = float(row0["hv"]) if not np.isnan(row0["hv"]) else short_iv
    T0 = total_dte / 252.0

    S0_arr = np.array([S0])
    short_iv_0 = np.array([hv0 * short_iv_ratio])
    long_iv_0 = np.array([hv0 * long_iv_ratio])
    short_put_0 = bsm_put_price_vec(S0_arr, short_strike, T0, cfg.risk_free_rate, short_iv_0)[0]
    long_put_0 = bsm_put_price_vec(S0_arr, long_strike, T0, cfg.risk_free_rate, long_iv_0)[0]
    entry_spread_value = (short_put_0 - long_put_0) * 100

    touched = False
    exit_reason = "expiration"
    exit_day = len(trade_df) - 1
    final_pnl = 0.0

    for day in range(1, len(trade_df)):
        row = trade_df.iloc[day]
        S = float(row["Close"])
        low = float(row["Low"])
        hv = float(row["hv"]) if not np.isnan(row["hv"]) else hv0

        # Touch detection
        if low <= short_strike:
            touched = True

        # Remaining DTE
        current_date = trade_df.index[day]
        if hasattr(current_date, "date"):
            current_date = current_date.date()
        remaining_dte = (exp_date - current_date).days

        T = remaining_dte / 252.0

        # Reprice spread
        S_arr = np.array([S])
        s_iv = np.array([hv * short_iv_ratio])
        l_iv = np.array([hv * long_iv_ratio])
        short_put = bsm_put_price_vec(S_arr, short_strike, T, cfg.risk_free_rate, s_iv)[0]
        long_put = bsm_put_price_vec(S_arr, long_strike, T, cfg.risk_free_rate, l_iv)[0]
        current_spread = (short_put - long_put) * 100

        # Unrealized P&L
        unrealized_pnl = entry_spread_value - current_spread
        change = current_spread - entry_spread_value
        mtm_pnl = credit - change - slippage

        # DTE close check (same priority as MC engine)
        if remaining_dte <= cfg.close_at_dte and remaining_dte > 0:
            final_pnl = mtm_pnl
            exit_reason = "dte_close"
            exit_day = day
            break

        # Profit target
        if unrealized_pnl >= profit_threshold:
            final_pnl = mtm_pnl
            exit_reason = "profit_target"
            exit_day = day
            break

        # Stop loss
        if unrealized_pnl <= -stop_threshold:
            final_pnl = mtm_pnl
            exit_reason = "stop_loss"
            exit_day = day
            break
    else:
        # Held to expiration — settle at intrinsic
        S_final = float(trade_df.iloc[-1]["Close"])
        short_intrinsic = max(short_strike - S_final, 0.0) * 100
        long_intrinsic = max(long_strike - S_final, 0.0) * 100
        spread_intrinsic = short_intrinsic - long_intrinsic
        final_pnl = credit - (spread_intrinsic - entry_spread_value)

    # Cap at max_loss
    final_pnl = max(final_pnl, -max_loss)

    return ResolvedOutcome(
        candidate_id="",
        symbol=symbol,
        actual_profit=1.0 if final_pnl > 0 else 0.0,
        actual_touch=1.0 if touched else 0.0,
        actual_stop=1.0 if exit_reason == "stop_loss" else 0.0,
        actual_pnl=round(final_pnl, 2),
        exit_reason=exit_reason,
        exit_day=exit_day,
    )


def resolve_outcomes(
    store, config: SimConfig | None = None, data_provider=None
) -> list[ResolvedOutcome]:
    """Batch resolve all pending calibration records whose candidates have expired.

    1. Query pending calibrations (actual_profit IS NULL AND expiration < today)
    2. For each, call resolve_single_outcome()
    3. Update DB with actual outcomes
    4. Return list of resolved outcomes
    """
    pending = store.get_pending_calibrations()
    if not pending:
        return []

    outcomes = []
    for rec in pending:
        exp = rec["expiration"]
        exp_date = date.fromisoformat(exp) if isinstance(exp, str) else exp

        # Skip future expirations (shouldn't happen due to SQL filter, but safety check)
        if exp_date >= date.today():
            continue

        # Determine entry date from created_at
        created = rec["created_at"]
        if isinstance(created, str):
            entry_date = datetime.fromisoformat(created).date()
        else:
            entry_date = created.date() if hasattr(created, "date") else date.today()

        try:
            outcome = resolve_single_outcome(
                symbol=rec["symbol"],
                entry_date=entry_date,
                expiration=exp,
                short_strike=rec["short_strike"],
                long_strike=rec["long_strike"],
                net_credit=rec["net_credit"],
                width=rec["width"],
                short_iv=rec["short_iv"],
                long_iv=rec["long_iv"],
                config=config,
                data_provider=data_provider,
            )
            outcome.candidate_id = rec["candidate_id"]

            # Update DB
            store.update_calibration_outcome(
                candidate_id=rec["candidate_id"],
                actual_profit=outcome.actual_profit,
                actual_touch=outcome.actual_touch,
                actual_stop=outcome.actual_stop,
                actual_pnl=outcome.actual_pnl,
            )
            outcomes.append(outcome)
        except Exception as e:
            logger.warning("Failed to resolve %s (%s): %s", rec["symbol"], rec["candidate_id"], e)

    return outcomes


def backtest_validate(
    store,
    symbol: str,
    start: str,
    end: str,
    config: SimConfig | None = None,
    n_paths: int = 10_000,
    data_provider=None,
    progress_callback: Callable[[str], None] | None = None,
) -> BacktestValidationResult:
    """Historical replay: re-run MC sim on past chain snapshots and validate against actuals.

    1. Query chain_snapshots grouped by snapshot date
    2. Calibrate engine once per symbol
    3. For each snapshot date: generate candidates, take top 5, run MC sim
    4. Resolve actual outcome via resolve_single_outcome()
    5. Compute aggregate Brier scores and calibration buckets
    """
    from advisor.simulator.calibration import calibrate
    from advisor.simulator.candidates import generate_pcs_candidates, pre_score_candidates
    from advisor.simulator.engine import MonteCarloEngine

    cfg = config or SimConfig(n_paths=n_paths)
    progress = progress_callback or (lambda msg: None)

    # Step 1: Get snapshots
    snapshots = store.get_chain_snapshots_by_date_range(symbol, start, end)
    if not snapshots:
        return BacktestValidationResult()

    # Group by snapshot date
    by_date: dict[str, list[dict]] = {}
    for snap in snapshots:
        snap_date = snap["snapshot_at"][:10]  # YYYY-MM-DD
        by_date.setdefault(snap_date, []).append(snap)

    # Step 2: Calibrate engine
    progress(f"Calibrating {symbol}...")
    try:
        cal_cfg = calibrate(symbol, cfg)
    except Exception:
        cal_cfg = cfg
    engine = MonteCarloEngine(cal_cfg)

    per_trade: list[dict] = []
    predicted_pops: list[float] = []
    actual_profits: list[float] = []
    predicted_touches: list[float] = []
    actual_touches: list[float] = []
    predicted_stops: list[float] = []
    actual_stops: list[float] = []
    predicted_evs: list[float] = []
    actual_pnls: list[float] = []

    # Step 3-4: Process each snapshot date
    for snap_date_str, chain in sorted(by_date.items()):
        progress(f"Processing {symbol} {snap_date_str}...")
        snap_date = date.fromisoformat(snap_date_str)

        # Generate candidates from this snapshot's chain
        candidates = generate_pcs_candidates(chain, symbol, cfg)
        candidates = pre_score_candidates(candidates, 50.0)

        # Take top 5 by sell_score
        top_candidates = candidates[:5]

        for cand in top_candidates:
            exp_date = date.fromisoformat(cand.expiration)

            # Skip if expiration is in the future
            if exp_date >= date.today():
                continue

            # Run MC simulation
            try:
                sim_result = engine.simulate(cand, n_paths=n_paths)
            except Exception as e:
                logger.warning("Sim failed for %s: %s", cand.short_strike, e)
                continue

            # Resolve actual outcome
            try:
                outcome = resolve_single_outcome(
                    symbol=symbol,
                    entry_date=snap_date,
                    expiration=cand.expiration,
                    short_strike=cand.short_strike,
                    long_strike=cand.long_strike,
                    net_credit=cand.net_credit,
                    width=cand.width,
                    short_iv=cand.short_iv,
                    long_iv=cand.long_iv,
                    config=cfg,
                    data_provider=data_provider,
                )
            except Exception as e:
                logger.warning("Resolve failed for %s %s: %s", symbol, snap_date_str, e)
                continue

            if outcome.exit_reason == "insufficient_data":
                continue

            # Collect predictions vs actuals
            predicted_pops.append(sim_result.pop)
            actual_profits.append(outcome.actual_profit)
            predicted_touches.append(sim_result.touch_prob)
            actual_touches.append(outcome.actual_touch)
            predicted_stops.append(sim_result.stop_prob)
            actual_stops.append(outcome.actual_stop)
            predicted_evs.append(sim_result.ev)
            actual_pnls.append(outcome.actual_pnl)

            per_trade.append(
                {
                    "date": snap_date_str,
                    "short_strike": cand.short_strike,
                    "long_strike": cand.long_strike,
                    "expiration": cand.expiration,
                    "predicted_pop": sim_result.pop,
                    "predicted_touch": sim_result.touch_prob,
                    "predicted_stop": sim_result.stop_prob,
                    "predicted_ev": sim_result.ev,
                    "actual_profit": outcome.actual_profit,
                    "actual_touch": outcome.actual_touch,
                    "actual_stop": outcome.actual_stop,
                    "actual_pnl": outcome.actual_pnl,
                    "exit_reason": outcome.exit_reason,
                    "exit_day": outcome.exit_day,
                }
            )

    n_resolved = len(predicted_pops)
    if n_resolved == 0:
        return BacktestValidationResult(
            n_predictions=len(by_date),
            n_resolved=0,
        )

    # Step 5: Compute aggregate metrics
    pred_pop = np.array(predicted_pops)
    act_prof = np.array(actual_profits)
    pred_touch = np.array(predicted_touches)
    act_touch = np.array(actual_touches)
    pred_stop = np.array(predicted_stops)
    act_stop = np.array(actual_stops)
    pred_ev = np.array(predicted_evs)
    act_pnl = np.array(actual_pnls)

    pop_brier = float(np.mean((pred_pop - act_prof) ** 2))
    touch_brier = float(np.mean((pred_touch - act_touch) ** 2))
    stop_brier = float(np.mean((pred_stop - act_stop) ** 2))
    ev_mae = float(np.mean(np.abs(pred_ev - act_pnl)))

    if n_resolved >= 2:
        ev_correlation = float(np.corrcoef(pred_ev, act_pnl)[0, 1])
        if np.isnan(ev_correlation):
            ev_correlation = 0.0
    else:
        ev_correlation = None

    calibration_buckets = _compute_calibration_buckets(predicted_pops, actual_profits)

    return BacktestValidationResult(
        n_predictions=len(by_date),
        n_resolved=n_resolved,
        pop_brier=round(pop_brier, 4),
        touch_brier=round(touch_brier, 4),
        stop_brier=round(stop_brier, 4),
        ev_mae=round(ev_mae, 2),
        ev_correlation=round(ev_correlation, 4) if ev_correlation is not None else None,
        calibration_buckets=calibration_buckets,
        per_trade=per_trade,
    )


def _compute_calibration_buckets(predicted: list[float], actual: list[float]) -> list[dict]:
    """Bin predictions into 10 equal-width buckets and compute actual rate per bin.

    Returns list of dicts with: bucket, predicted_mean, actual_mean, count.
    """
    buckets = []
    for i in range(10):
        lo = i * 0.1
        hi = (i + 1) * 0.1
        bucket_label = f"{int(lo * 100)}-{int(hi * 100)}%"

        # Find predictions in this bucket
        indices = [j for j, p in enumerate(predicted) if lo <= p < hi or (i == 9 and p == 1.0)]

        if indices:
            pred_vals = [predicted[j] for j in indices]
            act_vals = [actual[j] for j in indices]
            buckets.append(
                {
                    "bucket": bucket_label,
                    "predicted_mean": round(float(np.mean(pred_vals)), 4),
                    "actual_mean": round(float(np.mean(act_vals)), 4),
                    "count": len(indices),
                }
            )
        else:
            buckets.append(
                {
                    "bucket": bucket_label,
                    "predicted_mean": round((lo + hi) / 2, 4),
                    "actual_mean": 0.0,
                    "count": 0,
                }
            )

    return buckets
