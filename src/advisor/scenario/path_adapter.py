"""Path adapter — generates forward OHLCV paths by reusing the MC engine."""

from __future__ import annotations

import logging
from datetime import date, timedelta

import numpy as np
import pandas as pd
import yfinance as yf

from advisor.scenario.models import ScenarioConfig, ScenarioDefinition
from advisor.simulator.calibration import calibrate
from advisor.simulator.engine import MonteCarloEngine
from advisor.simulator.models import SimConfig

logger = logging.getLogger(__name__)


def generate_scenario_paths(
    symbol: str,
    scenario: ScenarioDefinition,
    config: ScenarioConfig,
    sim_config: SimConfig | None = None,
) -> np.ndarray:
    """Generate forward close-price paths for a scenario.

    Returns array of shape (n_paths, dte+1) starting at current price.
    The scenario's drift and vol_multiplier override the calibrated parameters.
    """
    calibrated = sim_config or calibrate(symbol)

    # Override drift via risk_free_rate (engine uses it as GBM drift)
    # Scale vol via vol_mean_level multiplier
    scenario_cfg = calibrated.model_copy(
        update={
            "risk_free_rate": scenario.annual_drift,
            "vol_mean_level": calibrated.vol_mean_level * scenario.vol_multiplier,
            "n_paths": config.n_paths,
            "seed": config.seed,
            # Disable options-specific settings
            "use_control_variate": False,
            "use_importance_sampling": False,
        }
    )

    engine = MonteCarloEngine(scenario_cfg)

    # Current price as S0, calibrated vol as iv0
    ticker = yf.Ticker(symbol)
    hist = ticker.history(period="5d")
    if hist.empty:
        raise ValueError(f"No price data for {symbol}")
    s0 = float(hist["Close"].iloc[-1])
    iv0 = scenario_cfg.vol_mean_level

    prices, _ivs = engine._generate_paths(s0, iv0, config.dte, config.n_paths)
    return prices


def synthesize_ohlcv(
    close_paths: np.ndarray,
    start_date: date,
    historical_volume_mean: float = 5_000_000.0,
    historical_avg_abs_return: float = 0.01,
    iv: float = 0.25,
    seed: int | None = None,
) -> list[pd.DataFrame]:
    """Convert close-only paths into OHLCV DataFrames.

    Returns one DataFrame per path with columns: Open, High, Low, Close, Volume
    and a DatetimeIndex of business days starting from start_date.
    """
    rng = np.random.default_rng(seed)
    n_paths, n_days = close_paths.shape
    result = []

    # Generate business day index
    dates = pd.bdate_range(start=start_date, periods=n_days, freq="B")

    for i in range(n_paths):
        closes = close_paths[i]

        # Open = previous close (first day open = first close)
        opens = np.empty_like(closes)
        opens[0] = closes[0]
        opens[1:] = closes[:-1]

        # Intraday noise scale proportional to IV
        daily_vol = iv / np.sqrt(252)
        noise = np.abs(rng.standard_normal(n_days)) * daily_vol

        # High >= max(open, close), Low <= min(open, close)
        max_oc = np.maximum(opens, closes)
        min_oc = np.minimum(opens, closes)
        highs = max_oc * (1 + noise)
        lows = min_oc * (1 - noise)

        # Volume: historical replay with noise and volume-price correlation
        abs_returns = np.zeros(n_days)
        abs_returns[1:] = np.abs(closes[1:] / closes[:-1] - 1)
        vol_noise = rng.standard_normal(n_days)
        safe_avg = max(historical_avg_abs_return, 1e-6)
        volume = historical_volume_mean * (1 + 0.3 * vol_noise) * (1 + 2 * abs_returns / safe_avg)
        volume = np.maximum(volume, 1000).astype(int)

        df = pd.DataFrame(
            {
                "Open": opens,
                "High": highs,
                "Low": lows,
                "Close": closes,
                "Volume": volume,
            },
            index=dates[:n_days],
        )
        result.append(df)

    return result


def fetch_warmup_data(symbol: str, warmup_bars: int = 200) -> tuple[pd.DataFrame, float, float]:
    """Fetch historical OHLCV for indicator warmup.

    Returns (warmup_df, avg_volume, avg_abs_return).
    """
    from advisor.data.yahoo import YahooDataProvider

    provider = YahooDataProvider()
    # Fetch enough history for warmup plus buffer
    end = date.today()
    start = end - timedelta(days=int(warmup_bars * 1.6))
    df = provider.get_stock_history(symbol, start, end)

    if len(df) < warmup_bars:
        logger.warning(
            "Only %d warmup bars available for %s (requested %d)",
            len(df),
            symbol,
            warmup_bars,
        )

    df = df.tail(warmup_bars)

    avg_volume = float(df["Volume"].mean()) if "Volume" in df.columns else 5_000_000.0
    returns = df["Close"].pct_change().dropna()
    avg_abs_return = float(returns.abs().mean()) if len(returns) > 0 else 0.01

    # Ensure tz-naive for Backtrader
    if hasattr(df.index, "tz") and df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    return df, avg_volume, avg_abs_return


def build_full_feeds(
    warmup_df: pd.DataFrame,
    simulated_ohlcv: list[pd.DataFrame],
) -> list[pd.DataFrame]:
    """Prepend warmup history to each simulated OHLCV path.

    Returns list of complete DataFrames ready for Backtrader.
    """
    result = []
    for sim_df in simulated_ohlcv:
        # Ensure no overlap between warmup and simulated dates
        sim_start = sim_df.index[0]
        warmup_trimmed = warmup_df[warmup_df.index < sim_start]
        combined = pd.concat([warmup_trimmed, sim_df])
        result.append(combined)
    return result
