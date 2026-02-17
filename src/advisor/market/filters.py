"""Three-layer filter pipeline for market-wide scanning.

Filter funnel (fast → slow, each layer reduces cost of the next):
  Layer 1: Volume + Market Cap gate  (yf.Tickers batch)
  Layer 2: Sector/Industry filter    (reuses Layer 1 data, zero cost)
  Layer 3: Technical pre-screen      (yf.download batch OHLCV)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

_BATCH_SIZE = 50


@dataclass
class FilterConfig:
    """Configuration for the filter pipeline."""

    min_avg_volume: int = 500_000
    min_market_cap: float = 2_000_000_000
    include_sectors: list[str] | None = None
    exclude_sectors: list[str] | None = None
    technical_lookback_days: int = 300


@dataclass
class FilterStats:
    """Running statistics from the filter funnel."""

    universe_total: int = 0
    after_volume_cap: int = 0
    after_sector: int = 0
    after_technical: int = 0
    fetch_error_count: int = 0

    @property
    def volume_cap_rejected_count(self) -> int:
        return self.universe_total - self.after_volume_cap - self.fetch_error_count

    @property
    def sector_rejected_count(self) -> int:
        return self.after_volume_cap - self.after_sector

    @property
    def technical_rejected_count(self) -> int:
        return self.after_sector - self.after_technical


@dataclass
class TickerInfo:
    """Cached info from yf.Tickers for a single symbol."""

    symbol: str
    market_cap: float = 0.0
    avg_volume: float = 0.0
    sector: str = ""
    industry: str = ""


def _batch_ticker_info(
    symbols: list[str],
    on_progress: callable | None = None,
) -> dict[str, TickerInfo]:
    """Fetch market cap, volume, sector for symbols using yf.Tickers in batches."""
    info_map: dict[str, TickerInfo] = {}
    total_batches = (len(symbols) + _BATCH_SIZE - 1) // _BATCH_SIZE

    for i in range(0, len(symbols), _BATCH_SIZE):
        batch = symbols[i : i + _BATCH_SIZE]
        batch_num = i // _BATCH_SIZE + 1
        logger.debug(
            "Fetching ticker info batch %d/%d (%d symbols)", batch_num, total_batches, len(batch)
        )

        try:
            tickers = yf.Tickers(" ".join(batch))
            for sym in batch:
                try:
                    info = tickers.tickers[sym].info
                    info_map[sym] = TickerInfo(
                        symbol=sym,
                        market_cap=float(info.get("marketCap") or 0),
                        avg_volume=float(info.get("averageVolume") or 0),
                        sector=str(info.get("sector") or ""),
                        industry=str(info.get("industry") or ""),
                    )
                except Exception:
                    logger.debug("Failed to get info for %s", sym)
        except Exception as e:
            logger.warning("Batch ticker info failed for batch %d: %s", batch_num, e)

        if on_progress:
            on_progress(len(batch))

    return info_map


def _compute_rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI using Wilder's smoothing (exponential moving average)."""
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _check_strategy_technical(
    close: pd.Series,
    volume: pd.Series,
    strategy_name: str,
) -> bool:
    """Pure pandas/numpy technical pre-screen on the last bar.

    Returns True if the symbol passes the quick technical filter for the
    given strategy. This is a cheap approximation — the full confluence
    pipeline will do the real signal check.
    """
    if len(close) < 200:
        return False

    price = close.iloc[-1]

    if strategy_name == "momentum_breakout":
        sma_20 = close.rolling(20).mean().iloc[-1]
        avg_vol = volume.rolling(20).mean().iloc[-1]
        last_vol = volume.iloc[-1]
        return bool(price > sma_20 and last_vol > 1.5 * avg_vol)

    elif strategy_name == "buy_the_dip":
        rsi = _compute_rsi(close, 14)
        sma_50 = close.rolling(50).mean().iloc[-1]
        sma_200 = close.rolling(200).mean().iloc[-1]
        return bool(rsi.iloc[-1] < 30 and price < sma_50 and price > sma_200)

    elif strategy_name == "sma_crossover":
        sma_20 = close.rolling(20).mean()
        sma_50 = close.rolling(50).mean()
        # Check if SMA(20) crossed above SMA(50) in last 5 bars
        for i in range(-5, 0):
            try:
                if sma_20.iloc[i - 1] <= sma_50.iloc[i - 1] and sma_20.iloc[i] > sma_50.iloc[i]:
                    return True
            except IndexError:
                continue
        return False

    elif strategy_name == "pead":
        # Volume spike >2x 20-day average in last 5 bars (proxy for earnings event)
        avg_vol = volume.rolling(20).mean()
        has_spike = False
        for i in range(-5, 0):
            try:
                if volume.iloc[i] > 2.0 * avg_vol.iloc[i]:
                    has_spike = True
                    break
            except IndexError:
                continue
        if not has_spike:
            return False
        # Price faded: current price < 98% of 10-day high
        high_10 = close.rolling(10).max().iloc[-1]
        return bool(price < 0.98 * high_10)

    elif strategy_name == "mean_reversion":
        rsi = _compute_rsi(close, 14)
        ema_20 = close.ewm(span=20, adjust=False).mean().iloc[-1]
        # ATR approx using close-to-close changes (no high/low in filter signature)
        daily_range = close.diff().abs()
        atr_14 = daily_range.rolling(14).mean().iloc[-1]
        avg_vol = volume.rolling(20).mean().iloc[-1]
        last_vol = volume.iloc[-1]
        price_below_ema = ema_20 - price
        return bool(
            rsi.iloc[-1] < 25
            and atr_14 > 0
            and price_below_ema > 2.0 * atr_14
            and avg_vol > 0
            and last_vol > 1.5 * avg_vol
        )

    else:
        # Default: price > SMA(50)
        sma_50 = close.rolling(50).mean().iloc[-1]
        return bool(price > sma_50)


def apply_filters(
    symbols: list[str],
    strategy_name: str,
    config: FilterConfig,
    sector_map: dict[str, str] | None = None,
    on_progress: callable | None = None,
) -> tuple[list[str], FilterStats]:
    """Run the 3-layer filter pipeline.

    Args:
        symbols: Full universe of ticker symbols.
        strategy_name: Strategy name for technical pre-screen logic.
        config: Filter thresholds.
        sector_map: Optional mapping of symbol → GICS sector from the universe
            data. Used for sector filtering so names match Wikipedia/GICS
            conventions rather than yfinance's shorter names.
        on_progress: Callback(phase: str, advance: int) for progress updates.

    Returns:
        (qualifying_symbols, filter_stats)
    """
    stats = FilterStats(universe_total=len(symbols))

    def _progress(phase: str, advance: int = 1) -> None:
        if on_progress:
            on_progress(phase, advance)

    # ── Layer 1: Volume + Market Cap ──────────────────────────────────────
    logger.info("Layer 1: Fetching ticker info for %d symbols...", len(symbols))

    def _ticker_progress(n: int) -> None:
        _progress("ticker_info", n)

    info_map = _batch_ticker_info(symbols, on_progress=_ticker_progress)
    stats.fetch_error_count = len(symbols) - len(info_map)

    passed_vol_cap: list[str] = []
    for sym in symbols:
        info = info_map.get(sym)
        if info is None:
            continue
        if info.avg_volume >= config.min_avg_volume and info.market_cap >= config.min_market_cap:
            passed_vol_cap.append(sym)

    stats.after_volume_cap = len(passed_vol_cap)
    logger.info(
        "Layer 1 result: %d/%d passed (volume >= %d, cap >= $%.0fB)",
        stats.after_volume_cap,
        stats.universe_total,
        config.min_avg_volume,
        config.min_market_cap / 1e9,
    )

    # ── Layer 2: Sector/Industry ──────────────────────────────────────────
    # Prefer universe sector_map (GICS names from Wikipedia) over yfinance
    # sector names, which use shorter labels (e.g. "Technology" vs
    # "Information Technology").
    passed_sector: list[str] = []
    for sym in passed_vol_cap:
        sector = (sector_map or {}).get(sym) or info_map[sym].sector
        if config.include_sectors and sector not in config.include_sectors:
            continue
        if config.exclude_sectors and sector in config.exclude_sectors:
            continue
        passed_sector.append(sym)

    stats.after_sector = len(passed_sector)
    logger.info(
        "Layer 2 result: %d/%d passed sector filter", stats.after_sector, stats.after_volume_cap
    )
    _progress("sector_done")

    if not passed_sector:
        stats.after_technical = 0
        return [], stats

    # ── Layer 3: Technical Pre-Screen ─────────────────────────────────────
    logger.info("Layer 3: Downloading OHLCV for %d symbols...", len(passed_sector))
    _progress("technical_start")

    try:
        ohlcv = yf.download(
            passed_sector,
            period=f"{config.technical_lookback_days}d",
            progress=False,
            threads=True,
        )
    except Exception as e:
        logger.error("OHLCV download failed: %s", e)
        stats.after_technical = 0
        return [], stats

    _progress("technical_download_done")

    passed_technical: list[str] = []
    for sym in passed_sector:
        try:
            if len(passed_sector) == 1:
                close = ohlcv["Close"]
                vol = ohlcv["Volume"]
            else:
                close = ohlcv["Close"][sym].dropna()
                vol = ohlcv["Volume"][sym].dropna()

            if len(close) < 50:
                continue

            if _check_strategy_technical(close, vol, strategy_name):
                passed_technical.append(sym)
        except Exception:
            logger.debug("Technical check failed for %s", sym)

    stats.after_technical = len(passed_technical)
    logger.info(
        "Layer 3 result: %d/%d passed technical pre-screen",
        stats.after_technical,
        stats.after_sector,
    )
    _progress("technical_done")

    return passed_technical, stats
