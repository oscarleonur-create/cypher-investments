"""Feature engineering — compute ML-ready features from OHLCV data."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

# Sector ETF mapping for relative strength calculation
SECTOR_ETFS = {
    "XLK": ["AAPL", "MSFT", "NVDA", "AVGO", "CRM", "ADBE", "AMD", "INTC", "CSCO", "ORCL"],
    "XLF": ["JPM", "BAC", "WFC", "GS", "MS", "C", "BLK", "SCHW", "AXP", "USB"],
    "XLE": ["XOM", "CVX", "COP", "SLB", "EOG", "MPC", "PSX", "VLO", "OXY", "HAL"],
    "XLV": ["UNH", "JNJ", "LLY", "PFE", "ABBV", "MRK", "TMO", "ABT", "DHR", "BMY"],
    "XLY": ["AMZN", "TSLA", "HD", "MCD", "NKE", "LOW", "SBUX", "TJX", "BKNG", "CMG"],
    "XLP": ["PG", "KO", "PEP", "COST", "WMT", "PM", "MDLZ", "CL", "MO", "KHC"],
    "XLI": ["HON", "UPS", "CAT", "GE", "BA", "RTX", "DE", "LMT", "MMM", "UNP"],
    "XLU": ["NEE", "DUK", "SO", "D", "AEP", "SRE", "EXC", "XEL", "ED", "WEC"],
    "XLRE": ["PLD", "AMT", "CCI", "EQIX", "PSA", "SPG", "O", "WELL", "DLR", "AVB"],
    "XLC": ["META", "GOOGL", "GOOG", "NFLX", "DIS", "CMCSA", "TMUS", "VZ", "T", "CHTR"],
    "XLB": ["LIN", "APD", "SHW", "FCX", "ECL", "NEM", "DOW", "NUE", "DD", "VMC"],
}

_SYMBOL_TO_SECTOR: dict[str, str] | None = None


def _get_sector_etf(symbol: str) -> str | None:
    """Look up sector ETF for a symbol."""
    global _SYMBOL_TO_SECTOR
    if _SYMBOL_TO_SECTOR is None:
        _SYMBOL_TO_SECTOR = {}
        for etf, members in SECTOR_ETFS.items():
            for s in members:
                _SYMBOL_TO_SECTOR[s] = etf
    return _SYMBOL_TO_SECTOR.get(symbol.upper())


def _compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    """Compute RSI from a price series."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _compute_macd(
    series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9
) -> tuple[pd.Series, pd.Series]:
    """Compute MACD line and signal line."""
    ema_fast = series.ewm(span=fast, min_periods=fast).mean()
    ema_slow = series.ewm(span=slow, min_periods=slow).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal, min_periods=signal).mean()
    return macd_line, signal_line


def _compute_bollinger(
    series: pd.Series, period: int = 20, num_std: float = 2.0
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute Bollinger Bands (upper, middle, lower)."""
    middle = series.rolling(period).mean()
    std = series.rolling(period).std()
    upper = middle + num_std * std
    lower = middle - num_std * std
    return upper, middle, lower


def _compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> pd.Series:
    """Compute Average True Range."""
    prev_close = close.shift(1)
    tr = pd.concat([high - low, (high - prev_close).abs(), (low - prev_close).abs()], axis=1).max(
        axis=1
    )
    return tr.rolling(period).mean()


# Features that use non-time-indexed snapshots (yfinance .info/.calendar/.options).
# Safe for live prediction (where they reflect current state) but leak future
# data when applied to historical dates during training.
SNAPSHOT_FEATURES = frozenset(
    {
        "earnings_proximity",
        "analyst_upside",
        "pe_vs_sector",
        "recommendation_score",
        "earnings_growth",
        "iv_rv_ratio",
        "put_call_oi_ratio",
        "iv_skew",
        "options_volume_ratio",
    }
)


class FeatureEngine:
    """Compute ML-ready features from OHLCV data using pure pandas."""

    def compute_features(self, symbol: str) -> dict[str, float]:
        """Compute a single snapshot of features for live prediction.

        Downloads recent data and returns the latest feature values.
        Includes snapshot features (fundamental/options) since they
        reflect the current state.
        """
        symbol = symbol.upper()
        try:
            df = yf.download(symbol, period="6mo", progress=False)
            if df.empty or len(df) < 60:
                logger.warning(f"Insufficient data for {symbol} ({len(df)} rows)")
                return {}

            # Flatten MultiIndex columns if present
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            features_df = self._compute_all_features(df, symbol)
            if features_df.empty:
                return {}

            row = features_df.iloc[-1]
            return {k: float(v) if pd.notna(v) else 0.0 for k, v in row.items()}
        except Exception as e:
            logger.error(f"Feature computation failed for {symbol}: {e}")
            return {}

    def compute_features_df(
        self,
        symbol: str,
        start: str | None = None,
        end: str | None = None,
        period: str = "2y",
        include_snapshots: bool = False,
    ) -> pd.DataFrame:
        """Compute rolling features for training over a date range.

        Args:
            symbol: Ticker symbol.
            start: Start date (YYYY-MM-DD). If set, ``period`` is ignored.
            end: End date (YYYY-MM-DD).
            period: yfinance period string (e.g. "2y", "5y"). Used when
                ``start`` is not provided.
            include_snapshots: If False (default), drops snapshot features
                (fundamental/options) that are not time-indexed and would
                leak future data during training.

        Returns a DataFrame indexed by date with feature columns.
        """
        symbol = symbol.upper()
        try:
            kwargs: dict[str, Any] = {"progress": False}
            if start:
                kwargs["start"] = start
            if end:
                kwargs["end"] = end
            if not start:
                kwargs["period"] = period

            df = yf.download(symbol, **kwargs)
            if df.empty or len(df) < 60:
                logger.warning(f"Insufficient data for {symbol}")
                return pd.DataFrame()

            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)

            features = self._compute_all_features(df, symbol)

            if not include_snapshots and not features.empty:
                drop_cols = [c for c in SNAPSHOT_FEATURES if c in features.columns]
                if drop_cols:
                    features = features.drop(columns=drop_cols)

            return features
        except Exception as e:
            logger.error(f"Feature computation failed for {symbol}: {e}")
            return pd.DataFrame()

    def _compute_all_features(self, df: pd.DataFrame, symbol: str) -> pd.DataFrame:
        """Build all feature columns from an OHLCV DataFrame.

        Cross-asset downloads (VIX, SPY, sector ETFs) are matched to the
        stock's date range so features are populated across the full history.
        """
        # Date range for cross-asset downloads — match the stock's span
        _dl_start = str(df.index[0].date())
        _dl_end = str(df.index[-1].date())
        close = df["Close"]
        high = df["High"]
        low = df["Low"]
        volume = df["Volume"]

        features = pd.DataFrame(index=df.index)

        # ── Momentum (6) ────────────────────────────────────────────────
        for n in [1, 5, 10, 20, 60]:
            features[f"ret_{n}d"] = close.pct_change(n)
        features["momentum_rank"] = features["ret_20d"].rolling(60).rank(pct=True)

        # ── Trend (4) ───────────────────────────────────────────────────
        sma_20 = close.rolling(20).mean()
        sma_50 = close.rolling(50).mean()
        features["sma_20_dist"] = (close - sma_20) / sma_20
        features["sma_50_dist"] = (close - sma_50) / sma_50

        macd_line, signal_line = _compute_macd(close)
        features["ema_12_26_diff"] = macd_line / close  # Normalize by price
        features["macd_signal_diff"] = (macd_line - signal_line) / close

        # ── Mean-reversion (3) ──────────────────────────────────────────
        features["rsi_14"] = _compute_rsi(close, 14) / 100  # Scale 0-1
        bb_upper, bb_middle, bb_lower = _compute_bollinger(close)
        bb_width = bb_upper - bb_lower
        features["bb_width"] = bb_width / bb_middle
        features["bb_pct"] = (close - bb_lower) / bb_width.replace(0, np.nan)

        # ── Volatility (4) ──────────────────────────────────────────────
        atr = _compute_atr(high, low, close, 14)
        features["atr_14"] = atr / close  # Normalize by price
        features["realized_vol_20"] = close.pct_change().rolling(20).std() * np.sqrt(252)
        features["realized_vol_5"] = close.pct_change().rolling(5).std() * np.sqrt(252)
        features["vol_ratio"] = features["realized_vol_5"] / features["realized_vol_20"].replace(
            0, np.nan
        )

        # ── Volume (3) ──────────────────────────────────────────────────
        vol_mean = volume.rolling(20).mean()
        vol_std = volume.rolling(20).std()
        features["volume_zscore"] = (volume - vol_mean) / vol_std.replace(0, np.nan)
        features["volume_trend"] = vol_mean / volume.rolling(60).mean().replace(0, np.nan)
        # OBV slope: slope of OBV over last 10 days
        obv = (np.sign(close.diff()) * volume).cumsum()
        features["obv_slope"] = obv.rolling(10).apply(
            lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == 10 else 0, raw=True
        )
        # Normalize OBV slope by average volume
        features["obv_slope"] = features["obv_slope"] / vol_mean.replace(0, np.nan)

        # ── Cross-asset (4) ─────────────────────────────────────────────
        features["vix_level"] = 0.0
        features["vix_change_5d"] = 0.0
        features["vix_regime"] = 1.0  # default: normal
        features["sector_rel_strength"] = 0.0

        try:
            vix = yf.download(
                "^VIX",
                start=_dl_start,
                end=_dl_end,
                progress=False,
            )
            if not vix.empty:
                if isinstance(vix.columns, pd.MultiIndex):
                    vix.columns = vix.columns.get_level_values(0)
                vix_close = vix["Close"].reindex(df.index, method="ffill")
                features["vix_level"] = vix_close / 100  # Scale down
                features["vix_change_5d"] = vix_close.pct_change(5)
                # VIX regime: 0=calm(<15), 1=normal(15-25), 2=elevated(>25)
                features["vix_regime"] = (
                    pd.cut(
                        vix_close,
                        bins=[-np.inf, 15, 25, np.inf],
                        labels=[0, 1, 2],
                    )
                    .astype(float)
                    .fillna(1.0)
                )
        except Exception:
            pass

        sector_etf = _get_sector_etf(symbol)
        if sector_etf:
            try:
                sector = yf.download(
                    sector_etf,
                    start=_dl_start,
                    end=_dl_end,
                    progress=False,
                )
                if not sector.empty:
                    if isinstance(sector.columns, pd.MultiIndex):
                        sector.columns = sector.columns.get_level_values(0)
                    sector_close = sector["Close"].reindex(df.index, method="ffill")
                    stock_ret = close.pct_change(20)
                    sector_ret = sector_close.pct_change(20)
                    features["sector_rel_strength"] = stock_ret - sector_ret
            except Exception:
                pass

        # ── Cross-sectional (1) ─────────────────────────────────────────
        # Percentile rank of 20d return vs SPY
        features["ret_20d_rank"] = 0.5
        try:
            spy = yf.download(
                "SPY",
                start=_dl_start,
                end=_dl_end,
                progress=False,
            )
            if not spy.empty:
                if isinstance(spy.columns, pd.MultiIndex):
                    spy.columns = spy.columns.get_level_values(0)
                spy_close = spy["Close"].reindex(df.index, method="ffill")
                spy_ret_20 = spy_close.pct_change(20)
                stock_ret_20 = close.pct_change(20)
                # Simple rank: 1 if stock > SPY, 0 otherwise, smoothed
                diff = stock_ret_20 - spy_ret_20
                features["ret_20d_rank"] = diff.rolling(20).rank(pct=True)
        except Exception:
            pass

        # ── Microstructure (5) ──────────────────────────────────────────
        # Volume-price divergence
        ret_5d_sign = np.sign(close.pct_change(5))
        vol_zscore_sign = np.sign(features["volume_zscore"])
        features["volume_price_divergence"] = (ret_5d_sign != vol_zscore_sign).astype(float)

        # High-low range (bid-ask liquidity proxy)
        features["high_low_range"] = ((high - low) / close).rolling(10).mean()

        # Overnight gap percentage
        features["gap_pct"] = (df["Open"] - close.shift(1)) / close.shift(1)

        # Up-volume ratio (rolling 20d)
        up_days = (close.diff() > 0).astype(float)
        up_vol = (up_days * volume).rolling(20).sum()
        total_vol = volume.rolling(20).sum()
        features["up_volume_ratio"] = up_vol / total_vol.replace(0, np.nan)

        # Consecutive direction (signed count of consecutive up/down days)
        direction = np.sign(close.diff())
        consec = direction.copy()
        for i in range(1, len(consec)):
            if direction.iloc[i] == direction.iloc[i - 1] and direction.iloc[i] != 0:
                consec.iloc[i] = consec.iloc[i - 1] + direction.iloc[i]
        features["consecutive_direction"] = consec

        # ── Distance from 52-week high (1) ─────────────────────────────
        rolling_max_252 = close.rolling(252, min_periods=60).max()
        features["dist_from_52w_high"] = (close - rolling_max_252) / rolling_max_252

        # ── Fundamental (snapshot, constant across date range) (5) ────
        features["earnings_proximity"] = 0.0
        features["analyst_upside"] = 0.0
        features["pe_vs_sector"] = 0.0
        features["recommendation_score"] = 0.5
        features["earnings_growth"] = 0.0

        try:
            ticker = yf.Ticker(symbol)
            info = ticker.info or {}

            # Days to next earnings (0-1 scale, 1 = imminent)
            try:
                cal = ticker.calendar
                if cal is not None:
                    # calendar may be dict or DataFrame
                    if isinstance(cal, dict):
                        ed = cal.get("Earnings Date")
                        if isinstance(ed, list) and ed:
                            ed = ed[0]
                    else:
                        ed = cal.iloc[0, 0] if not cal.empty else None
                    if ed is not None:
                        ed_ts = pd.Timestamp(ed)
                        days_to = max(0, (ed_ts - df.index[-1]).days)
                        # Sigmoid-like: 1.0 when 0 days, ~0 when 90+ days
                        features["earnings_proximity"] = np.exp(-days_to / 15.0)
            except Exception:
                pass

            # Analyst target upside (% above current price, capped)
            target = info.get("targetMeanPrice")
            cur_price = info.get("currentPrice") or info.get("regularMarketPrice")
            if target and cur_price and cur_price > 0:
                upside = (target - cur_price) / cur_price
                features["analyst_upside"] = np.clip(upside, -0.5, 1.0)

            # PE vs sector (discount = positive, premium = negative)
            pe = info.get("trailingPE")
            sector_pe = info.get("sectorTrailingPE") or info.get("industryTrailingPE")
            if pe and sector_pe and sector_pe > 0:
                features["pe_vs_sector"] = np.clip(
                    (sector_pe - pe) / sector_pe,
                    -1.0,
                    1.0,
                )

            # Analyst recommendation (1=strong buy .. 5=sell → scaled 0-1)
            rec = info.get("recommendationMean")
            if rec is not None:
                features["recommendation_score"] = np.clip(
                    1.0 - (rec - 1.0) / 4.0,
                    0.0,
                    1.0,
                )

            # Earnings growth estimate
            eg = info.get("earningsGrowth") or info.get("earningsQuarterlyGrowth")
            if eg is not None:
                features["earnings_growth"] = np.clip(float(eg), -1.0, 2.0)

        except Exception as e:
            logger.debug(f"Fundamental features failed for {symbol}: {e}")

        # ── Options-derived (snapshot) (4) ────────────────────────────
        features["iv_rv_ratio"] = 1.0
        features["put_call_oi_ratio"] = 1.0
        features["iv_skew"] = 0.0
        features["options_volume_ratio"] = 0.0

        try:
            ticker = yf.Ticker(symbol)
            expirations = ticker.options
            if expirations:
                cur_price = float(close.iloc[-1])
                atm_range = cur_price * 0.05

                chain = ticker.option_chain(expirations[0])
                calls = chain.calls
                puts = chain.puts

                # ATM implied volatility
                atm_calls = calls[
                    (calls["strike"] >= cur_price - atm_range)
                    & (calls["strike"] <= cur_price + atm_range)
                ]
                atm_puts = puts[
                    (puts["strike"] >= cur_price - atm_range)
                    & (puts["strike"] <= cur_price + atm_range)
                ]
                all_iv = []
                if not atm_calls.empty:
                    all_iv.extend(atm_calls["impliedVolatility"].dropna().tolist())
                if not atm_puts.empty:
                    all_iv.extend(atm_puts["impliedVolatility"].dropna().tolist())

                if all_iv:
                    iv = float(np.mean(all_iv))
                    rv = features["realized_vol_20"].iloc[-1]
                    if rv and rv > 0:
                        features["iv_rv_ratio"] = np.clip(iv / rv, 0.3, 3.0)

                # Put/call OI ratio
                call_oi = calls["openInterest"].fillna(0).sum()
                put_oi = puts["openInterest"].fillna(0).sum()
                if call_oi > 0:
                    features["put_call_oi_ratio"] = np.clip(
                        put_oi / call_oi,
                        0.1,
                        5.0,
                    )

                # IV skew (OTM put IV - OTM call IV)
                otm_puts = puts[puts["strike"] < cur_price * 0.95]
                otm_calls = calls[calls["strike"] > cur_price * 1.05]
                if not otm_puts.empty and not otm_calls.empty:
                    avg_put_iv = otm_puts["impliedVolatility"].dropna().mean()
                    avg_call_iv = otm_calls["impliedVolatility"].dropna().mean()
                    if avg_call_iv > 0:
                        features["iv_skew"] = np.clip(
                            (avg_put_iv - avg_call_iv) / avg_call_iv,
                            -1.0,
                            2.0,
                        )

                # Options volume / OI ratio (unusual activity)
                total_vol = calls["volume"].fillna(0).sum() + puts["volume"].fillna(0).sum()
                total_oi = call_oi + put_oi
                if total_oi > 0:
                    features["options_volume_ratio"] = np.clip(
                        total_vol / total_oi,
                        0.0,
                        3.0,
                    )

        except Exception as e:
            logger.debug(f"Options features failed for {symbol}: {e}")

        # ── Alpha library factors (19) ────────────────────────────────
        try:
            from advisor.ml.alpha_library import compute_all_alphas

            alphas = compute_all_alphas(df)
            for col in alphas.columns:
                features[col] = alphas[col]
        except Exception as e:
            logger.debug(f"Alpha library features failed for {symbol}: {e}")

        # ── Fractional differentiation (4) ────────────────────────────
        try:
            from advisor.ml.fracdiff import fracdiff_series

            log_close = np.log(close.replace(0, np.nan)).dropna()
            for d in [0.3, 0.4, 0.5]:
                col = f"fracdiff_d{d:.1f}".replace(".", "")
                fd = fracdiff_series(log_close, d=d, window=100)
                fd_std = fd.rolling(20).std().replace(0, np.nan)
                features[col] = fd / fd_std
        except Exception as e:
            logger.debug(f"Fracdiff features failed for {symbol}: {e}")

        # Drop warm-up rows (need at least 60 bars for all indicators)
        features = features.iloc[60:]

        return features

    @staticmethod
    def feature_names(include_snapshots: bool = True) -> list[str]:
        """Return ordered list of feature names.

        Args:
            include_snapshots: If True, include fundamental/options snapshot
                features.  Set False for training to avoid look-ahead bias.
        """
        names = [
            # Momentum (6)
            "ret_1d",
            "ret_5d",
            "ret_10d",
            "ret_20d",
            "ret_60d",
            "momentum_rank",
            # Trend (4)
            "sma_20_dist",
            "sma_50_dist",
            "ema_12_26_diff",
            "macd_signal_diff",
            # Mean-reversion (3)
            "rsi_14",
            "bb_width",
            "bb_pct",
            # Volatility (4)
            "atr_14",
            "realized_vol_20",
            "realized_vol_5",
            "vol_ratio",
            # Volume (3)
            "volume_zscore",
            "volume_trend",
            "obv_slope",
            # Cross-asset (4)
            "vix_level",
            "vix_change_5d",
            "vix_regime",
            "sector_rel_strength",
            # Cross-sectional (1)
            "ret_20d_rank",
            # Microstructure (5)
            "volume_price_divergence",
            "high_low_range",
            "gap_pct",
            "up_volume_ratio",
            "consecutive_direction",
            # Relative (1)
            "dist_from_52w_high",
            # Alpha library (19)
            "alpha_mom_12_1",
            "alpha_short_reversal",
            "alpha_acceleration",
            "alpha_ret_consistency",
            "alpha_overnight_sentiment",
            "alpha_close_location",
            "alpha_volume_surprise",
            "alpha_price_volume_corr",
            "alpha_intraday_intensity",
            "alpha_volume_price_trend",
            "alpha_vol_of_vol",
            "alpha_skewness",
            "alpha_kurtosis",
            "alpha_down_vol_ratio",
            "alpha_amihud_illiquidity",
            "alpha_high_low_momentum",
            "alpha_trend_strength",
            "alpha_max_drawdown_20d",
            "alpha_vwap_deviation",
            # Fractional differentiation (3)
            "fracdiff_d03",
            "fracdiff_d04",
            "fracdiff_d05",
        ]
        if include_snapshots:
            names.extend(
                [
                    # Fundamental (5)
                    "earnings_proximity",
                    "analyst_upside",
                    "pe_vs_sector",
                    "recommendation_score",
                    "earnings_growth",
                    # Options-derived (4)
                    "iv_rv_ratio",
                    "put_call_oi_ratio",
                    "iv_skew",
                    "options_volume_ratio",
                ]
            )
        return names
