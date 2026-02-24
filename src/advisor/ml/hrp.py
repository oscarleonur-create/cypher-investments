"""Hierarchical Risk Parity (HRP) portfolio construction for position sizing.

Implements the HRP algorithm from Marcos Lopez de Prado:
  1. Compute correlation/covariance matrices from asset returns
  2. Cluster assets via hierarchical clustering on correlation-distance
  3. Quasi-diagonalize by reordering along dendrogram leaves
  4. Recursive bisection — allocate by inverse variance at cluster boundaries

References:
  Lopez de Prado, M. (2016). "Building Diversified Portfolios that Outperform
  Out-of-Sample." *Journal of Portfolio Management*, 42(4), 59-69.
"""

from __future__ import annotations

import logging
from typing import Sequence

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.cluster.hierarchy import leaves_list, linkage
from scipy.spatial.distance import squareform

logger = logging.getLogger(__name__)

_MIN_WEIGHT = 0.01  # 1 % floor per position


class HRPAllocator:
    """Hierarchical Risk Parity allocator.

    Usage::

        allocator = HRPAllocator()

        # From a pre-built returns DataFrame
        weights = allocator.compute_weights(returns_df)

        # From confluence/ML signals (downloads data automatically)
        signals = [("AAPL", 0.9), ("MSFT", 0.7), ("GOOG", 0.5)]
        weights = allocator.compute_weights_from_signals(signals)
    """

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def compute_weights(self, returns: pd.DataFrame) -> dict[str, float]:
        """Compute HRP weights from a daily-returns DataFrame.

        Args:
            returns: DataFrame where columns are symbols and rows are dates.
                     Values should be simple daily returns (e.g. 0.01 for +1 %).

        Returns:
            Dict mapping each symbol to its portfolio weight (sums to 1.0).
        """
        returns = self._clean_returns(returns)
        symbols = list(returns.columns)

        if len(symbols) == 0:
            logger.warning("Empty returns DataFrame — returning empty weights")
            return {}

        if len(symbols) == 1:
            logger.info("Single asset — assigning weight 1.0")
            return {symbols[0]: 1.0}

        # Step 1: correlation and covariance
        corr = returns.corr()
        cov = returns.cov()

        # Step 2: hierarchical clustering
        dist = self._correlation_distance(corr)
        link = linkage(squareform(dist), method="single")

        # Step 3: quasi-diagonalization (reorder by dendrogram leaves)
        sort_ix = leaves_list(link).tolist()
        sorted_symbols = [symbols[i] for i in sort_ix]

        # Step 4: recursive bisection
        weights = self._recursive_bisection(cov, sorted_symbols)

        # Enforce minimum weight floor
        weights = self._apply_weight_floor(weights)

        return weights

    def compute_weights_from_signals(
        self,
        signals: Sequence[tuple[str, float]],
        lookback_days: int = 252,
    ) -> dict[str, float]:
        """Compute HRP weights tilted by conviction scores.

        Downloads historical returns via yfinance, computes base HRP weights,
        then multiplies each weight by its conviction score and renormalises.

        Args:
            signals: List of (symbol, conviction_score) tuples.  Conviction
                     scores are positive floats (higher = stronger view).
            lookback_days: Calendar days of history to download for the
                          covariance estimate.

        Returns:
            Dict mapping each symbol to its conviction-tilted weight.
        """
        if not signals:
            logger.warning("No signals provided — returning empty weights")
            return {}

        symbols = [s.upper() for s, _ in signals]
        convictions = {s.upper(): max(c, 0.0) for s, c in signals}

        if len(symbols) == 1:
            return {symbols[0]: 1.0}

        # Download historical prices
        returns = self._download_returns(symbols, lookback_days)

        if returns.empty:
            logger.error("Could not download returns — falling back to equal weight")
            n = len(symbols)
            return {s: 1.0 / n for s in symbols}

        # Drop symbols that had no data
        available = [s for s in symbols if s in returns.columns]
        if not available:
            logger.error("No valid return series after download")
            return {}

        returns = returns[available]

        # Base HRP weights
        base_weights = self.compute_weights(returns)

        # Tilt by conviction
        tilted: dict[str, float] = {}
        for sym, w in base_weights.items():
            tilted[sym] = w * convictions.get(sym, 1.0)

        total = sum(tilted.values())
        if total <= 0:
            logger.warning("All tilted weights are zero — falling back to base HRP")
            return base_weights

        tilted = {s: w / total for s, w in tilted.items()}

        # Re-apply floor after tilting
        tilted = self._apply_weight_floor(tilted)

        return tilted

    # ------------------------------------------------------------------
    # HRP internals
    # ------------------------------------------------------------------

    @staticmethod
    def _correlation_distance(corr: pd.DataFrame) -> pd.DataFrame:
        """Convert a correlation matrix to a proper distance metric.

        distance_ij = sqrt(0.5 * (1 - corr_ij))

        This satisfies the triangle inequality and maps corr=1 to dist=0,
        corr=-1 to dist=1.
        """
        dist = ((1.0 - corr) / 2.0) ** 0.5
        # Ensure exact zeros on diagonal (floating-point hygiene)
        np.fill_diagonal(dist.values, 0.0)
        return dist

    @staticmethod
    def _recursive_bisection(
        cov: pd.DataFrame,
        sorted_symbols: list[str],
    ) -> dict[str, float]:
        """Allocate weights via top-down recursive bisection.

        At each split the total allocation for the cluster is divided between
        the left and right sub-clusters in proportion to inverse variance
        (so the lower-variance half gets more weight).
        """
        weights = pd.Series(1.0, index=sorted_symbols)
        clusters: list[list[str]] = [sorted_symbols]

        while clusters:
            next_clusters: list[list[str]] = []
            for cluster in clusters:
                if len(cluster) <= 1:
                    continue

                mid = len(cluster) // 2
                left = cluster[:mid]
                right = cluster[mid:]

                var_left = _cluster_variance(cov, left)
                var_right = _cluster_variance(cov, right)

                # Inverse-variance split factor
                denom = var_left + var_right
                if denom < 1e-16:
                    alpha = 0.5
                else:
                    alpha = 1.0 - var_left / denom  # left gets *less* if its var is higher

                weights[left] *= alpha
                weights[right] *= 1.0 - alpha

                if len(left) > 1:
                    next_clusters.append(left)
                if len(right) > 1:
                    next_clusters.append(right)

            clusters = next_clusters

        return weights.to_dict()

    # ------------------------------------------------------------------
    # Data helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _clean_returns(returns: pd.DataFrame) -> pd.DataFrame:
        """Forward-fill NaNs, then drop any remaining rows with NaN."""
        returns = returns.ffill()
        before = len(returns)
        returns = returns.dropna()
        dropped = before - len(returns)
        if dropped > 0:
            logger.info(f"Dropped {dropped} rows with NaN after forward-fill")
        return returns

    @staticmethod
    def _download_returns(
        symbols: list[str],
        lookback_days: int,
    ) -> pd.DataFrame:
        """Download daily close prices and compute simple returns.

        Uses yfinance batch download for efficiency.
        """
        period = f"{lookback_days}d"
        logger.info(f"Downloading {len(symbols)} symbols, period={period}")

        try:
            prices = yf.download(
                symbols,
                period=period,
                progress=False,
                auto_adjust=True,
            )
        except Exception as exc:
            logger.error(f"yfinance download failed: {exc}")
            return pd.DataFrame()

        if prices.empty:
            return pd.DataFrame()

        # yf.download returns MultiIndex columns when len(symbols) > 1
        if isinstance(prices.columns, pd.MultiIndex):
            close = prices["Close"]
        else:
            # Single symbol — wrap in DataFrame with symbol as column name
            close = prices[["Close"]].rename(columns={"Close": symbols[0]})

        # Flatten any remaining MultiIndex (e.g. ("Close", "AAPL") -> "AAPL")
        if isinstance(close.columns, pd.MultiIndex):
            close.columns = close.columns.get_level_values(-1)

        returns = close.pct_change().iloc[1:]  # drop first NaN row
        return returns

    @staticmethod
    def _apply_weight_floor(weights: dict[str, float]) -> dict[str, float]:
        """Enforce a minimum weight of *_MIN_WEIGHT* per position.

        Positions below the floor are bumped up, and the excess mass is
        removed proportionally from positions above the floor.
        """
        n = len(weights)
        if n == 0:
            return weights

        # If floor * n >= 1.0, equal-weight is the only feasible solution
        if _MIN_WEIGHT * n >= 1.0:
            equal = 1.0 / n
            return {s: equal for s in weights}

        result = dict(weights)
        for _ in range(10):  # iterate to convergence (usually 1-2 rounds)
            deficit = 0.0
            above_floor: list[str] = []

            for sym, w in result.items():
                if w < _MIN_WEIGHT:
                    deficit += _MIN_WEIGHT - w
                    result[sym] = _MIN_WEIGHT
                else:
                    above_floor.append(sym)

            if deficit <= 0:
                break

            above_total = sum(result[s] for s in above_floor)
            if above_total <= 0:
                break

            for sym in above_floor:
                result[sym] -= deficit * (result[sym] / above_total)

        # Final normalisation to guarantee sum == 1.0
        total = sum(result.values())
        if total > 0:
            result = {s: w / total for s, w in result.items()}

        return result


# ------------------------------------------------------------------
# Module-level helpers
# ------------------------------------------------------------------


def _cluster_variance(cov: pd.DataFrame, symbols: list[str]) -> float:
    """Compute the variance of an inverse-variance-weighted sub-portfolio.

    For a cluster of assets, the IVP (inverse-variance portfolio) weights
    are w_i = (1/var_i) / sum(1/var_j).  The cluster variance is then
    w^T * Cov * w.
    """
    cov_slice = cov.loc[symbols, symbols]
    ivp = 1.0 / np.diag(cov_slice.values)
    ivp = ivp / ivp.sum()
    return float(ivp @ cov_slice.values @ ivp)
