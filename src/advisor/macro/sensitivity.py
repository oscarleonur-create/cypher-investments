"""Per-symbol factor sensitivities.

Answers "how does this position actually behave", as opposed to how its thesis
says it behaves. A rolling ridge regression of a symbol's daily returns on the
factor panel produces loadings that make the difference between commentary
("CPI ran hot") and advice ("your book's duration loading is the largest
unhedged bet you own, and 68% of it sits in three names").

**Ridge, not OLS.** The factors are deliberately correlated — growth/value,
size and breadth all overlap with the market. OLS on collinear regressors
produces enormous, unstable, sign-flipping coefficients. Ridge trades a little
bias for coefficients that stay put week to week, which is what a monitoring
system needs.

The reported t-statistics are computed the OLS way on the ridge fit. Ridge
estimates are biased, so these are **approximate** and are used only as a
materiality filter (is this loading worth mentioning at all), never as a formal
significance test.
"""

from __future__ import annotations

import logging
from datetime import date

import numpy as np
import pandas as pd
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Shrinkage as a fraction of the sample size, applied to standardized
# regressors so that a factor is not penalised for being noisy (VIX moves ~8x
# as much as SPY per day).
RIDGE_ALPHA_FRACTION = 0.02

DEFAULT_WINDOW_DAYS = 250
MIN_OBSERVATIONS = 120

# A loading must clear both bars to count as real exposure.
MATERIAL_LOADING = 0.30
MATERIAL_TSTAT = 2.0


class FactorLoading(BaseModel):
    """One factor's estimated effect on a symbol."""

    factor: str
    loading: float  # return per unit of factor return
    tstat: float  # approximate; see module docstring

    @property
    def is_material(self) -> bool:
        return abs(self.loading) >= MATERIAL_LOADING and abs(self.tstat) >= MATERIAL_TSTAT


class SymbolSensitivity(BaseModel):
    """A symbol's factor profile over one estimation window."""

    symbol: str
    asof: date
    window_days: int
    n_obs: int
    r2: float
    resid_vol: float  # daily residual stdev — the yardstick for divergence
    sector_etf: str | None = None
    loadings: list[FactorLoading] = Field(default_factory=list)

    def loading(self, factor: str) -> float:
        for entry in self.loadings:
            if entry.factor == factor:
                return entry.loading
        return 0.0

    def material(self) -> list[FactorLoading]:
        return [entry for entry in self.loadings if entry.is_material]

    def as_dict(self) -> dict[str, float]:
        return {entry.factor: entry.loading for entry in self.loadings}


def ridge_fit(
    y: np.ndarray, x: np.ndarray, *, alpha_fraction: float = RIDGE_ALPHA_FRACTION
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """Ridge regression on standardized regressors.

    Returns ``(betas, tstats, r2, resid_vol)`` with betas expressed in the
    original units of ``x`` (return per unit factor return), not standardized
    units.
    """
    n, k = x.shape
    x_mean, x_std = x.mean(axis=0), x.std(axis=0, ddof=1)
    # A factor with no variation carries no information; neutralise it rather
    # than dividing by zero.
    x_std = np.where(x_std > 0, x_std, 1.0)
    xs = (x - x_mean) / x_std
    y_mean = y.mean()
    yc = y - y_mean

    alpha = alpha_fraction * n
    gram = xs.T @ xs + alpha * np.eye(k)
    inv = np.linalg.pinv(gram)
    beta_std = inv @ xs.T @ yc

    fitted = xs @ beta_std
    resid = yc - fitted
    dof = max(n - k - 1, 1)
    sigma2 = float(resid @ resid) / dof

    # OLS-form standard errors on the ridge fit — approximate by construction.
    cov = sigma2 * (inv @ (xs.T @ xs) @ inv)
    se = np.sqrt(np.maximum(np.diag(cov), 1e-18))
    tstats = beta_std / se

    ss_tot = float(yc @ yc)
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else 0.0
    resid_vol = float(np.std(resid, ddof=1)) if n > 1 else 0.0

    return beta_std / x_std, tstats, r2, resid_vol


def estimate_sensitivity(
    symbol: str,
    symbol_returns: pd.Series,
    factor_returns: pd.DataFrame,
    *,
    window_days: int = DEFAULT_WINDOW_DAYS,
    sector_etf: str | None = None,
    asof: date | None = None,
) -> SymbolSensitivity | None:
    """Estimate ``symbol``'s factor loadings, or None if there is too little data.

    A newly listed name with 40 days of history cannot support a nine-factor
    regression; returning None is the honest answer and keeps a meaningless
    loading out of the book aggregate.
    """
    aligned = pd.concat([symbol_returns.rename("_y"), factor_returns], axis=1).dropna()
    if len(aligned) > window_days:
        aligned = aligned.iloc[-window_days:]

    if len(aligned) < MIN_OBSERVATIONS:
        logger.debug(
            "sensitivity: %s has %d usable observations, need %d",
            symbol,
            len(aligned),
            MIN_OBSERVATIONS,
        )
        return None

    y = aligned["_y"].to_numpy(dtype=float)
    factor_names = [c for c in aligned.columns if c != "_y"]
    x = aligned[factor_names].to_numpy(dtype=float)

    betas, tstats, r2, resid_vol = ridge_fit(y, x)

    return SymbolSensitivity(
        symbol=symbol.upper(),
        asof=asof or aligned.index[-1].date(),
        window_days=window_days,
        n_obs=len(aligned),
        r2=round(r2, 4),
        resid_vol=round(resid_vol, 6),
        sector_etf=sector_etf,
        loadings=[
            FactorLoading(factor=name, loading=round(float(b), 4), tstat=round(float(t), 2))
            for name, b, t in zip(factor_names, betas, tstats, strict=True)
        ],
    )


def expected_return(sensitivity: SymbolSensitivity, factor_moves: dict[str, float]) -> float:
    """Predicted return given a set of factor moves.

    This is what makes a macro event concrete: it converts "CPI ran hot and
    duration sold off 1.5%" into an expected effect on a specific position.
    """
    return sum(sensitivity.loading(f) * move for f, move in factor_moves.items())


def residual_z(
    sensitivity: SymbolSensitivity, actual_return: float, factor_moves: dict[str, float]
) -> float:
    """How many residual standard deviations the actual move was off model.

    A large residual means the symbol moved for a reason the macro panel cannot
    explain — often the earliest visible sign of something company-specific.
    """
    if sensitivity.resid_vol <= 0:
        return 0.0
    return (actual_return - expected_return(sensitivity, factor_moves)) / sensitivity.resid_vol
