"""The macro factor panel.

Nine factors, all free from yfinance, chosen so each answers a question you
would actually ask about a book: how much of this is just the market, who
breaks when yields rise, who suffers on a strong dollar, what happens if risk
appetite turns.

Spread factors (``CREDIT``, ``GROWTH_VALUE``, ``SIZE``, ``BREADTH``) are built
as the return *difference* between two legs. That matters: raw HYG returns are
mostly duration and market beta, so a naive regression would attribute credit
sensitivity to almost everything. HYG minus LQD isolates the risk-appetite
component the factor is supposed to measure.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class Factor(StrEnum):
    MKT = "MKT"  # broad equity beta
    DURATION = "DURATION"  # long bonds: + means benefits from falling yields
    DOLLAR = "DOLLAR"  # + means benefits from a stronger dollar
    ENERGY = "ENERGY"  # energy complex / input costs
    CREDIT = "CREDIT"  # risk appetite (high yield vs investment grade)
    GROWTH_VALUE = "GROWTH_VALUE"  # + means growth-leaning
    SIZE = "SIZE"  # + means small-cap-leaning
    VOL = "VOL"  # + means gains when volatility spikes
    BREADTH = "BREADTH"  # + means benefits from broadening markets


@dataclass(frozen=True)
class FactorSpec:
    """How one factor's return series is constructed."""

    long: str
    short: str | None = None  # when set, the factor is the long-minus-short spread
    label: str = ""

    @property
    def tickers(self) -> tuple[str, ...]:
        return (self.long,) if self.short is None else (self.long, self.short)


FACTOR_SPECS: dict[Factor, FactorSpec] = {
    Factor.MKT: FactorSpec("SPY", label="broad equity market"),
    Factor.DURATION: FactorSpec("TLT", label="long-duration Treasuries"),
    Factor.DOLLAR: FactorSpec("UUP", label="US dollar index"),
    Factor.ENERGY: FactorSpec("XLE", label="energy sector"),
    Factor.CREDIT: FactorSpec("HYG", "LQD", label="high yield vs investment grade"),
    Factor.GROWTH_VALUE: FactorSpec("IWF", "IWD", label="growth vs value"),
    Factor.SIZE: FactorSpec("IWM", "SPY", label="small vs large cap"),
    Factor.VOL: FactorSpec("^VIX", label="implied volatility"),
    Factor.BREADTH: FactorSpec("RSP", "SPY", label="equal vs cap weighted"),
}

# Minimum overlapping observations before a regression is worth running.
MIN_OBSERVATIONS = 120


def all_factor_tickers() -> list[str]:
    """Every distinct ticker the panel needs."""
    out: set[str] = set()
    for spec in FACTOR_SPECS.values():
        out.update(spec.tickers)
    return sorted(out)


def fetch_prices(tickers: list[str], *, period: str = "2y") -> pd.DataFrame:
    """Adjusted closes for ``tickers``, one column each.

    Returns an empty frame rather than raising when the download fails — a
    macro refresh that cannot reach the network must degrade, not take down
    the job that called it.
    """
    import yfinance as yf

    unique = sorted({t for t in tickers if t})
    if not unique:
        return pd.DataFrame()
    try:
        raw = yf.download(unique, period=period, progress=False, auto_adjust=True, threads=True)
    except Exception as exc:  # noqa: BLE001
        logger.warning("factor price download failed: %s", exc)
        return pd.DataFrame()

    if raw is None or raw.empty:
        return pd.DataFrame()

    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    if len(unique) == 1:
        close.columns = unique
    return close.dropna(how="all")


def log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Daily log returns, with non-finite values dropped.

    Log returns rather than simple returns so that spread factors are a clean
    subtraction of two return series.
    """
    if prices.empty:
        return prices
    rets = np.log(prices / prices.shift(1))
    return rets.replace([np.inf, -np.inf], np.nan).dropna(how="all")


def build_factor_returns(prices: pd.DataFrame | None = None, *, period: str = "2y"):
    """Return a DataFrame of daily factor returns, one column per Factor.

    Any factor whose legs are missing from ``prices`` is skipped rather than
    filled with zeros — a silently zeroed factor would read as "no exposure"
    instead of "unknown".
    """
    if prices is None:
        prices = fetch_prices(all_factor_tickers(), period=period)
    if prices.empty:
        return pd.DataFrame()

    rets = log_returns(prices)
    columns: dict[str, pd.Series] = {}
    for factor, spec in FACTOR_SPECS.items():
        if spec.long not in rets.columns:
            logger.debug("factor %s skipped: %s missing", factor, spec.long)
            continue
        series = rets[spec.long]
        if spec.short is not None:
            if spec.short not in rets.columns:
                logger.debug("factor %s skipped: %s missing", factor, spec.short)
                continue
            series = series - rets[spec.short]
        columns[factor.value] = series

    if not columns:
        return pd.DataFrame()
    return pd.DataFrame(columns).dropna(how="all")
