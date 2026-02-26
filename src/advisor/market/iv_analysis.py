"""IV analysis — percentile, term structure, expected move, earnings proximity."""

from __future__ import annotations

import logging
import math
from datetime import date, datetime

import numpy as np
import yfinance as yf
from pydantic import BaseModel

logger = logging.getLogger(__name__)


# ── Models ────────────────────────────────────────────────────────────────────


class IVPercentileResult(BaseModel):
    symbol: str
    iv_percentile: float  # 0-100
    current_iv: float  # annualized ATM IV
    hv30_current: float = 0.0  # current 30-day HV
    iv_rank: float | None = None  # from TastyTrade if available
    source: str = "estimated"  # "estimated" or "tastytrade"


class TermStructureResult(BaseModel):
    symbol: str
    classification: str  # "contango", "backwardation", "flat"
    slope: float = 0.0  # positive = contango
    term_ivs: list[tuple[int, float]] = []  # [(dte, iv), ...]


# ── ATM IV extraction ─────────────────────────────────────────────────────────


def _extract_atm_iv(ticker: yf.Ticker, price: float, expiry: str) -> float | None:
    """Extract average ATM implied volatility from a single expiration."""
    try:
        chain = ticker.option_chain(expiry)
    except Exception:
        return None

    atm_range = price * 0.05
    all_iv = []

    for df in (chain.puts, chain.calls):
        if df is not None and not df.empty:
            atm = df[(df["strike"] >= price - atm_range) & (df["strike"] <= price + atm_range)]
            ivs = atm["impliedVolatility"].dropna().tolist()
            all_iv.extend(iv for iv in ivs if iv > 0)

    return float(np.mean(all_iv)) if all_iv else None


def _get_price(ticker: yf.Ticker) -> float | None:
    """Get current price from ticker."""
    info = ticker.info or {}
    price = info.get("currentPrice") or info.get("regularMarketPrice")
    if price:
        return float(price)
    hist = ticker.history(period="1d")
    if not hist.empty:
        return float(hist["Close"].iloc[-1])
    return None


# ── IV Percentile ─────────────────────────────────────────────────────────────


def compute_iv_percentile(
    symbol: str,
    tt_data: dict | None = None,
    ticker: yf.Ticker | None = None,
) -> IVPercentileResult:
    """Compute 52-week IV percentile for a symbol.

    If tt_data is provided (from TastyTrade --live-iv), uses the precise
    iv_percentile value. Otherwise approximates by ranking current ATM IV
    against a 1-year distribution of 30-day historical volatility.
    """
    symbol = symbol.upper()

    # Fast path: TastyTrade live data
    if tt_data and symbol in tt_data:
        tt = tt_data[symbol]
        return IVPercentileResult(
            symbol=symbol,
            iv_percentile=tt.get("iv_percentile", 0) * 100,
            current_iv=tt.get("iv_index", 0),
            iv_rank=tt.get("iv_rank"),
            source="tastytrade",
        )

    # Estimated path: ATM IV vs 1yr HV30 distribution
    if ticker is None:
        ticker = yf.Ticker(symbol)

    price = _get_price(ticker)
    if price is None or price <= 0:
        return IVPercentileResult(symbol=symbol, iv_percentile=50.0, current_iv=0.0)

    # Current ATM IV from nearest expiration
    expirations = ticker.options
    if not expirations:
        return IVPercentileResult(symbol=symbol, iv_percentile=50.0, current_iv=0.0)

    current_iv = _extract_atm_iv(ticker, price, expirations[0])
    if current_iv is None or current_iv <= 0:
        return IVPercentileResult(symbol=symbol, iv_percentile=50.0, current_iv=0.0)

    # 1-year daily history for rolling 30-day HV
    hist = ticker.history(period="1y")
    if hist.empty or len(hist) < 60:
        return IVPercentileResult(symbol=symbol, iv_percentile=50.0, current_iv=current_iv)

    close = hist["Close"]
    daily_returns = close.pct_change().dropna()
    hv30_series = daily_returns.rolling(30).std() * np.sqrt(252)
    hv30_series = hv30_series.dropna()

    if hv30_series.empty:
        return IVPercentileResult(symbol=symbol, iv_percentile=50.0, current_iv=current_iv)

    hv30_current = float(hv30_series.iloc[-1])

    # IV percentile = % of HV30 values below current ATM IV
    pctile = float(np.mean(hv30_series.values < current_iv) * 100)

    return IVPercentileResult(
        symbol=symbol,
        iv_percentile=round(pctile, 1),
        current_iv=round(current_iv, 4),
        hv30_current=round(hv30_current, 4),
        source="estimated",
    )


# ── Term Structure ────────────────────────────────────────────────────────────


def classify_term_structure(
    symbol: str,
    ticker: yf.Ticker | None = None,
    price: float | None = None,
) -> TermStructureResult:
    """Sample ATM IV across 3-5 expirations and classify the term structure.

    Contango: IV rises with DTE (normal, favorable for selling near-term).
    Backwardation: IV falls with DTE (event risk, cautious).
    Flat: No meaningful slope.
    """
    symbol = symbol.upper()
    if ticker is None:
        ticker = yf.Ticker(symbol)
    if price is None:
        price = _get_price(ticker)
    if price is None or price <= 0:
        return TermStructureResult(symbol=symbol, classification="flat")

    expirations = ticker.options
    if not expirations or len(expirations) < 2:
        return TermStructureResult(symbol=symbol, classification="flat")

    today = date.today()
    term_ivs: list[tuple[int, float]] = []

    # Sample up to 5 expirations spread across the term
    sample_expiries = expirations[:5]
    for exp_str in sample_expiries:
        try:
            exp_date = date.fromisoformat(exp_str)
            dte = (exp_date - today).days
            if dte <= 0:
                continue
            iv = _extract_atm_iv(ticker, price, exp_str)
            if iv is not None and iv > 0:
                term_ivs.append((dte, iv))
        except Exception:
            continue

    if len(term_ivs) < 2:
        return TermStructureResult(symbol=symbol, classification="flat", term_ivs=term_ivs)

    # Linear regression slope: IV vs DTE
    dtes = np.array([t[0] for t in term_ivs], dtype=float)
    ivs = np.array([t[1] for t in term_ivs], dtype=float)

    # Normalize DTE to [0,1] range to get meaningful slope magnitude
    dte_range = dtes.max() - dtes.min()
    if dte_range == 0:
        return TermStructureResult(symbol=symbol, classification="flat", term_ivs=term_ivs)

    dtes_norm = (dtes - dtes.min()) / dte_range
    slope = float(np.polyfit(dtes_norm, ivs, 1)[0])

    # Classify: slope relative to mean IV level
    mean_iv = float(ivs.mean())
    if mean_iv <= 0:
        classification = "flat"
    else:
        relative_slope = slope / mean_iv
        if relative_slope > 0.05:
            classification = "contango"
        elif relative_slope < -0.05:
            classification = "backwardation"
        else:
            classification = "flat"

    return TermStructureResult(
        symbol=symbol,
        classification=classification,
        slope=round(slope, 4),
        term_ivs=term_ivs,
    )


# ── Expected Move ─────────────────────────────────────────────────────────────


def compute_expected_move(price: float, iv: float, dte: int) -> float:
    """Compute the 1-standard-deviation expected move for a given DTE.

    Formula: price * IV * sqrt(DTE / 365)
    """
    if price <= 0 or iv <= 0 or dte <= 0:
        return 0.0
    return price * iv * math.sqrt(dte / 365)


# ── Earnings Date ─────────────────────────────────────────────────────────────


def get_next_earnings_date(symbol: str, ticker: yf.Ticker | None = None) -> date | None:
    """Get next earnings date from yfinance calendar.

    Returns None if no upcoming earnings date is found.
    """
    if ticker is None:
        ticker = yf.Ticker(symbol.upper())

    try:
        cal = ticker.calendar
        if cal is None:
            return None

        # yfinance returns calendar as a dict with "Earnings Date" key
        # containing a list of datetime objects
        if isinstance(cal, dict):
            earnings_dates = cal.get("Earnings Date", [])
            if earnings_dates:
                today = date.today()
                for dt in earnings_dates:
                    if isinstance(dt, datetime):
                        d = dt.date()
                    elif isinstance(dt, date):
                        d = dt
                    else:
                        continue
                    if d >= today:
                        return d
        # Fallback: DataFrame format (older yfinance versions)
        elif hasattr(cal, "columns"):
            if "Earnings Date" in cal.columns:
                vals = cal["Earnings Date"].dropna().tolist()
                today = date.today()
                for v in vals:
                    d = v.date() if hasattr(v, "date") else v
                    if d >= today:
                        return d
    except Exception as e:
        logger.debug(f"Failed to get earnings date for {symbol}: {e}")

    return None
