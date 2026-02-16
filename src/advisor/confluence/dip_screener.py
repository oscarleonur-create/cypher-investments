"""3-layer fundamental screener for buy-the-dip strategy.

Layers:
  1. Safety Gate — reject bankrupt / distressed companies
  2. Value Trap Detector — confirm the dip is a real deal
  3. Fast Fundamentals — timing confirmation (insiders + analysts)
"""

from __future__ import annotations

import logging
from datetime import date, timedelta

import yfinance as yf

from advisor.confluence.models import (
    DipScreenerResult,
    FastFundamentalsResult,
    FundamentalResult,
    SafetyCheckResult,
    ValueTrapResult,
)

logger = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────
CURRENT_RATIO_MIN = 1.5
DEBT_TO_EQUITY_MAX = 2.0
PE_DISCOUNT_THRESHOLD = 0.20  # 20% below 5-year avg
PRICE_DROP_THRESHOLD = -0.10  # 10% drop
ANALYST_UPSIDE_MIN = 0.15  # 15% upside
ANALYST_COUNT_MIN = 3

C_SUITE_TITLES = {"ceo", "cfo", "coo", "cto", "director", "president", "chairman"}


def _check_safety(ticker: yf.Ticker) -> SafetyCheckResult:
    """Layer 1: Safety gate — all checks must pass."""
    info = ticker.info

    # Current ratio
    current_ratio = info.get("currentRatio")
    cr_ok = current_ratio is not None and current_ratio > CURRENT_RATIO_MIN

    # Debt-to-equity (yfinance returns as %, e.g. 150 = 150%)
    raw_de = info.get("debtToEquity")
    debt_to_equity = raw_de / 100.0 if raw_de is not None else None
    de_ok = debt_to_equity is not None and debt_to_equity < DEBT_TO_EQUITY_MAX

    # Free cash flow — last 4 quarters
    fcf_values: list[float] = []
    fcf_ok = False
    try:
        qcf = ticker.quarterly_cashflow
        if qcf is not None and not qcf.empty:
            for label in ("Free Cash Flow", "FreeCashFlow"):
                if label in qcf.index:
                    vals = qcf.loc[label].dropna().head(4).tolist()
                    fcf_values = [float(v) for v in vals]
                    break
            fcf_ok = len(fcf_values) >= 4 and all(v > 0 for v in fcf_values)
    except Exception as e:
        logger.warning(f"Could not fetch quarterly cash flow: {e}")

    passes = cr_ok and de_ok and fcf_ok

    return SafetyCheckResult(
        current_ratio=current_ratio,
        current_ratio_ok=cr_ok,
        debt_to_equity=debt_to_equity,
        debt_to_equity_ok=de_ok,
        fcf_values=fcf_values,
        fcf_ok=fcf_ok,
        passes=passes,
    )


def _check_value_trap(ticker: yf.Ticker) -> ValueTrapResult:
    """Layer 2: Value trap detector — either P/E discount or RSI divergence."""
    info = ticker.info
    result = ValueTrapResult()

    # P/E vs 5-year average
    trailing_pe = info.get("trailingPE")
    trailing_eps = info.get("trailingEps")
    forward_eps = info.get("forwardEps")
    result.trailing_eps = trailing_eps
    result.forward_eps = forward_eps
    result.current_pe = trailing_pe

    if trailing_eps and trailing_eps > 0:
        try:
            hist = ticker.history(period="5y", interval="1mo")
            if hist is not None and not hist.empty and len(hist) >= 12:
                avg_price = hist["Close"].mean()
                five_yr_avg_pe = avg_price / trailing_eps
                result.five_year_avg_pe = five_yr_avg_pe

                if trailing_pe is not None and five_yr_avg_pe > 0:
                    discount = (five_yr_avg_pe - trailing_pe) / five_yr_avg_pe
                    result.pe_discount_pct = round(discount * 100, 1)
                    result.pe_on_sale = discount >= PE_DISCOUNT_THRESHOLD
        except Exception as e:
            logger.warning(f"Could not compute 5yr avg P/E: {e}")

    # RSI divergence: price dropped 10%+ but EPS stable/growing
    try:
        recent = ticker.history(period="3mo")
        if recent is not None and not recent.empty and len(recent) >= 10:
            high_price = recent["Close"].max()
            current_price = recent["Close"].iloc[-1]
            if high_price > 0:
                pct_change = (current_price - high_price) / high_price
                result.price_change_pct = round(pct_change * 100, 1)

                if (
                    pct_change <= PRICE_DROP_THRESHOLD
                    and forward_eps is not None
                    and trailing_eps is not None
                    and trailing_eps > 0
                    and forward_eps >= trailing_eps
                ):
                    result.rsi_divergence = True
    except Exception as e:
        logger.warning(f"Could not compute price change: {e}")

    result.is_value = result.pe_on_sale or result.rsi_divergence
    return result


def _check_fast_fundamentals(ticker: yf.Ticker) -> FastFundamentalsResult:
    """Layer 3: Timing confirmation — insider buying + analyst targets."""
    info = ticker.info
    result = FastFundamentalsResult()

    # Insider transactions
    try:
        transactions = ticker.insider_transactions
        if transactions is not None and not transactions.empty:
            # Detect purchases
            if "Transaction" in transactions.columns:
                buys = transactions[
                    transactions["Transaction"].str.contains("Purchase|Buy", case=False, na=False)
                ]
            elif "Shares" in transactions.columns:
                buys = transactions[transactions["Shares"] > 0]
            else:
                buys = transactions.iloc[0:0]  # empty

            if len(buys) > 0:
                result.insider_buying = True

                # Check for C-suite buying
                title_col = None
                for col in ("Title", "Position", "Insider Trading"):
                    if col in buys.columns:
                        title_col = col
                        break

                name_col = None
                for col in ("Insider", "Name", "Insider Trading"):
                    if col in buys.columns:
                        name_col = col
                        break

                for _, row in buys.head(5).iterrows():
                    detail: dict = {}
                    if name_col:
                        detail["name"] = str(row.get(name_col, ""))
                    if title_col:
                        detail["title"] = str(row.get(title_col, ""))
                    if "Shares" in buys.columns:
                        detail["shares"] = int(row.get("Shares", 0))
                    result.insider_details.append(detail)

                    # C-suite detection
                    text = " ".join(
                        str(row.get(c, "")).lower() for c in [title_col, name_col] if c is not None
                    )
                    if any(t in text for t in C_SUITE_TITLES):
                        result.c_suite_buying = True
    except Exception as e:
        logger.warning(f"Could not fetch insider transactions: {e}")

    # Analyst targets
    target_price = info.get("targetMeanPrice")
    current_price = info.get("currentPrice") or info.get("regularMarketPrice")
    n_analysts = info.get("numberOfAnalystOpinions", 0) or 0

    result.analyst_target_price = target_price
    result.n_analysts = n_analysts

    if target_price and current_price and current_price > 0:
        upside = (target_price - current_price) / current_price
        result.analyst_upside_pct = round(upside * 100, 1)
        result.analyst_bullish = upside >= ANALYST_UPSIDE_MIN and n_analysts >= ANALYST_COUNT_MIN

    result.has_confirmation = result.insider_buying or result.analyst_bullish
    return result


def _compute_score(
    safety: SafetyCheckResult,
    value_trap: ValueTrapResult | None,
    fast_fundamentals: FastFundamentalsResult | None,
) -> str:
    """Compute overall dip score from the 3 layers."""
    if not safety.passes:
        return "FAIL"

    has_value = value_trap is not None and value_trap.is_value
    has_timing = fast_fundamentals is not None and fast_fundamentals.has_confirmation
    has_rsi_div = value_trap is not None and value_trap.rsi_divergence
    has_csuite = fast_fundamentals is not None and fast_fundamentals.c_suite_buying

    if has_rsi_div and has_csuite:
        return "STRONG_BUY"
    if has_value and has_timing:
        return "BUY"
    if has_value:
        return "LEAN_BUY"
    if has_timing:
        return "WATCH"
    return "WEAK"


def check_dip_fundamental(symbol: str) -> FundamentalResult:
    """Run the 3-layer dip screener and return a FundamentalResult.

    The returned FundamentalResult is backward-compatible with the orchestrator:
    - is_clear reflects whether the safety gate passed
    - insider_buying_detected reflects Layer 3 insider signals
    - dip_screener holds the full 3-layer breakdown
    """
    ticker = yf.Ticker(symbol)

    # --- Standard earnings check (reuse logic from fundamental.py) ---
    today = date.today()
    earnings_within_7 = False
    earnings_dt: date | None = None

    try:
        calendar = ticker.calendar
        if calendar is not None:
            if isinstance(calendar, dict):
                raw_dates = calendar.get("Earnings Date", [])
                if not isinstance(raw_dates, list):
                    raw_dates = [raw_dates]
            else:
                raw_dates = []

            for raw in raw_dates:
                if hasattr(raw, "date"):
                    dt = raw.date()
                elif isinstance(raw, date):
                    dt = raw
                else:
                    continue

                if today <= dt <= today + timedelta(days=7):
                    earnings_within_7 = True
                    earnings_dt = dt
                    break
                if dt >= today and (earnings_dt is None or dt < earnings_dt):
                    earnings_dt = dt
    except Exception as e:
        logger.warning(f"Could not fetch earnings calendar for {symbol}: {e}")

    # --- Layer 1: Safety gate ---
    safety = _check_safety(ticker)

    rejection_reason: str | None = None
    value_trap: ValueTrapResult | None = None
    fast_fund: FastFundamentalsResult | None = None

    if not safety.passes:
        # Build rejection reason
        failures = []
        if not safety.current_ratio_ok:
            failures.append(f"Current ratio {safety.current_ratio or 'N/A'} < {CURRENT_RATIO_MIN}")
        if not safety.debt_to_equity_ok:
            failures.append(f"D/E {safety.debt_to_equity or 'N/A'} > {DEBT_TO_EQUITY_MAX}")
        if not safety.fcf_ok:
            failures.append("FCF not positive for 4 consecutive quarters")
        rejection_reason = "; ".join(failures)
    else:
        # --- Layer 2: Value trap detector ---
        value_trap = _check_value_trap(ticker)

        # --- Layer 3: Fast fundamentals ---
        fast_fund = _check_fast_fundamentals(ticker)

    score = _compute_score(safety, value_trap, fast_fund)

    dip_result = DipScreenerResult(
        safety=safety,
        value_trap=value_trap,
        fast_fundamentals=fast_fund,
        overall_score=score,
        rejection_reason=rejection_reason,
    )

    # Map back to standard FundamentalResult fields
    insider_buying = fast_fund.insider_buying if fast_fund is not None else False
    is_clear = safety.passes and not earnings_within_7

    return FundamentalResult(
        earnings_within_7_days=earnings_within_7,
        earnings_date=earnings_dt,
        insider_buying_detected=insider_buying,
        is_clear=is_clear,
        dip_screener=dip_result,
    )
