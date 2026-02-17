"""PEAD (Post-Earnings Announcement Drift) screener.

Layers:
  1. Earnings Surprise — EPS beat >5% reported within last 7 days
  2. Fade Setup — price faded below pre-earnings high (gap-and-fade)

Scoring (requires BOTH layers to pass):
  EPS >10% (+2), >5% (+1), revenue beat (+1), gap-and-fade (+1), deep fade >5% (+1)
  4+ = STRONG_BUY, 3 = BUY, 2 = LEAN_BUY, else WATCH
"""

from __future__ import annotations

import logging
from datetime import date

import pandas as pd
import yfinance as yf

from advisor.confluence.models import (
    EarningsSurpriseResult,
    FadeSetupResult,
    FundamentalResult,
    PeadScreenerResult,
)

logger = logging.getLogger(__name__)

# ── Thresholds ────────────────────────────────────────────────────────────────
EPS_SURPRISE_MIN_PCT = 5.0
PEAD_WINDOW_DAYS = 7  # single window: must report within 7 calendar days
FADE_MIN_DAYS = 2  # min days after report before entry
PRE_EARNINGS_LOOKBACK = 5  # trading days


def _check_earnings_surprise(ticker: yf.Ticker) -> EarningsSurpriseResult:
    """Layer 1: Check if the stock recently beat earnings by >5%."""
    result = EarningsSurpriseResult()
    today = date.today()

    try:
        earnings_dates = ticker.earnings_dates
        if earnings_dates is None or earnings_dates.empty:
            return result

        # Filter to rows that have reported (Reported EPS is not NaN)
        if "Reported EPS" not in earnings_dates.columns:
            return result

        reported = earnings_dates[earnings_dates["Reported EPS"].notna()]
        if reported.empty:
            return result

        # Get the most recent reported earnings
        latest = reported.iloc[0]
        report_idx = reported.index[0]

        # Extract the report date
        if hasattr(report_idx, "date"):
            report_date = report_idx.date()
        elif isinstance(report_idx, date):
            report_date = report_idx
        else:
            return result

        result.reported_date = report_date
        result.days_since_report = (today - report_date).days

        # Must be within the PEAD window
        if result.days_since_report < 0 or result.days_since_report > PEAD_WINDOW_DAYS:
            return result

        # Compute EPS surprise
        eps_estimate = latest.get("EPS Estimate")
        eps_actual = latest.get("Reported EPS")

        if pd.isna(eps_estimate) or pd.isna(eps_actual):
            return result

        eps_estimate = float(eps_estimate)
        eps_actual = float(eps_actual)
        result.eps_estimate = eps_estimate
        result.eps_actual = eps_actual

        if abs(eps_estimate) > 0:
            result.eps_surprise_pct = round(
                (eps_actual - eps_estimate) / abs(eps_estimate) * 100, 2
            )

        # Revenue surprise: compare latest quarter to year-ago quarter
        result.revenue_surprise = _check_revenue_surprise(ticker)

        # Pass gate: surprise >5% AND reported within window
        result.passes = (
            result.eps_surprise_pct is not None and result.eps_surprise_pct >= EPS_SURPRISE_MIN_PCT
        )

    except Exception as e:
        logger.warning(f"Could not check earnings surprise: {e}")

    return result


def _check_revenue_surprise(ticker: yf.Ticker) -> bool | None:
    """Check quarterly revenue growth vs year-ago quarter.

    Uses quarterly_financials to compare the most recent quarter's revenue
    to the same quarter one year prior, which is more accurate than the
    trailing annual revenueGrowth from ticker.info.
    """
    try:
        qf = ticker.quarterly_financials
        if qf is None or qf.empty:
            # Fall back to ticker.info revenueGrowth
            growth = ticker.info.get("revenueGrowth")
            return growth > 0 if growth is not None else None

        # Look for a revenue row
        rev_row = None
        for label in ("Total Revenue", "TotalRevenue", "Revenue"):
            if label in qf.index:
                rev_row = qf.loc[label].dropna()
                break

        if rev_row is None or len(rev_row) < 5:
            # Not enough quarters; fall back
            growth = ticker.info.get("revenueGrowth")
            return growth > 0 if growth is not None else None

        # Most recent quarter vs 4 quarters ago (year-over-year)
        latest_rev = float(rev_row.iloc[0])
        yoy_rev = float(rev_row.iloc[4])
        if yoy_rev > 0:
            return latest_rev > yoy_rev

    except Exception as e:
        logger.debug(f"Could not check quarterly revenue: {e}")

    # Final fallback
    try:
        growth = ticker.info.get("revenueGrowth")
        return growth > 0 if growth is not None else None
    except Exception:
        return None


def _check_fade_setup(
    ticker: yf.Ticker,
    report_date: date,
) -> FadeSetupResult:
    """Layer 2: Check if the stock has faded below pre-earnings high."""
    result = FadeSetupResult()
    today = date.today()
    result.days_since_earnings = (today - report_date).days

    try:
        # Fetch enough history to cover pre-earnings + post-earnings window
        lookback_days = PRE_EARNINGS_LOOKBACK + result.days_since_earnings + 10
        hist = ticker.history(period=f"{lookback_days}d")
        if hist is None or hist.empty or len(hist) < 5:
            return result

        # Convert index to dates for comparison
        hist_dates = hist.index.normalize().tz_localize(None)
        report_dt = pd.Timestamp(report_date)

        # Pre-earnings high: max HIGH in 5 trading days before report
        pre_mask = hist_dates < report_dt
        pre_earnings = hist[pre_mask]
        if pre_earnings.empty:
            return result

        pre_earnings_window = pre_earnings.tail(PRE_EARNINGS_LOOKBACK)
        pre_earnings_high = float(pre_earnings_window["High"].max())
        result.pre_earnings_high = round(pre_earnings_high, 2)

        # Current price
        current_price = float(hist["Close"].iloc[-1])
        result.current_price = round(current_price, 2)

        # Fade percentage
        if pre_earnings_high > 0:
            result.fade_pct = round((current_price - pre_earnings_high) / pre_earnings_high, 4)

        # Has faded: price is below pre-earnings high
        result.has_faded = current_price < pre_earnings_high

        # Gap-and-fade: gapped up on earnings day then fell back below pre-earnings high
        earnings_mask = hist_dates >= report_dt
        post_earnings = hist[earnings_mask]
        if not post_earnings.empty:
            earnings_day_high = float(post_earnings["High"].iloc[0])
            result.gap_and_fade = (
                earnings_day_high > pre_earnings_high and current_price < pre_earnings_high
            )

        # Pass gate: price below pre-earnings high AND at least FADE_MIN_DAYS
        # since report AND within PEAD_WINDOW_DAYS
        result.passes = (
            result.has_faded
            and result.days_since_earnings is not None
            and FADE_MIN_DAYS <= result.days_since_earnings <= PEAD_WINDOW_DAYS
        )

    except Exception as e:
        logger.warning(f"Could not check fade setup: {e}")

    return result


def _compute_pead_score(
    surprise: EarningsSurpriseResult,
    fade: FadeSetupResult | None,
) -> str:
    """Compute PEAD score: FAIL / WATCH / LEAN_BUY / BUY / STRONG_BUY.

    Both layers must pass for any score above FAIL.
    """
    if not surprise.passes:
        return "FAIL"

    if fade is None or not fade.passes:
        return "FAIL"

    points = 0

    # EPS surprise magnitude
    if surprise.eps_surprise_pct is not None:
        if surprise.eps_surprise_pct >= 10.0:
            points += 2
        elif surprise.eps_surprise_pct >= EPS_SURPRISE_MIN_PCT:
            points += 1

    # Revenue beat
    if surprise.revenue_surprise:
        points += 1

    # Gap-and-fade pattern
    if fade.gap_and_fade:
        points += 1

    # Deep fade (>5% below pre-earnings high)
    if fade.fade_pct is not None and fade.fade_pct < -0.05:
        points += 1

    if points >= 4:
        return "STRONG_BUY"
    if points >= 3:
        return "BUY"
    if points >= 2:
        return "LEAN_BUY"
    return "WATCH"


def check_pead_fundamental(symbol: str) -> FundamentalResult:
    """Run the PEAD screener and return a FundamentalResult.

    Compatible with the orchestrator's expected return type:
    - is_clear reflects whether both surprise and fade gates passed
    - pead_screener holds the full breakdown
    """
    ticker = yf.Ticker(symbol)

    # --- Layer 1: Earnings surprise ---
    surprise = _check_earnings_surprise(ticker)

    rejection_reason: str | None = None
    fade: FadeSetupResult | None = None

    if not surprise.passes:
        if (
            surprise.eps_surprise_pct is not None
            and surprise.eps_surprise_pct < EPS_SURPRISE_MIN_PCT
        ):
            rejection_reason = (
                f"EPS surprise {surprise.eps_surprise_pct:.1f}% < {EPS_SURPRISE_MIN_PCT}% threshold"
            )
        elif (
            surprise.days_since_report is not None and surprise.days_since_report > PEAD_WINDOW_DAYS
        ):
            rejection_reason = (
                f"Earnings reported {surprise.days_since_report} days ago "
                f"(>{PEAD_WINDOW_DAYS} day window)"
            )
        else:
            rejection_reason = "No recent earnings report with sufficient EPS beat"
    else:
        # --- Layer 2: Fade setup ---
        fade = _check_fade_setup(ticker, surprise.reported_date)
        if not fade.passes:
            if not fade.has_faded:
                rejection_reason = "Price has not faded below pre-earnings high"
            elif fade.days_since_earnings is not None:
                if fade.days_since_earnings < FADE_MIN_DAYS:
                    rejection_reason = (
                        f"Only {fade.days_since_earnings} day(s) since earnings "
                        f"(need {FADE_MIN_DAYS}+)"
                    )
                elif fade.days_since_earnings > PEAD_WINDOW_DAYS:
                    rejection_reason = (
                        f"{fade.days_since_earnings} days since earnings "
                        f"(>{PEAD_WINDOW_DAYS} day window expired)"
                    )

    score = _compute_pead_score(surprise, fade)

    pead_result = PeadScreenerResult(
        earnings_surprise=surprise,
        fade_setup=fade,
        overall_score=score,
        rejection_reason=rejection_reason,
    )

    is_clear = surprise.passes and (fade is not None and fade.passes)

    # earnings_within_7_days should reflect actual state
    has_recent_earnings = (
        surprise.days_since_report is not None
        and 0 <= surprise.days_since_report <= PEAD_WINDOW_DAYS
    )

    return FundamentalResult(
        earnings_within_7_days=has_recent_earnings,
        earnings_date=surprise.reported_date,
        insider_buying_detected=False,
        is_clear=is_clear,
        pead_screener=pead_result,
    )
