"""Fundamental agent — checks earnings risk and insider buying via yfinance."""

from __future__ import annotations

import logging
from datetime import date, timedelta

import yfinance as yf

from advisor.confluence.models import FundamentalResult

logger = logging.getLogger(__name__)


def check_fundamental(symbol: str, use_research: bool = False) -> FundamentalResult:
    """Check for earnings risk and insider buying confirmation.

    Uses yfinance to:
    - Flag earnings within the next 7 days as risk
    - Detect recent insider purchases as confirmation

    Returns FundamentalResult with is_clear=True when no imminent earnings risk.
    """
    ticker = yf.Ticker(symbol)
    today = date.today()

    # --- Earnings date check ---
    earnings_within_7 = False
    earnings_dt: date | None = None

    try:
        calendar = ticker.calendar
        if calendar is not None:
            # calendar can be a dict with 'Earnings Date' key or similar
            if isinstance(calendar, dict):
                raw_dates = calendar.get("Earnings Date", [])
                if not isinstance(raw_dates, list):
                    raw_dates = [raw_dates]
            else:
                # DataFrame format — try to get the first row
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
                # Keep the nearest future date
                if dt >= today and (earnings_dt is None or dt < earnings_dt):
                    earnings_dt = dt
    except Exception as e:
        logger.warning(f"Could not fetch earnings calendar for {symbol}: {e}")

    # --- Insider transactions check ---
    insider_buying = False

    try:
        transactions = ticker.insider_transactions
        if transactions is not None and not transactions.empty:
            # Check Transaction column for explicit purchase labels
            if "Transaction" in transactions.columns:
                non_empty = transactions["Transaction"].str.strip().ne("")
                if non_empty.any():
                    buys = transactions[
                        transactions["Transaction"].str.contains(
                            "Purchase|Buy", case=False, na=False
                        )
                    ]
                    insider_buying = len(buys) > 0
                elif "Shares" in transactions.columns:
                    # Transaction column exists but all empty — fall back to Shares
                    insider_buying = bool((transactions["Shares"] > 0).any())
            elif "Shares" in transactions.columns:
                insider_buying = bool((transactions["Shares"] > 0).any())
    except Exception as e:
        logger.warning(f"Could not fetch insider transactions for {symbol}: {e}")

    is_clear = not earnings_within_7
    research_notes = _research_notes(symbol) if use_research else []

    return FundamentalResult(
        earnings_within_7_days=earnings_within_7,
        earnings_date=earnings_dt,
        insider_buying_detected=insider_buying,
        is_clear=is_clear,
        research_notes=research_notes,
    )


def _research_notes(symbol: str) -> list[str]:
    """Load the latest cached ResearchReport and extract key notes."""
    try:
        from advisor.research.config import get_settings
        from advisor.research.store import ResearchStore

        report = ResearchStore(get_settings().db_path).load_latest_report(symbol.upper())
        if report is None:
            return []

        notes: list[str] = []

        if report.ratios and report.ratios.latest():
            r = report.ratios.latest()
            if r.roic is not None:
                notes.append(f"ROIC {r.roic:.1%}")
            if r.debt_to_ebitda is not None:
                notes.append(f"D/EBITDA {r.debt_to_ebitda:.1f}x")
            if r.fcf_margin is not None:
                notes.append(f"FCF margin {r.fcf_margin:.1%}")

        if report.red_flags and report.red_flags.flags:
            high = report.red_flags.high_severity_count
            total = len(report.red_flags.flags)
            notes.append(f"Red flags: {total} ({high} HIGH)")

        if report.dcf and report.dcf.base:
            upside = report.dcf.base.upside_pct
            notes.append(f"DCF base upside {upside:+.1%}")

        if report.thesis and report.thesis.conviction:
            notes.append(f"Thesis conviction: {report.thesis.conviction}")

        return notes
    except Exception as exc:  # noqa: BLE001
        logger.warning("Research notes failed for %s: %s", symbol, exc)
        return []
