"""Orchestrator — runs the full confluence pipeline for any strategy.

Pipeline:
  1. Technical — run the strategy via SignalScanner for breakout signal
  2. If no breakout → PASS (no API spend)
  3. If breakout → search sentiment + check fundamentals
  4. Verdict: ENTER (all 3), CAUTION (2/3), PASS (<2)
"""

from __future__ import annotations

import logging

from advisor.confluence.fundamental import check_fundamental
from advisor.confluence.models import (
    ConfluenceResult,
    ConfluenceVerdict,
    FundamentalResult,
    SentimentResult,
)
from advisor.confluence.sentiment import check_sentiment
from advisor.confluence.technical import check_technical

logger = logging.getLogger(__name__)


def run_confluence(symbol: str, strategy_name: str = "momentum_breakout") -> ConfluenceResult:
    """Run the full confluence scan for any strategy.

    Args:
        symbol: Ticker symbol to scan.
        strategy_name: Registered strategy name to use for the technical check.

    Returns:
        ConfluenceResult with verdict, all agent results, and reasoning.
    """
    symbol = symbol.upper()
    logger.info(f"Confluence scan for {symbol} via {strategy_name}")

    # ── Step 1: Technical breakout ───────────────────────────────────────
    technical = check_technical(symbol, strategy_name=strategy_name)
    logger.info(f"Technical: signal={technical.signal}, is_bullish={technical.is_bullish}")

    if not technical.is_bullish:
        return ConfluenceResult(
            symbol=symbol,
            strategy_name=strategy_name,
            verdict=ConfluenceVerdict.PASS,
            technical=technical,
            sentiment=SentimentResult(
                score=0.0, positive_pct=0.0,
                key_headlines=["Skipped — no technical breakout"],
                sources=[], is_bullish=False,
            ),
            fundamental=FundamentalResult(
                earnings_within_7_days=False, earnings_date=None,
                insider_buying_detected=False, is_clear=False,
            ),
            reasoning=(
                f"No breakout for {symbol} via {strategy_name} "
                f"(signal: {technical.signal}, price: ${technical.price:,.2f}). "
                f"Sentiment and fundamental checks skipped."
            ),
            suggested_hold_days=0,
        )

    # ── Step 2: Search sentiment ─────────────────────────────────────────
    logger.info("Breakout detected — searching news sentiment...")
    sentiment = check_sentiment(symbol)
    logger.info(f"Sentiment: is_bullish={sentiment.is_bullish} ({sentiment.positive_pct:.0f}%)")

    # ── Step 3: Fundamental check ────────────────────────────────────────
    logger.info("Checking earnings risk and insider activity...")
    fundamental = check_fundamental(symbol)
    logger.info(f"Fundamental: is_clear={fundamental.is_clear}")

    # ── Verdict ──────────────────────────────────────────────────────────
    confirmations = sum([sentiment.is_bullish, fundamental.is_clear])

    if confirmations == 2:
        verdict = ConfluenceVerdict.ENTER
        reasoning = (
            f"All three checks aligned for {symbol}: "
            f"technical breakout confirmed, sentiment >70% positive, "
            f"no earnings risk."
        )
        hold_days = 5
    elif confirmations == 1:
        verdict = ConfluenceVerdict.CAUTION
        failing = []
        if not sentiment.is_bullish:
            failing.append("sentiment below 70% threshold")
        if not fundamental.is_clear:
            failing.append("earnings risk within 7 days")
        reasoning = (
            f"Breakout detected for {symbol} but only 2/3 checks passed. "
            f"Concern: {'; '.join(failing)}."
        )
        hold_days = 3
    else:
        verdict = ConfluenceVerdict.PASS
        reasoning = (
            f"Breakout detected for {symbol} but both sentiment and "
            f"fundamental checks failed."
        )
        hold_days = 0

    if fundamental.insider_buying_detected and verdict != ConfluenceVerdict.PASS:
        reasoning += " Insider buying detected — strengthens conviction."

    return ConfluenceResult(
        symbol=symbol,
        strategy_name=strategy_name,
        verdict=verdict,
        technical=technical,
        sentiment=sentiment,
        fundamental=fundamental,
        reasoning=reasoning,
        suggested_hold_days=hold_days,
    )
