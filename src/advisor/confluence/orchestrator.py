"""Orchestrator — runs the full confluence pipeline for any strategy.

Pipeline:
  1. Technical — run the strategy via SignalScanner for breakout signal
  2. If no breakout → PASS (no API spend)
  3. If breakout → search sentiment + check fundamentals
  4. Verdict: ENTER (all 3), CAUTION (2/3), PASS (<2)
"""

from __future__ import annotations

import logging

from advisor.confluence.dip_screener import check_dip_fundamental
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


def run_confluence(
    symbol: str,
    strategy_name: str = "momentum_breakout",
    force_all: bool = False,
) -> ConfluenceResult:
    """Run the full confluence scan for any strategy.

    Args:
        symbol: Ticker symbol to scan.
        strategy_name: Registered strategy name to use for the technical check.
        force_all: If True, run sentiment + fundamental checks even without
            a technical breakout. Useful for dip-buying analysis where the
            stock is below its SMAs by design.

    Returns:
        ConfluenceResult with verdict, all agent results, and reasoning.
    """
    symbol = symbol.upper()
    logger.info(f"Confluence scan for {symbol} via {strategy_name} (force_all={force_all})")

    # ── Step 1: Technical breakout ───────────────────────────────────────
    technical = check_technical(symbol, strategy_name=strategy_name)
    logger.info(f"Technical: signal={technical.signal}, is_bullish={technical.is_bullish}")

    if not technical.is_bullish and not force_all:
        return ConfluenceResult(
            symbol=symbol,
            strategy_name=strategy_name,
            verdict=ConfluenceVerdict.PASS,
            technical=technical,
            sentiment=SentimentResult(
                score=0.0,
                positive_pct=0.0,
                key_headlines=["Skipped — no technical breakout"],
                sources=[],
                is_bullish=False,
            ),
            fundamental=FundamentalResult(
                earnings_within_7_days=False,
                earnings_date=None,
                insider_buying_detected=False,
                is_clear=False,
            ),
            reasoning=(
                f"No breakout for {symbol} via {strategy_name} "
                f"(signal: {technical.signal}, price: ${technical.price:,.2f}). "
                f"Sentiment and fundamental checks skipped."
            ),
            suggested_hold_days=0,
        )

    # ── Step 2: Search sentiment ─────────────────────────────────────────
    if technical.is_bullish:
        logger.info("Breakout detected — searching news sentiment...")
    else:
        logger.info("No breakout but force_all=True — searching news sentiment...")
    sentiment = check_sentiment(symbol)
    logger.info(f"Sentiment: is_bullish={sentiment.is_bullish} ({sentiment.positive_pct:.0f}%)")

    # ── Step 3: Fundamental check ────────────────────────────────────────
    logger.info("Checking earnings risk and insider activity...")
    if strategy_name == "buy_the_dip":
        fundamental = check_dip_fundamental(symbol)
        logger.info(
            f"Fundamental (dip screener): is_clear={fundamental.is_clear}, "
            f"score={fundamental.dip_screener.overall_score if fundamental.dip_screener else 'N/A'}"
        )
    else:
        fundamental = check_fundamental(symbol)
        logger.info(f"Fundamental: is_clear={fundamental.is_clear}")

    # ── Verdict ──────────────────────────────────────────────────────────
    has_breakout = technical.is_bullish
    confirmations = sum([sentiment.is_bullish, fundamental.is_clear])

    if has_breakout and confirmations == 2:
        verdict = ConfluenceVerdict.ENTER
        reasoning = (
            f"All three checks aligned for {symbol}: "
            f"technical breakout confirmed, sentiment >70% positive, "
            f"no earnings risk."
        )
        hold_days = 5
    elif has_breakout and confirmations == 1:
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
    elif not has_breakout and confirmations == 2:
        # Forced mode: no breakout but sentiment + fundamentals both bullish
        verdict = ConfluenceVerdict.CAUTION
        reasoning = (
            f"No technical breakout for {symbol} "
            f"(signal: {technical.signal}, price: ${technical.price:,.2f}), "
            f"but sentiment is bullish ({sentiment.positive_pct:.0f}% positive) "
            f"and fundamentals are clear. Dip may be buyable — wait for confirmation candle."
        )
        hold_days = 3
    elif not has_breakout and confirmations == 1:
        verdict = ConfluenceVerdict.PASS
        passing = []
        if sentiment.is_bullish:
            passing.append("sentiment bullish")
        if fundamental.is_clear:
            passing.append("fundamentals clear")
        failing = []
        if not sentiment.is_bullish:
            failing.append("sentiment bearish")
        if not fundamental.is_clear:
            failing.append("fundamental risk")
        reasoning = (
            f"No breakout for {symbol}. Mixed signals: {'; '.join(passing)} "
            f"but {'; '.join(failing)}."
        )
        hold_days = 0
    else:
        verdict = ConfluenceVerdict.PASS
        reasoning = (
            f"{'Breakout detected' if has_breakout else 'No breakout'} for {symbol} "
            f"but both sentiment and fundamental checks failed."
        )
        hold_days = 0

    if fundamental.insider_buying_detected and verdict != ConfluenceVerdict.PASS:
        reasoning += " Insider buying detected — strengthens conviction."

    # Dip screener detail
    ds = fundamental.dip_screener
    if ds is not None:
        if ds.overall_score == "FAIL":
            reasoning += f" Dip screener FAIL: {ds.rejection_reason}."
        elif ds.overall_score == "STRONG_BUY":
            reasoning += " Dip screener STRONG_BUY: RSI divergence with C-suite insider buying."
        elif ds.overall_score in ("BUY", "LEAN_BUY", "WATCH", "WEAK"):
            reasoning += f" Dip screener score: {ds.overall_score}."

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
