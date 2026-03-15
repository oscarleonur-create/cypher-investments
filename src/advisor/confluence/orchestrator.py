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
    MLResult,
    PeadScreenerResult,
    SentimentResult,
    TechnicalResult,
    VolumeConfirmationResult,
)
from advisor.confluence.pead_screener import check_pead_fundamental
from advisor.confluence.sentiment import check_sentiment
from advisor.confluence.technical import check_technical
from advisor.confluence.volume_confirmation import check_volume_confirmation

logger = logging.getLogger(__name__)


def _default_verdict(
    symbol: str,
    technical: TechnicalResult,
    sentiment: SentimentResult,
    fundamental: FundamentalResult,
) -> tuple[ConfluenceVerdict, str, int]:
    """Standard 3-way verdict for momentum/dip/sma strategies."""
    has_breakout = technical.is_bullish
    # Require grounding confidence >= 0.5 for sentiment to count as bullish.
    # A 30% confidence "bullish" sentiment is noise, not signal.
    sentiment_bullish = sentiment.is_bullish and sentiment.confidence >= 0.5
    confirmations = sum([sentiment_bullish, fundamental.is_clear])

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
        if not sentiment_bullish:
            failing.append("sentiment below 70% threshold")
        if not fundamental.is_clear:
            failing.append("earnings risk within 7 days")
        reasoning = (
            f"Breakout detected for {symbol} but only 2/3 checks passed. "
            f"Concern: {'; '.join(failing)}."
        )
        hold_days = 3
    elif not has_breakout and confirmations == 2:
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

    if sentiment.is_bullish and sentiment.confidence < 0.5:
        reasoning += (
            f" Note: sentiment marked bullish but grounding confidence is low "
            f"({sentiment.confidence:.0%}) — treated as unconfirmed."
        )

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

    return verdict, reasoning, hold_days


def _pead_verdict(
    symbol: str,
    technical: TechnicalResult,
    sentiment: SentimentResult,
    fundamental: FundamentalResult,
    ps: PeadScreenerResult,
) -> tuple[ConfluenceVerdict, str, int]:
    """PEAD-specific verdict: screener score is the primary signal.

    PEAD stocks have faded below their highs, so technical breakout is
    not expected. The PEAD screener score drives the verdict, with
    sentiment as a confirmation/contrarian indicator.
    """
    if ps is None:
        return ConfluenceVerdict.PASS, f"No PEAD screener data for {symbol}.", 0

    score = ps.overall_score
    is_actionable = score in ("STRONG_BUY", "BUY", "LEAN_BUY")

    if score == "FAIL":
        reasoning = f"PEAD screener rejected {symbol}: {ps.rejection_reason}."
        return ConfluenceVerdict.PASS, reasoning, 0

    if not is_actionable:
        reasoning = (
            f"PEAD screener scored {symbol} as {score} — "
            f"earnings beat confirmed but conviction is low."
        )
        return ConfluenceVerdict.PASS, reasoning, 0

    # Actionable score (LEAN_BUY / BUY / STRONG_BUY)
    es = ps.earnings_surprise
    fs = ps.fade_setup

    # Build detail string
    details = []
    if es.eps_surprise_pct is not None:
        details.append(f"EPS beat {es.eps_surprise_pct:.1f}%")
    if es.revenue_surprise:
        details.append("revenue beat")
    if fs is not None and fs.gap_and_fade:
        details.append("gap-and-fade confirmed")
    if fs is not None and fs.fade_pct is not None:
        details.append(f"faded {fs.fade_pct:.1%} from pre-earnings high")

    detail_str = ", ".join(details) if details else "earnings drift setup"

    # Sentiment acts as confirmation (bullish) or contrarian signal
    if score in ("STRONG_BUY", "BUY") and sentiment.is_bullish:
        verdict = ConfluenceVerdict.ENTER
        reasoning = (
            f"PEAD {score} for {symbol}: {detail_str}. "
            f"Sentiment confirms at {sentiment.positive_pct:.0f}% positive."
        )
        hold_days = 45
    elif score == "STRONG_BUY":
        # STRONG_BUY even without sentiment = CAUTION (contrarian)
        verdict = ConfluenceVerdict.CAUTION
        reasoning = (
            f"PEAD STRONG_BUY for {symbol}: {detail_str}. "
            f"Sentiment bearish ({sentiment.positive_pct:.0f}% positive) — "
            f"contrarian entry, post-earnings drift may overcome near-term negativity."
        )
        hold_days = 45
    elif score == "BUY":
        verdict = ConfluenceVerdict.CAUTION
        reasoning = (
            f"PEAD BUY for {symbol}: {detail_str}. "
            f"Sentiment at {sentiment.positive_pct:.0f}% positive — "
            f"wait for sentiment confirmation or add on next fade."
        )
        hold_days = 30
    else:
        # LEAN_BUY
        if sentiment.is_bullish:
            verdict = ConfluenceVerdict.CAUTION
            reasoning = (
                f"PEAD LEAN_BUY for {symbol}: {detail_str}. "
                f"Sentiment bullish ({sentiment.positive_pct:.0f}% positive) "
                f"provides additional confidence."
            )
            hold_days = 30
        else:
            verdict = ConfluenceVerdict.PASS
            reasoning = (
                f"PEAD LEAN_BUY for {symbol}: {detail_str}. "
                f"Low conviction — sentiment bearish ({sentiment.positive_pct:.0f}% positive), "
                f"consider monitoring for stronger setup."
            )
            hold_days = 0

    return verdict, reasoning, hold_days


def _compute_ml_signal(symbol: str) -> MLResult | None:
    """Compute the optional ML signal layer. Returns None on failure."""
    try:
        from advisor.ml.signal_generator import MLSignalGenerator

        gen = MLSignalGenerator()
        if not gen.is_available():
            return None

        explanation = gen.explain_prediction(symbol)
        if "error" in explanation:
            return None

        prob = explanation["win_probability"]
        if prob >= 0.65:
            confidence = "high" if prob >= 0.80 else "medium"
        elif prob <= 0.35:
            confidence = "high" if prob <= 0.20 else "medium"
        else:
            confidence = "low"

        return MLResult(
            win_probability=prob,
            signal=explanation["signal"],
            confidence=confidence,
            top_features=explanation.get("top_features", []),
            model_type=explanation.get("model_type", "unknown"),
            is_available=True,
        )
    except Exception as e:
        logger.warning(f"ML signal computation failed: {e}")
        return None


def _apply_ml_adjustment(
    verdict: ConfluenceVerdict,
    reasoning: str,
    hold_days: int,
    ml: MLResult | None,
) -> tuple[ConfluenceVerdict, str, int]:
    """Apply ML signal adjustment to the base verdict.

    Conservative rules:
    - ML BUY + high confidence: can upgrade CAUTION -> ENTER
    - ML SELL + high confidence: can downgrade CAUTION -> PASS
    - ML never overrides consensus ENTER or PASS alone
    """
    if ml is None or not ml.is_available:
        return verdict, reasoning, hold_days

    ml_note = (
        f" ML signal: {ml.signal} ({ml.win_probability:.0%} win prob,"
        f" {ml.confidence} confidence)."
    )

    if verdict == ConfluenceVerdict.CAUTION:
        if ml.signal == "BUY" and ml.confidence == "high":
            reasoning += ml_note + " ML upgrades CAUTION to ENTER."
            return ConfluenceVerdict.ENTER, reasoning, max(hold_days, 5)
        elif ml.signal == "SELL" and ml.confidence == "high":
            reasoning += ml_note + " ML downgrades CAUTION to PASS."
            return ConfluenceVerdict.PASS, reasoning, 0

    reasoning += ml_note
    return verdict, reasoning, hold_days


def run_confluence(
    symbol: str,
    strategy_name: str = "momentum_breakout",
    force_all: bool = False,
    include_ml: bool = True,
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

    # ── PEAD fast path: run screener before sentiment ────────────────────
    # For PEAD, the screener is cheap (yfinance only). If it rejects the
    # stock, skip the expensive sentiment call entirely.
    if strategy_name == "pead":
        fundamental = check_pead_fundamental(symbol)
        ps = fundamental.pead_screener
        logger.info(
            f"Fundamental (PEAD screener): is_clear={fundamental.is_clear}, "
            f"score={ps.overall_score if ps else 'N/A'}"
        )

        is_actionable = ps is not None and ps.overall_score in (
            "STRONG_BUY",
            "BUY",
            "LEAN_BUY",
            "WATCH",
        )
        if not is_actionable:
            # FAIL — skip sentiment entirely
            logger.info("PEAD screener rejected — skipping sentiment (saves API cost)")
            sentiment = SentimentResult(
                score=0.0,
                positive_pct=0.0,
                key_headlines=["Skipped — PEAD screener rejected"],
                sources=[],
                is_bullish=False,
            )
            verdict, reasoning, hold_days = _pead_verdict(
                symbol, technical, sentiment, fundamental, ps
            )
        else:
            # Actionable — run sentiment for confirmation
            logger.info("PEAD screener passed — searching news sentiment...")
            sentiment = check_sentiment(symbol)
            logger.info(
                f"Sentiment: is_bullish={sentiment.is_bullish} " f"({sentiment.positive_pct:.0f}%)"
            )
            verdict, reasoning, hold_days = _pead_verdict(
                symbol, technical, sentiment, fundamental, ps
            )

        # ── ML signal (PEAD path) ─────────────────────────────────────
        ml = _compute_ml_signal(symbol) if include_ml else None
        if ml is not None:
            verdict, reasoning, hold_days = _apply_ml_adjustment(verdict, reasoning, hold_days, ml)

        return ConfluenceResult(
            symbol=symbol,
            strategy_name=strategy_name,
            verdict=verdict,
            technical=technical,
            sentiment=sentiment,
            fundamental=fundamental,
            ml_signal=ml,
            reasoning=reasoning,
            suggested_hold_days=hold_days,
        )

    # ── Step 2: Search sentiment (non-PEAD strategies) ────────────────────
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

    # ── Volume confirmation ─────────────────────────────────────────────
    vol_confirm: VolumeConfirmationResult | None = None
    try:
        vol_confirm = check_volume_confirmation(symbol)
        if vol_confirm.score > 0:
            logger.info(
                f"Volume confirmation: score={vol_confirm.score:.0f}, "
                f"capitulation={vol_confirm.capitulation_detected}, "
                f"dryup={vol_confirm.volume_dryup}, obv_div={vol_confirm.obv_divergence}"
            )
    except Exception as e:
        logger.warning(f"Volume confirmation failed: {e}")

    # ── Verdict ──────────────────────────────────────────────────────────
    verdict, reasoning, hold_days = _default_verdict(symbol, technical, sentiment, fundamental)

    # Append volume confirmation detail to reasoning
    if vol_confirm is not None and vol_confirm.score >= 40:
        signals = []
        if vol_confirm.capitulation_detected:
            signals.append("capitulation spike")
        if vol_confirm.volume_dryup:
            signals.append("volume dry-up")
        if vol_confirm.obv_divergence:
            signals.append("OBV divergence")
        reasoning += (
            f" Volume confirmation score {vol_confirm.score:.0f}/100 ({', '.join(signals)})."
        )

    # ── ML signal (default path) ──────────────────────────────────────
    ml = _compute_ml_signal(symbol) if include_ml else None
    if ml is not None:
        verdict, reasoning, hold_days = _apply_ml_adjustment(verdict, reasoning, hold_days, ml)

    return ConfluenceResult(
        symbol=symbol,
        strategy_name=strategy_name,
        verdict=verdict,
        technical=technical,
        sentiment=sentiment,
        fundamental=fundamental,
        ml_signal=ml,
        volume_confirmation=vol_confirm,
        reasoning=reasoning,
        suggested_hold_days=hold_days,
    )
