"""Alpha Scorer — unified 0-100 composite conviction score across all signal layers.

Each layer is called independently (not through the confluence pipeline, which
gates on technical breakout).  Results are normalized to 0-100, weighted, and
combined into a single alpha score.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Callable

from advisor.confluence.models import (
    AlphaLayerScore,
    AlphaResult,
    AlphaSignal,
)

logger = logging.getLogger(__name__)

# ── Default weights ──────────────────────────────────────────────────────

DEFAULT_WEIGHTS: dict[str, float] = {
    "smart_money": 0.25,
    "mispricing": 0.20,
    "ml_signal": 0.15,
    "sentiment": 0.15,
    "technical": 0.10,
    "fundamental": 0.05,
    "dip_screener": 0.05,
    "pead_screener": 0.05,
}

# ── Signal classification ───────────────────────────────────────────────


def classify_signal(score: float) -> AlphaSignal:
    """Map a 0-100 alpha score to a discrete signal."""
    if score >= 80:
        return AlphaSignal.STRONG_BUY
    if score >= 65:
        return AlphaSignal.BUY
    if score >= 55:
        return AlphaSignal.LEAN_BUY
    if score >= 40:
        return AlphaSignal.NEUTRAL
    if score >= 25:
        return AlphaSignal.LEAN_SELL
    return AlphaSignal.AVOID


# ── Normalizers (raw result → 0-100) ────────────────────────────────────


def _normalize_technical(result: Any) -> float:
    """Normalize TechnicalResult to 0-100.

    Bullish breakout → 70 + volume bonus (up to 100).
    No breakout → 0-55 based on price-vs-SMA proximity.
    """
    if result.is_bullish:
        vol_bonus = min((result.volume_ratio - 1.0) * 30, 30) if result.volume_ratio > 1.0 else 0
        return min(70 + vol_bonus, 100)
    # No breakout: proximity of price to SMA
    if result.sma_20 > 0:
        ratio = result.price / result.sma_20
        # ratio 1.0 = at SMA → 55, ratio 0.9 → 0
        return max(0.0, min(55.0, (ratio - 0.9) / 0.1 * 55))
    return 0.0


def _normalize_sentiment(result: Any) -> float:
    """Sentiment score is already 0-100."""
    return max(0.0, min(100.0, result.score))


def _normalize_fundamental(result: Any) -> float:
    """Fundamental is_clear → 60, +insider_buying → +40."""
    score = 0.0
    if result.is_clear:
        score += 60
    if result.insider_buying_detected:
        score += 40
    return score


def _normalize_smart_money(result: Any) -> float:
    """Linear map from [-35, +100] → [0, 100]."""
    raw = result.total_score
    return max(0.0, min(100.0, (raw + 35) / 135 * 100))


def _normalize_mispricing(result: Any) -> float:
    """Mispricing total_score is already 0-100."""
    return max(0.0, min(100.0, result.total_score))


def _normalize_ml(result: Any) -> float:
    """ML win_probability * 100."""
    return max(0.0, min(100.0, result.win_probability * 100))


_DIP_SCORE_MAP = {
    "FAIL": 0,
    "WEAK": 20,
    "WATCH": 40,
    "LEAN_BUY": 60,
    "BUY": 80,
    "STRONG_BUY": 100,
}


def _normalize_dip(result: Any) -> float:
    """Map dip screener overall_score string to 0-100."""
    return float(_DIP_SCORE_MAP.get(result.overall_score, 0))


_PEAD_SCORE_MAP = {
    "FAIL": 0,
    "WATCH": 30,
    "LEAN_BUY": 55,
    "BUY": 80,
    "STRONG_BUY": 100,
}


def _normalize_pead(result: Any) -> float:
    """Map PEAD screener overall_score string to 0-100."""
    return float(_PEAD_SCORE_MAP.get(result.overall_score, 0))


# ── Layer runners ────────────────────────────────────────────────────────


def _run_technical(symbol: str) -> Any:
    from advisor.confluence.technical import check_technical

    return check_technical(symbol)


def _run_sentiment(symbol: str) -> Any:
    from advisor.confluence.sentiment import check_sentiment

    return check_sentiment(symbol)


def _run_fundamental(symbol: str) -> Any:
    from advisor.confluence.fundamental import check_fundamental

    return check_fundamental(symbol)


def _run_smart_money(symbol: str) -> Any:
    from advisor.confluence.smart_money_screener import screen_smart_money

    return screen_smart_money(symbol)


def _run_mispricing(symbol: str) -> Any:
    from advisor.confluence.mispricing_screener import screen_mispricing

    return screen_mispricing(symbol)


def _run_ml(symbol: str) -> Any:
    from advisor.ml.signal_generator import MLSignalGenerator

    gen = MLSignalGenerator()
    if not gen.is_available():
        return None
    explanation = gen.explain_prediction(symbol)
    if "error" in explanation:
        return None

    from advisor.confluence.models import MLResult

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


def _run_dip(symbol: str) -> Any:
    from advisor.confluence.dip_screener import check_dip_fundamental

    result = check_dip_fundamental(symbol)
    return result.dip_screener


def _run_pead(symbol: str) -> Any:
    from advisor.confluence.pead_screener import check_pead_fundamental

    result = check_pead_fundamental(symbol)
    return result.pead_screener


# ── Layer registry ───────────────────────────────────────────────────────

_LayerDef = tuple[str, Callable[[str], Any], Callable[[Any], float]]

_LAYERS: list[_LayerDef] = [
    ("technical", _run_technical, _normalize_technical),
    ("sentiment", _run_sentiment, _normalize_sentiment),
    ("fundamental", _run_fundamental, _normalize_fundamental),
    ("smart_money", _run_smart_money, _normalize_smart_money),
    ("mispricing", _run_mispricing, _normalize_mispricing),
    ("ml_signal", _run_ml, _normalize_ml),
    ("dip_screener", _run_dip, _normalize_dip),
    ("pead_screener", _run_pead, _normalize_pead),
]


# ── Core scoring function ───────────────────────────────────────────────


def compute_alpha(
    symbol: str,
    weights: dict[str, float] | None = None,
    skip_layers: set[str] | None = None,
) -> AlphaResult:
    """Compute a unified alpha score for *symbol* across all signal layers.

    Args:
        symbol: Ticker symbol.
        weights: Override default layer weights.  Keys must match layer names.
        skip_layers: Layer names to skip entirely.

    Returns:
        AlphaResult with composite score, signal, and per-layer breakdown.
    """
    symbol = symbol.upper()
    w = dict(DEFAULT_WEIGHTS) if weights is None else dict(weights)
    skip = skip_layers or set()

    layer_scores: list[AlphaLayerScore] = []

    for name, runner, normalizer in _LAYERS:
        if name in skip:
            layer_scores.append(
                AlphaLayerScore(
                    name=name,
                    available=False,
                    error="skipped",
                )
            )
            continue

        try:
            raw = runner(symbol)
            if raw is None:
                layer_scores.append(
                    AlphaLayerScore(
                        name=name,
                        available=False,
                        error="no data",
                    )
                )
                continue

            normalized = normalizer(raw)
            layer_scores.append(
                AlphaLayerScore(
                    name=name,
                    raw_value=getattr(raw, "total_score", None)
                    or getattr(raw, "score", None)
                    or getattr(raw, "win_probability", None),
                    normalized=round(normalized, 2),
                    available=True,
                )
            )
        except Exception as e:
            logger.warning(f"Alpha layer {name} failed for {symbol}: {e}")
            layer_scores.append(
                AlphaLayerScore(
                    name=name,
                    available=False,
                    error=str(e),
                )
            )

    # ── Weight redistribution ────────────────────────────────────────
    active = [ls for ls in layer_scores if ls.available]
    total_layers = len(_LAYERS)
    active_count = len(active)

    if active_count == 0:
        return AlphaResult(
            symbol=symbol,
            alpha_score=0.0,
            signal=AlphaSignal.AVOID,
            layers=layer_scores,
            active_layers=0,
            total_layers=total_layers,
        )

    active_weight_sum = sum(w.get(ls.name, 0) for ls in active)
    scale = 1.0 / active_weight_sum if active_weight_sum > 0 else 0

    composite = 0.0
    for ls in layer_scores:
        if not ls.available:
            continue
        base_w = w.get(ls.name, 0)
        adj_w = base_w * scale
        ls.weight = round(adj_w, 4)
        ls.weighted_contribution = round(ls.normalized * adj_w, 2)
        composite += ls.weighted_contribution

    composite = max(0.0, min(100.0, round(composite, 2)))

    return AlphaResult(
        symbol=symbol,
        alpha_score=composite,
        signal=classify_signal(composite),
        layers=layer_scores,
        active_layers=active_count,
        total_layers=total_layers,
        scanned_at=datetime.now(),
    )
