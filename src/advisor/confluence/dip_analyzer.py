"""Dip Analyzer — unified dip-buying conviction score across all signal layers.

Runs 6 layers (dip_screener, smart_money, mispricing, confluence, ml_signal,
technical_dip), normalizes each to 0-100, applies dip-specific weights and
optional regime adjustment, then returns a single DipAnalysisResult.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

from advisor.confluence.alpha_scorer import (
    _normalize_dip,
    _normalize_mispricing,
    _normalize_ml,
    _normalize_smart_money,
    _run_dip,
    _run_mispricing,
    _run_ml,
    _run_smart_money,
)
from advisor.confluence.models import (
    AlphaLayerScore,
    DipAnalysisResult,
    DipVerdict,
)

logger = logging.getLogger(__name__)

# ── Default weights (dip-specific) ──────────────────────────────────────

DIP_WEIGHTS: dict[str, float] = {
    "dip_screener": 0.25,
    "smart_money": 0.20,
    "mispricing": 0.20,
    "confluence": 0.15,
    "ml_signal": 0.10,
    "technical_dip": 0.10,
}

# ── Regime adjustments ──────────────────────────────────────────────────

_REGIME_ADJ: dict[str, float] = {
    "low_vol": 5.0,
    "normal": 0.0,
    "high_vol": -10.0,
}

# ── Verdict classification ──────────────────────────────────────────────


def _classify_dip(score: float) -> DipVerdict:
    if score >= 75:
        return DipVerdict.STRONG_BUY
    if score >= 60:
        return DipVerdict.BUY
    if score >= 45:
        return DipVerdict.LEAN_BUY
    if score >= 30:
        return DipVerdict.WATCH
    return DipVerdict.PASS


# ── Normalizers (new, dip-specific) ─────────────────────────────────────

_CONFLUENCE_SCORE_MAP = {
    "ENTER": 90.0,
    "CAUTION": 55.0,
    "PASS": 15.0,
}


def _normalize_confluence(result: Any) -> float:
    """Map ConfluenceResult verdict to 0-100."""
    return _CONFLUENCE_SCORE_MAP.get(result.verdict.value, 15.0)


def _normalize_technical_dip(result: Any) -> float:
    """Map buy_the_dip technical signal to 0-100.

    Bullish → 80.  Otherwise use RSI proxy via SMA ratio → 0-50.
    """
    if result.is_bullish:
        return 80.0
    if result.sma_20 > 0:
        ratio = result.price / result.sma_20
        # Below SMA is more interesting for dip buying: ratio < 1.0 → higher score
        # ratio 0.90 → 50, ratio 1.0 → 20, ratio 1.05+ → 0
        return max(0.0, min(50.0, (1.05 - ratio) / 0.15 * 50))
    return 0.0


# ── Layer runners (new) ────────────────────────────────────────────────


def _run_confluence_layer(symbol: str) -> Any:
    from advisor.confluence.orchestrator import run_confluence

    return run_confluence(symbol, "buy_the_dip", force_all=True)


def _run_technical_dip(symbol: str) -> Any:
    from advisor.confluence.technical import check_technical

    return check_technical(symbol, "buy_the_dip")


# ── Layer registry ──────────────────────────────────────────────────────

_LayerDef = tuple[str, Any, Any]

_DIP_LAYERS: list[_LayerDef] = [
    ("dip_screener", _run_dip, _normalize_dip),
    ("smart_money", _run_smart_money, _normalize_smart_money),
    ("mispricing", _run_mispricing, _normalize_mispricing),
    ("confluence", _run_confluence_layer, _normalize_confluence),
    ("ml_signal", _run_ml, _normalize_ml),
    ("technical_dip", _run_technical_dip, _normalize_technical_dip),
]


# ── Reasoning builder ──────────────────────────────────────────────────


def _build_dip_reasoning(
    layers: list[AlphaLayerScore],
    regime: str,
    regime_adj: float,
    dip_raw: Any | None,
) -> str:
    parts: list[str] = []

    # Safety gate
    if dip_raw is not None:
        if dip_raw.safety.passes:
            parts.append("Safety gate: PASSED")
        else:
            parts.append("Safety gate: FAILED (balance sheet risk)")
        if dip_raw.value_trap and dip_raw.value_trap.is_value:
            parts.append("Value signal confirmed")
        if dip_raw.fast_fundamentals:
            if dip_raw.fast_fundamentals.insider_buying:
                if dip_raw.fast_fundamentals.c_suite_buying:
                    parts.append("C-suite insider buying detected")
                else:
                    parts.append("Insider buying detected")
            if dip_raw.fast_fundamentals.analyst_bullish:
                upside = dip_raw.fast_fundamentals.analyst_upside_pct
                parts.append(f"Analyst upside {upside:+.0f}%")

    # Layer highlights
    layer_map = {ls.name: ls for ls in layers}
    sm = layer_map.get("smart_money")
    if sm and sm.available and sm.normalized >= 60:
        parts.append(f"Smart money bullish ({sm.normalized:.0f})")

    mp = layer_map.get("mispricing")
    if mp and mp.available and mp.normalized >= 60:
        parts.append(f"Mispricing signal ({mp.normalized:.0f})")

    cf = layer_map.get("confluence")
    if cf and cf.available:
        # Infer verdict from normalized score
        if cf.normalized >= 85:
            parts.append("Confluence: ENTER")
        elif cf.normalized >= 50:
            parts.append("Confluence: CAUTION")
        else:
            parts.append("Confluence: PASS")

    # Regime
    if regime != "unknown":
        label = {"low_vol": "Calm", "normal": "Normal", "high_vol": "Stressed"}.get(regime, regime)
        adj_str = f"{regime_adj:+.0f}" if regime_adj else "+0"
        parts.append(f"Regime: {label} ({adj_str} pts)")

    return "; ".join(parts) if parts else "Insufficient data"


# ── Core scoring function ──────────────────────────────────────────────


def analyze_dip(
    symbol: str,
    skip_layers: set[str] | None = None,
    include_regime: bool = True,
) -> DipAnalysisResult:
    """Compute a unified dip-buying score for *symbol* across all signal layers.

    Args:
        symbol: Ticker symbol.
        skip_layers: Layer names to skip entirely.
        include_regime: Whether to apply regime adjustment.

    Returns:
        DipAnalysisResult with composite score, verdict, and per-layer breakdown.
    """
    symbol = symbol.upper()
    skip = skip_layers or set()
    w = dict(DIP_WEIGHTS)

    layer_scores: list[AlphaLayerScore] = []
    dip_raw: Any = None
    price = 0.0

    for name, runner, normalizer in _DIP_LAYERS:
        if name in skip:
            layer_scores.append(AlphaLayerScore(name=name, available=False, error="skipped"))
            continue

        try:
            raw = runner(symbol)
            if raw is None:
                layer_scores.append(AlphaLayerScore(name=name, available=False, error="no data"))
                continue

            normalized = normalizer(raw)

            # Capture dip screener raw for reasoning
            if name == "dip_screener":
                dip_raw = raw

            # Capture price from technical or confluence layer
            if name == "technical_dip" and hasattr(raw, "price"):
                price = raw.price
            elif name == "confluence" and price == 0.0 and hasattr(raw, "technical"):
                price = raw.technical.price

            # Extract a numeric raw_value for display; skip string fields
            raw_val = None
            for attr in ("total_score", "score", "win_probability"):
                v = getattr(raw, attr, None)
                if isinstance(v, (int, float)):
                    raw_val = v
                    break

            layer_scores.append(
                AlphaLayerScore(
                    name=name,
                    raw_value=raw_val,
                    normalized=round(normalized, 2),
                    available=True,
                )
            )
        except Exception as e:
            logger.warning(f"Dip layer {name} failed for {symbol}: {e}")
            layer_scores.append(AlphaLayerScore(name=name, available=False, error=str(e)))

    # ── Weight redistribution ────────────────────────────────────────
    active = [ls for ls in layer_scores if ls.available]
    total_layers = len(_DIP_LAYERS)
    active_count = len(active)

    if active_count == 0:
        return DipAnalysisResult(
            symbol=symbol,
            price=price,
            dip_score=0.0,
            verdict=DipVerdict.PASS,
            layers=layer_scores,
            active_layers=0,
            total_layers=total_layers,
            reasoning="No layers available",
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

    # ── Regime adjustment ────────────────────────────────────────────
    regime = "unknown"
    regime_adj = 0.0

    if include_regime:
        try:
            from advisor.ml.regime import RegimeDetector

            if RegimeDetector.model_exists():
                det = RegimeDetector.load()
                info = det.detect_regime()
                regime = info["regime_name"]
                regime_adj = _REGIME_ADJ.get(regime, 0.0)
        except Exception as e:
            logger.warning(f"Regime detection failed: {e}")

    composite = max(0.0, min(100.0, round(composite + regime_adj, 2)))
    verdict = _classify_dip(composite)
    reasoning = _build_dip_reasoning(layer_scores, regime, regime_adj, dip_raw)

    return DipAnalysisResult(
        symbol=symbol,
        price=price,
        dip_score=composite,
        verdict=verdict,
        regime=regime,
        regime_adjustment=regime_adj,
        layers=layer_scores,
        active_layers=active_count,
        total_layers=total_layers,
        reasoning=reasoning,
        scanned_at=datetime.now(),
    )
