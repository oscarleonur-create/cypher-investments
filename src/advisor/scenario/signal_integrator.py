"""Signal integrator — adjusts scenario probabilities using alpha/confluence signals."""

from __future__ import annotations

import logging

from advisor.scenario.models import ScenarioDefinition, SignalContext

logger = logging.getLogger(__name__)


def fetch_signal_context(symbol: str) -> SignalContext:
    """Fetch current alpha score and confluence verdict for a symbol.

    Uses compute_alpha() which is fast (no API calls required).
    Confluence is optional and more expensive.
    """
    from advisor.confluence.alpha_scorer import compute_alpha

    try:
        alpha = compute_alpha(symbol)
        return SignalContext(
            alpha_score=alpha.alpha_score,
            alpha_signal=alpha.signal.value,
        )
    except Exception as e:
        logger.warning("Alpha scoring failed for %s: %s", symbol, e)
        return SignalContext()


def fetch_full_signal_context(symbol: str) -> SignalContext:
    """Fetch alpha score plus confluence verdict.

    More expensive — calls the full confluence pipeline which may use APIs.
    """
    ctx = fetch_signal_context(symbol)

    try:
        from advisor.confluence.orchestrator import run_confluence

        result = run_confluence(symbol)
        ctx.confluence_verdict = result.verdict.value
    except Exception as e:
        logger.warning("Confluence check failed for %s: %s", symbol, e)

    return ctx


def adjust_scenario_weights(
    scenarios: list[ScenarioDefinition],
    signal_context: SignalContext,
) -> dict[str, float]:
    """Adjust scenario probabilities based on signal context.

    Returns dict mapping scenario name to adjusted probability (sums to 1.0).
    """
    weights = {s.name: s.base_probability for s in scenarios}

    signal = signal_context.alpha_signal
    if not signal:
        return weights

    # Define adjustments based on signal strength
    if signal in ("STRONG_BUY", "BUY"):
        adjustments = {"bull": 0.15, "sideways": 0.0, "bear": -0.10, "crash": -0.05}
    elif signal == "LEAN_BUY":
        adjustments = {"bull": 0.08, "sideways": 0.02, "bear": -0.07, "crash": -0.03}
    elif signal == "LEAN_SELL":
        adjustments = {"bull": -0.08, "sideways": 0.0, "bear": 0.05, "crash": 0.03}
    elif signal in ("AVOID",):
        adjustments = {"bull": -0.15, "sideways": 0.0, "bear": 0.10, "crash": 0.05}
    else:
        # NEUTRAL — no adjustment
        return weights

    # Apply confluence override if available
    verdict = signal_context.confluence_verdict
    if verdict == "ENTER":
        adjustments = {
            k: v + (0.05 if k == "bull" else -0.02 if k in ("bear", "crash") else 0.0)
            for k, v in adjustments.items()
        }
    elif verdict == "PASS":
        adjustments = {
            k: v
            + (-0.05 if k == "bull" else 0.03 if k == "bear" else 0.02 if k == "crash" else 0.0)
            for k, v in adjustments.items()
        }

    # Apply adjustments and renormalize
    for name in list(weights.keys()):
        adj = adjustments.get(name, 0.0)
        weights[name] = max(weights[name] + adj, 0.01)  # Floor at 1%

    # Renormalize
    total = sum(weights.values())
    weights = {k: v / total for k, v in weights.items()}

    return weights
