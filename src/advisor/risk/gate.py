"""Position sizing and go/no-go checks for trade candidates.

A trader must never act on a raw candidate — only on one that has passed
through `gate_signals` with `risk_approved=True`. This module has no
execution path of its own; it only sizes and vetoes.
"""

from __future__ import annotations

from pydantic import BaseModel

from advisor.risk.models import TradeCandidate


class RiskLimits(BaseModel):
    """Tunable risk budget. Conservative starting points, not backtested."""

    max_risk_pct: float = 0.005  # % of net liq risked per trade (entry→stop distance)
    max_symbol_exposure_pct: float = 0.10  # % of net liq in one symbol (existing + new)
    max_position_notional_pct: float = 0.20  # % of net liq in one position
    min_risk_reward: float = 1.5


def assess_signal(
    signal: TradeCandidate,
    *,
    net_liq: float,
    existing_symbol_notional: float = 0.0,
    limits: RiskLimits | None = None,
) -> TradeCandidate:
    """Return a copy of `signal` annotated with a sizing/approval verdict."""
    limits = limits or RiskLimits()

    if net_liq <= 0:
        return signal.model_copy(
            update={"risk_approved": False, "risk_quantity": 0, "risk_note": "no account data"}
        )

    risk_per_share = abs(signal.entry - signal.stop)
    if risk_per_share <= 0 or signal.entry <= 0:
        return signal.model_copy(
            update={"risk_approved": False, "risk_quantity": 0, "risk_note": "invalid entry/stop"}
        )

    reasons: list[str] = []
    quantity = int((net_liq * limits.max_risk_pct) // risk_per_share)

    notional_cap = int((net_liq * limits.max_position_notional_pct) // signal.entry)
    if notional_cap < quantity:
        quantity = notional_cap
        reasons.append("capped by max position notional")

    exposure_room = max(0.0, net_liq * limits.max_symbol_exposure_pct - existing_symbol_notional)
    exposure_cap = int(exposure_room // signal.entry)
    if exposure_cap < quantity:
        quantity = exposure_cap
        reasons.append("capped by max symbol exposure")

    quantity = max(quantity, 0)
    if quantity < 1:
        reasons.append("risk budget rounds to 0 shares")
    if signal.risk_reward < limits.min_risk_reward:
        reasons.append(f"R:R {signal.risk_reward} < {limits.min_risk_reward}")

    approved = quantity >= 1 and signal.risk_reward >= limits.min_risk_reward
    return signal.model_copy(
        update={
            "risk_approved": approved,
            "risk_quantity": quantity,
            "risk_note": "; ".join(reasons),
        }
    )


def gate_signals(
    signals: list[TradeCandidate],
    *,
    net_liq: float,
    open_notional_by_symbol: dict[str, float] | None = None,
    limits: RiskLimits | None = None,
) -> list[TradeCandidate]:
    """Gate a batch of signals; returns them approved-first."""
    exposure = open_notional_by_symbol or {}
    gated = [
        assess_signal(
            s,
            net_liq=net_liq,
            existing_symbol_notional=exposure.get(s.symbol, 0.0),
            limits=limits,
        )
        for s in signals
    ]
    gated.sort(key=lambda s: (not s.risk_approved, -s.score))
    return gated
