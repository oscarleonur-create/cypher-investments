"""Fixed-fractional contract sizing for options actions.

Salvaged from the removed pipeline orchestrator. Sits beside the risk gate:
the gate sizes share-based candidates from an entry/stop distance, this sizes
contract-based ones from max-loss and buying-power per contract.
"""

from __future__ import annotations

import math

from pydantic import BaseModel, Field


class ContractSizingConfig(BaseModel):
    """Account-level budget for a contract-based position."""

    account_size: float = Field(default=5_000.0, description="Total account value in dollars")
    max_bp_pct: float = Field(default=50.0, description="Max buying power usage as % of account")
    max_risk_pct: float = Field(default=5.0, description="Max loss per trade as % of account")


class SizingResult(BaseModel):
    """Outcome of a contract sizing decision."""

    suggested_contracts: int = 0
    position_bp: float = 0.0  # total BP used
    bp_utilization_pct: float = 0.0  # position_bp / available_bp * 100
    max_loss_total: float = 0.0  # max_loss_per_contract * contracts
    risk_pct: float = 0.0  # actual % of account at risk
    sizing_feasible: bool = False
    sizing_score: float = 0.0  # 0-100


def compute_position_size(
    max_loss_per_contract: float,
    buying_power_per_contract: float,
    config: ContractSizingConfig,
) -> SizingResult:
    """Compute position size using fixed fractional sizing.

    Rules:
        1. max_position_loss = account_size * max_risk_pct / 100
        2. contracts = floor(max_position_loss / max_loss_per_contract)
        3. Cap by available BP: account_size * max_bp_pct / 100
        4. Ensure at least 1 contract if affordable

    Scoring: feasibility base (50) + BP efficiency (0-50, lower utilization = higher).
    """
    available_bp = config.account_size * config.max_bp_pct / 100
    max_position_loss = config.account_size * config.max_risk_pct / 100

    if max_loss_per_contract <= 0 or buying_power_per_contract <= 0:
        return SizingResult(sizing_feasible=False, sizing_score=0.0)

    # Fixed fractional: how many contracts fit within risk budget
    contracts_by_risk = math.floor(max_position_loss / max_loss_per_contract)

    # BP cap: how many contracts fit within BP budget
    contracts_by_bp = math.floor(available_bp / buying_power_per_contract)

    contracts = min(contracts_by_risk, contracts_by_bp)

    # Ensure at least 1 if we can afford it
    if contracts < 1:
        if max_loss_per_contract <= max_position_loss and buying_power_per_contract <= available_bp:
            contracts = 1
        else:
            return SizingResult(
                suggested_contracts=0,
                sizing_feasible=False,
                sizing_score=0.0,
            )

    position_bp = contracts * buying_power_per_contract
    bp_utilization_pct = (position_bp / available_bp * 100) if available_bp > 0 else 100.0
    max_loss_total = contracts * max_loss_per_contract
    risk_pct = (max_loss_total / config.account_size * 100) if config.account_size > 0 else 100.0

    # Scoring: base 50 for feasible + up to 50 for BP efficiency
    # Lower utilization → higher score (more room for other trades)
    bp_efficiency_score = max(0.0, 50.0 * (1.0 - bp_utilization_pct / 100.0))
    sizing_score = min(100.0, 50.0 + bp_efficiency_score)

    return SizingResult(
        suggested_contracts=contracts,
        position_bp=round(position_bp, 2),
        bp_utilization_pct=round(bp_utilization_pct, 2),
        max_loss_total=round(max_loss_total, 2),
        risk_pct=round(risk_pct, 2),
        sizing_feasible=True,
        sizing_score=round(sizing_score, 2),
    )
