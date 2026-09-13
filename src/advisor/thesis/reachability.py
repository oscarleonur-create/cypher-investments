"""Whether a claim's trigger can ever fire *for this symbol*.

Found by writing a real thesis. SPCX carried a claim triggered on
`RESIDUAL_DIVERGENCE`, and the CLI reported it as machine-checked. It can
never fire: residual divergence requires a factor sensitivity, SPCX has 59
sessions of price history against a 120-session floor, and so no sensitivity
exists. The claim was monitored in principle and unreachable in practice.

"Monitored" has to mean checkable *here*, or the count is a comfort rather
than a fact — and a thesis that believes it is watched when it is not is worse
than one known to be unwatched.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from advisor.daemon.store import DaemonStore
from advisor.thesis.models import Claim

logger = logging.getLogger(__name__)

# Event kinds that only exist for symbols the factor model can see.
NEEDS_SENSITIVITY = frozenset({"RESIDUAL_DIVERGENCE"})

# Event kinds that only exist for symbols with an SEC registration.
NEEDS_EDGAR = frozenset(
    {
        "FILING_DILUTION",
        "FILING_RESULTS",
        "FILING_RESTATEMENT",
        "FILING_AUDITOR_CHANGE",
        "FILING_MERGER",
        "FILING_DELISTING",
        "FILING_LATE_FILING",
        "FILING_ACTIVIST_STAKE",
        "FILING_MANAGEMENT_CHANGE",
        "FILING_MATERIAL_AGREEMENT",
        "FILING_PERIODIC_REPORT",
        "FILING_SHELF",
        "FILING_INSIDER_TRADE",
        "FILING_OTHER",
    }
)

# Kinds that need a stored valuation before they can be evaluated.
NEEDS_VALUATION = frozenset({"IMPLIED_EXPECTATIONS_SHIFT"})


@dataclass
class Reachability:
    claim_id: str | None
    reachable: bool
    reason: str = ""

    @property
    def blocked(self) -> bool:
        return not self.reachable


def claim_reachability(store: DaemonStore, symbol: str, claim: Claim) -> Reachability:
    """Can this claim's trigger produce an event for this symbol today?"""
    trigger = claim.trigger
    if not trigger.is_testable:
        return Reachability(claim.id, False, "no trigger — nothing will ever check this")

    symbol = symbol.upper()

    if trigger.factor:
        if store.load_sensitivity(symbol) is None:
            return Reachability(
                claim.id,
                False,
                f"no factor estimate for {symbol}, so its loadings cannot be compared",
            )
        return Reachability(claim.id, True)

    kinds = set(trigger.event_kinds)

    if kinds & NEEDS_SENSITIVITY and store.load_sensitivity(symbol) is None:
        return Reachability(
            claim.id,
            False,
            f"{symbol} has no factor estimate, so residual divergence can never fire "
            "— too little price history",
        )

    if kinds & NEEDS_VALUATION and store.load_latest_valuation(symbol) is None:
        return Reachability(
            claim.id, False, f"no valuation stored for {symbol} yet — run the valuation job"
        )

    # A field-and-threshold trigger on a kind that never carries that field is
    # a quieter version of the same failure.
    if trigger.field and kinds:
        known = KNOWN_FIELDS.get(frozenset(kinds))
        if known is not None and trigger.field not in known:
            return Reachability(
                claim.id,
                False,
                f"{', '.join(sorted(kinds))} does not carry a '{trigger.field}' field",
            )

    return Reachability(claim.id, True)


# Payload fields each event kind actually publishes. Only kinds whose payload
# is fixed are listed; an absent entry means "not checked", never "invalid".
KNOWN_FIELDS: dict[frozenset[str], frozenset[str]] = {
    frozenset({"FILING_DILUTION"}): frozenset({"dilution_pct", "offering_usd", "market_cap"}),
    frozenset({"RESIDUAL_DIVERGENCE"}): frozenset(
        {"residual_z", "actual_return", "expected_return", "r2"}
    ),
    frozenset({"IMPLIED_EXPECTATIONS_SHIFT"}): frozenset(
        {"implied_cagr", "previous_implied_cagr", "change", "ev_to_revenue", "price"}
    ),
}


def audit(store: DaemonStore, symbol: str, claims: list[Claim]) -> list[Reachability]:
    """Reachability for every claim on a symbol."""
    return [claim_reachability(store, symbol, c) for c in claims]
