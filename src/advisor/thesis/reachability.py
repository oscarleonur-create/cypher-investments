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
        # Per kind, then unioned: the old lookup keyed on the whole trigger's
        # kind set, so `--on FILING_RESULTS,FILING_DILUTION` missed the table
        # entirely and went unchecked. A field is reachable when *any* of the
        # triggered kinds can carry it; one unknown kind abstains for all,
        # because an absent entry means "not checked", never "invalid".
        known = [KNOWN_FIELDS.get(k) for k in kinds]
        if all(fields is not None for fields in known):
            carriers: set[str] = set().union(*known)  # type: ignore[arg-type]
            if trigger.field not in carriers:
                return Reachability(
                    claim.id,
                    False,
                    f"{', '.join(sorted(kinds))} does not carry a '{trigger.field}' field",
                )

    return Reachability(claim.id, True)


# Payload fields each event kind actually publishes. Only kinds whose payload
# is fixed *by its emitter* are listed; an absent entry means "not checked",
# never "invalid".
#
# The gap this table exists to close: NBIS carried a claim triggered on
# `FILING_RESULTS` with a `revenue_growth_yoy` field. No results payload has
# ever carried a figure — the classifier stores the form, the label and the
# URL, and nothing reads the statements — so the claim was reported as
# monitored and could never fire. Two of the six claims on that thesis were
# unreachable in exactly this way, and both were the ones that mattered.

# Every FILING_* payload is built from one literal in `news/ingest.py`.
_FILING_BASE = frozenset(
    {"form", "items", "kind", "label", "url", "accession", "accepted_at", "provider", "match"}
)

# Set only when the offering could be sized and a market cap was known.
_DILUTION_EXTRA = frozenset(
    {"offering_usd", "quote", "market_cap", "dilution_pct", "offering_pct_of_cap"}
)

_INSIDER_CLUSTER = frozenset(
    {
        "side",
        "insiders",
        "insider_count",
        "trade_count",
        "total_value",
        "net_value",
        "window_days",
        "latest_filed",
        "positions",
        # Both only when a market cap was available.
        "market_cap",
        "pct_of_cap",
    }
)

_CROSSING = frozenset(
    {"account", "instrument", "entry", "price", "threshold", "unrealized_pct", "unrealized_usd"}
)

KNOWN_FIELDS: dict[str, frozenset[str]] = {
    "FILING_DILUTION": _FILING_BASE | _DILUTION_EXTRA,
    "FILING_RESULTS": _FILING_BASE,
    "FILING_RESTATEMENT": _FILING_BASE,
    "FILING_AUDITOR_CHANGE": _FILING_BASE,
    "FILING_MERGER": _FILING_BASE,
    "FILING_DELISTING": _FILING_BASE,
    "FILING_LATE_FILING": _FILING_BASE,
    "FILING_ACTIVIST_STAKE": _FILING_BASE,
    "FILING_MANAGEMENT_CHANGE": _FILING_BASE,
    "FILING_MATERIAL_AGREEMENT": _FILING_BASE,
    "FILING_PERIODIC_REPORT": _FILING_BASE,
    "FILING_SHELF": _FILING_BASE,
    "FILING_INSIDER_TRADE": _FILING_BASE,
    "FILING_OTHER": _FILING_BASE,
    "INSIDER_SELLING_CLUSTER": _INSIDER_CLUSTER,
    "INSIDER_BUYING_CLUSTER": _INSIDER_CLUSTER,
    "RESIDUAL_DIVERGENCE": frozenset({"residual_z", "actual_return", "expected_return", "r2"}),
    "IMPLIED_EXPECTATIONS_SHIFT": frozenset(
        {"implied_cagr", "previous_implied_cagr", "change", "ev_to_revenue", "price"}
    ),
    "CONCENTRATION_WARNING": frozenset({"weight", "notional", "net_liq", "threshold"}),
    "STOP_BREACHED": _CROSSING | {"previous_pct"},
    "PROFIT_TARGET_HIT": _CROSSING | {"previous_pct"},
    "DEEP_DRAWDOWN": _CROSSING,
    "POSITION_OPENED": frozenset({"account", "instrument", "quantity", "notional", "price"}),
    "POSITION_CLOSED": frozenset({"account", "instrument", "quantity", "realized_pct"}),
    "POSITION_SIZE_CHANGED": frozenset(
        {"account", "instrument", "direction", "from_quantity", "to_quantity"}
    ),
    "DATA_QUALITY_FAILURE": frozenset({"check", "detail", "failed", "symbols"}),
}


def audit(store: DaemonStore, symbol: str, claims: list[Claim]) -> list[Reachability]:
    """Reachability for every claim on a symbol."""
    return [claim_reachability(store, symbol, c) for c in claims]
