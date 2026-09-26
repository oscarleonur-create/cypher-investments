"""The entry module's rules, declared so every proposal carries the version that made it.

The kinds follow the user's decisions of 2026-09-25: risk budgets, the book
limit and the two-year window are theirs (``decided``) and the learning loop
may never search them; what qualifies as a trigger and where a stop sits are
``threshold``s it may propose changing. A test fails if ``proposal``, ``zone``,
``sheet`` or ``exits`` gains a constant that is neither declared here nor listed as not
a rule.
"""

from __future__ import annotations

from advisor.learning.rules import Kind, RuleStamp, stamp

RULESET = "entry"

DECLARED: dict[str, dict[str, Kind]] = {
    "proposal": {
        "TRADE_RISK": Kind.DECIDED,
        "POSITION_RISK": Kind.DECIDED,
        "POSITION_RISK_CHEAP": Kind.DECIDED,
        "CHEAP_PERCENTILE": Kind.DECIDED,
        "MAX_TOTAL_RISK": Kind.DECIDED,
        "THESIS_BONUS": Kind.DECIDED,
        "MAX_TOTAL_RISK_THESIS": Kind.DECIDED,
        "BOOK_LIMIT": Kind.DECIDED,
        "POSITION_STOP_MIN": Kind.THRESHOLD,
        "POSITION_STOP_MAX": Kind.THRESHOLD,
        "POSITION_STOP_SIGMAS": Kind.THRESHOLD,
        "POSITION_STOP_SESSIONS": Kind.THRESHOLD,
        "TRADE_STOP_SIGMAS": Kind.THRESHOLD,
        "DIP_SIGMAS": Kind.THRESHOLD,
        "ENTRY_CONFIRM_SESSIONS": Kind.THRESHOLD,
        # No new entry within this many sessions of results (RDDT replay, 2026-09-25).
        "EARNINGS_GUARD_SESSIONS": Kind.THRESHOLD,
    },
    "zone": {
        "WINDOW_DAYS": Kind.DECIDED,  # "P/S against its own two-year median"
        "MIN_OBSERVATIONS": Kind.THRESHOLD,
        # Below MIN_OBSERVATIONS but at least this: shown, never opens a position leg.
        "SHORT_MIN_OBSERVATIONS": Kind.THRESHOLD,
        # How the P/S history is built: a day whose TTM revenue is older is dropped.
        "MAX_TTM_AGE_DAYS": Kind.MODEL,
    },
    "sheet": {
        "SIGMA_SESSIONS": Kind.MODEL,
        "THESIS_LOOKBACK_DAYS": Kind.MODEL,
        # How long an exit-grade filing keeps asking after it lands.
        "FILINGS_LOOKBACK_DAYS": Kind.MODEL,
    },
    # Exit calls on held names. The user decided (2026-09-26) that a rich P/S is
    # a REVIEW and which filings end the case; the percentile is theirs too.
    "exits": {
        "RICH_PERCENTILE": Kind.DECIDED,
        "EXIT_ITEMS": Kind.DECIDED,
        "EXIT_KINDS": Kind.DECIDED,
        "REVIEW_ITEMS": Kind.DECIDED,
        "REVIEW_KINDS": Kind.DECIDED,
        "SEVERITY": Kind.DECIDED,
    },
}

# Constants of the rule modules that decide nothing about a proposal.
NOT_RULES: dict[str, str] = {
    "proposal.NEEDS_RATIONALE": "a ledger invariant: which actions must carry reasons",
    "proposal.PARAM_CONSTANTS": "a map from EntryParams fields to the constants above",
}


def entry_rules(params=None) -> RuleStamp:
    """The version of the rules ``build_proposal`` runs with ``params`` (default: the code's)."""
    from dataclasses import asdict

    from advisor.entry import exits, proposal, sheet, zone

    modules = {"proposal": proposal, "zone": zone, "sheet": sheet, "exits": exits}
    declared = {
        f"{mod}.{name}": (getattr(modules[mod], name), kind)
        for mod, names in DECLARED.items()
        for name, kind in names.items()
    }
    if params is not None:
        for field_, value in asdict(params).items():
            key = f"proposal.{proposal.PARAM_CONSTANTS[field_]}"
            declared[key] = (value, declared[key][1])
    return stamp(RULESET, declared)
