"""The entry module's rules, declared so every proposal carries the version that made it.

The kinds follow the user's decisions of 2026-09-25: risk budgets, the book
limit and the two-year window are theirs (``decided``) and the learning loop
may never search them; what qualifies as a trigger and where a stop sits are
``threshold``s it may propose changing. A test fails if ``proposal``, ``zone``
or ``sheet`` gains a constant that is neither declared here nor listed as not
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
    },
}

# Constants of the rule modules that decide nothing about a proposal.
NOT_RULES: dict[str, str] = {}


def entry_rules() -> RuleStamp:
    """The version of the rules ``build_proposal`` runs now."""
    from advisor.entry import proposal, sheet, zone

    modules = {"proposal": proposal, "zone": zone, "sheet": sheet}
    return stamp(
        RULESET,
        {
            f"{mod}.{name}": (getattr(modules[mod], name), kind)
            for mod, names in DECLARED.items()
            for name, kind in names.items()
        },
    )
