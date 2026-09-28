"""The breadth layer's rules, declared so every universe snapshot carries the version that cut it.

The universe floor is the user's scope decision (``docs/breadth-plan.md``,
accepted 2026-09-27), not something the learning loop may search: widening
it would change which names can be found, not how well a rule works on them.
What history a name needs is a property of the signals that read it.

A test fails if ``universe``, ``listings``, ``bars`` or ``facts`` gains a
constant that is neither declared here nor listed as not a rule.
"""

from __future__ import annotations

from advisor.learning.rules import Kind, RuleStamp, stamp

RULESET = "breadth.universe"

DECLARED: dict[str, dict[str, Kind]] = {
    "universe": {
        "MIN_PRICE": Kind.DECIDED,
        "MIN_DOLLAR_VOLUME": Kind.DECIDED,
        # A year of sessions: the 52-week and 12-month families need it.
        "MIN_SESSIONS": Kind.MODEL,
        "ADV_WINDOW": Kind.MODEL,
        "STALE_BAR_DAYS": Kind.MODEL,
    },
    "listings": {
        # Which listings count as common stock decides who can be found.
        "_NOT_COMMON": Kind.MODEL,
        "_SPAC": Kind.MODEL,
        "_DELINQUENT": Kind.MODEL,
        "_BANKRUPT": Kind.MODEL,
    },
}

NOT_RULES: dict[str, str] = {
    "listings.NASDAQ_URL": "a source address",
    "listings.OTHER_URL": "a source address",
    "listings.SEC_TICKERS_URL": "a source address",
    "listings._OTHER_EXCHANGES": "exchange-code labels",
    "bars.BACKFILL_DAYS": "how much history is stored, not what qualifies",
    "bars.OVERLAP_DAYS": "how far each pull re-reads, a data-integrity margin",
    "bars.CHUNK": "request size",
    "bars.PAUSE_SECONDS": "request pacing",
    "bars.RATE_LIMIT_MIN_CHUNK": "when an empty reply is read as a refusal",
    "bars.RATE_LIMIT_BACKOFF_SECONDS": "request pacing",
    "bars.REBASE_TOLERANCE": "when stored bars are refetched after a split",
    "bars.EMPTY_RETRY_DAYS": "how often a symbol with no data is asked again",
    "bars.SETTLE_MINUTES": "when a session's bar is final; a data-integrity margin",
    "facts.FIRST_YEAR": "how much history is stored",
    "facts.SETTLED_AFTER_DAYS": "when a frame stops being re-read",
    "facts.STALE_TTM_DAYS": "when a company's own series is read to fill a gap",
    "facts.CONCEPT_REFRESH_DAYS": "how often that series is re-read",
    "facts._UNITS": "the SEC's unit names",
    "facts.CONCEPTS": "which SEC concepts are stored",
}


def universe_rules() -> RuleStamp:
    """The version of the E0 cut this code runs."""
    from advisor.breadth import listings, universe

    modules = {"universe": universe, "listings": listings}
    declared = {}
    for mod, names in DECLARED.items():
        for name, kind in names.items():
            value = getattr(modules[mod], name)
            if mod == "listings" and name in ("_NOT_COMMON",):
                value = [(label, p.pattern) for label, p in value]
            elif hasattr(value, "pattern"):
                value = value.pattern
            declared[f"{mod}.{name}"] = (value, kind)
    return stamp(RULESET, declared)
