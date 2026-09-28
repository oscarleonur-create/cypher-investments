"""The breadth layer's rules, declared so every universe snapshot carries the version that cut it.

The universe floor is the user's scope decision (``docs/breadth-plan.md``,
accepted 2026-09-27), not something the learning loop may search: widening
it would change which names can be found, not how well a rule works on them.
What history a name needs is a property of the signals that read it.

A test fails if ``universe``, ``listings``, ``bars``, ``facts``, ``signals``,
``measure`` or ``companies`` gains a constant that is neither declared here
nor listed as not a rule.

The signal families (B1) have their own ruleset, ``breadth.signals``. Every
field of ``signals.Thresholds`` is a threshold the learning loop may propose
changing; the windows that define what a family measures are ``model``; the
universe floor is folded in, because it decides which names can signal.
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
    "signals.DEFAULT": "the default Thresholds, declared field by field",
    "signals.GROUPS": "the names of the record groups",
    "measure.LIVE_HISTORY_DAYS": "how much history a live run loads, enough for every window",
    "measure.CAVEATS": "report text",
    "filings.INDEX_URL": "a source address",
    "filings.FIRST_YEAR": "how much history is stored",
    "filings.PERIODIC_FORMS": "which forms are stored: the union of the two declared sets",
    "filings._SCHEMA": "table definitions",
}

SIGNALS = "breadth.signals"

SIGNAL_DECLARED: dict[str, dict[str, Kind]] = {
    "signals": {
        "MOMENTUM_LOOKBACK": Kind.MODEL,  # 12-1 momentum: a year...
        "MOMENTUM_SKIP": Kind.MODEL,  # ...skipping the last month
        "MIN_CROSS_SECTION": Kind.MODEL,
        "HIGH_WINDOW": Kind.MODEL,
        "SIGMA_WINDOW": Kind.MODEL,
        "VOLUME_WINDOW": Kind.MODEL,
        "BASE_WINDOW": Kind.MODEL,
        # How close in time two families must be to count as agreeing.
        "CONVERGE_SESSIONS": Kind.THRESHOLD,
        "COOLDOWN_SESSIONS": Kind.MODEL,
    },
    "measure": {
        "HORIZONS": Kind.MODEL,
        "TAILS": Kind.MODEL,
        "CONTROLS": Kind.MODEL,
        "MIN_POOL": Kind.MODEL,
        "SIZE_BUCKETS": Kind.MODEL,
        "FILL_LIMIT": Kind.MODEL,
    },
    "companies": {"_DIVISIONS": Kind.MODEL},
    # When a quarter's revenue counts as public decides when F can fire.
    "filings": {
        "QUARTERLY_FORMS": Kind.MODEL,
        "ANNUAL_FORMS": Kind.MODEL,
        "MAX_FILING_LAG_DAYS": Kind.MODEL,
    },
}


def signal_rules(t=None) -> RuleStamp:
    """The version of the families and their measurement, universe floor included."""
    from dataclasses import asdict

    from advisor.breadth import companies, filings, measure, signals

    t = t or signals.DEFAULT
    modules = {"signals": signals, "measure": measure, "companies": companies, "filings": filings}
    declared = {f"thresholds.{k}": (v, Kind.THRESHOLD) for k, v in asdict(t).items()}
    for mod, names in SIGNAL_DECLARED.items():
        for name, kind in names.items():
            declared[f"{mod}.{name}"] = (getattr(modules[mod], name), kind)
    universe = universe_rules()
    for name, value in universe.params.items():
        declared[name] = (value, universe.kinds[name])
    return stamp(SIGNALS, declared)


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
