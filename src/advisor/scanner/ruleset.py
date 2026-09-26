"""The scanner's rules, declared so every candidate carries the version that found it.

Every field of ``Thresholds`` and ``PremarketThresholds`` is taken as a
threshold automatically, so a field added there is versioned without anyone
remembering to. Module constants are declared by name below; a test fails if
a rule module gains a constant that is neither declared nor listed as not a
rule.
"""

from __future__ import annotations

from dataclasses import asdict

from advisor.learning.rules import Kind, RuleStamp, stamp
from advisor.scanner import detect, premarket

SESSION = "scanner.session"
PREMARKET = "scanner.premarket"

# Constants of the rule modules that decide nothing about what qualifies.
NOT_RULES: dict[str, str] = {
    "detect._FULL_SESSION_MINUTES": "a market fact: 390 minutes",
    "detect.DEFAULT": "the default Thresholds, declared field by field",
    "premarket.DEFAULT_WATCHLIST": "which watchlist is read, not what qualifies on it",
    "premarket.PREMARKET_OPEN": "a market fact: 04:00 ET",
    "premarket.SCAN_UNTIL": "the regular open, a market fact",
    "premarket.PM_DEFAULT": "the default PremarketThresholds, declared field by field",
    "premarket._EPS": "floating-point tolerance, not a threshold",
    "sources._EXCHANGES": "which listings the screener reads",
    "sources._PAGE": "screener page size",
    "scan.DEFAULT_NEWS_BUDGET": "a credit budget: decides whether news is looked up, not if a "
    "setup qualifies (an unchecked candidate is recorded as such)",
}


def _fields(t) -> dict:
    return {f"thresholds.{k}": (v, Kind.THRESHOLD) for k, v in asdict(t).items()}


def session_rules(t: detect.Thresholds = detect.DEFAULT) -> RuleStamp:
    """The version of the session scan (setups A and C) run with ``t``."""
    return stamp(
        SESSION,
        {
            **_fields(t),
            # The volume curve decides rvol, and so whether an A qualifies.
            "detect._VOLUME_CURVE": (detect._VOLUME_CURVE, Kind.MODEL),
            # The first scan's time is the earliest entry price a record can have.
            "detect.FIRST_SCAN": (detect.FIRST_SCAN, Kind.MODEL),
        },
    )


def premarket_rules(t: premarket.PremarketThresholds = premarket.PM_DEFAULT) -> RuleStamp:
    """The version of the premarket scan (setups A, B and C) run with ``t``."""
    return stamp(
        PREMARKET,
        {
            **_fields(t),
            "premarket.SCAN_FROM": (premarket.SCAN_FROM, Kind.MODEL),
        },
    )
