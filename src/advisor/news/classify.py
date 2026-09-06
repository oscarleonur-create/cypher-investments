"""What a filing *is*, from the regulator's own taxonomy.

No model, no sentiment score. The SEC already publishes a classification
system and companies are legally obliged to use it correctly: the form type
says what kind of document this is, and an 8-K's item numbers say which
category of material event triggered it.

That makes this layer fully auditable — every classification traces to a code
in the filing itself, and a wrong answer here is a mapping bug, not a
hallucination. Direction and magnitude are deliberately *not* inferred: a
$600M offering is DILUTION with a size you can compute against market cap,
which is a fact. Whether that is "bearish" is a judgment for the thesis layer,
with the thesis in hand.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum


class Materiality(StrEnum):
    """How hard this kind of filing usually hits, absent other information."""

    HIGH = "HIGH"  # routinely moves the stock; worth an interrupt
    MEDIUM = "MEDIUM"  # worth reading today
    LOW = "LOW"  # housekeeping; log it


class FilingKind(StrEnum):
    DILUTION = "DILUTION"  # new shares offered
    SHELF = "SHELF"  # capacity registered, not yet used
    RESULTS = "RESULTS"  # earnings / operating results
    GUIDANCE = "GUIDANCE"
    MANAGEMENT_CHANGE = "MANAGEMENT_CHANGE"
    MATERIAL_AGREEMENT = "MATERIAL_AGREEMENT"
    AGREEMENT_TERMINATED = "AGREEMENT_TERMINATED"
    IMPAIRMENT = "IMPAIRMENT"
    RESTATEMENT = "RESTATEMENT"  # previously issued financials cannot be relied upon
    AUDITOR_CHANGE = "AUDITOR_CHANGE"
    BANKRUPTCY = "BANKRUPTCY"
    DELISTING = "DELISTING"
    ACTIVIST_STAKE = "ACTIVIST_STAKE"
    PASSIVE_STAKE = "PASSIVE_STAKE"
    INSIDER_TRADE = "INSIDER_TRADE"
    MERGER = "MERGER"
    LATE_FILING = "LATE_FILING"  # a company that cannot file on time
    PERIODIC_REPORT = "PERIODIC_REPORT"
    OTHER = "OTHER"


@dataclass(frozen=True)
class Classification:
    kind: FilingKind
    materiality: Materiality
    label: str


# ── 8-K item codes ───────────────────────────────────────────────────────
#
# The full list is in SEC Form 8-K. Only the items that change a position
# thesis are given their own kind; the rest fall through to OTHER.
ITEM_MAP: dict[str, Classification] = {
    "1.01": Classification(
        FilingKind.MATERIAL_AGREEMENT, Materiality.MEDIUM, "entered a material definitive agreement"
    ),
    "1.02": Classification(
        FilingKind.AGREEMENT_TERMINATED, Materiality.MEDIUM, "terminated a material agreement"
    ),
    "1.03": Classification(FilingKind.BANKRUPTCY, Materiality.HIGH, "bankruptcy or receivership"),
    "2.01": Classification(
        FilingKind.MERGER, Materiality.HIGH, "completed an acquisition or disposition"
    ),
    "2.02": Classification(FilingKind.RESULTS, Materiality.HIGH, "reported results of operations"),
    "2.03": Classification(
        FilingKind.MATERIAL_AGREEMENT, Materiality.MEDIUM, "created a direct financial obligation"
    ),
    "2.04": Classification(
        FilingKind.MATERIAL_AGREEMENT,
        Materiality.HIGH,
        "triggered an acceleration of an obligation",
    ),
    "2.05": Classification(
        FilingKind.IMPAIRMENT, Materiality.MEDIUM, "committed to exit or disposal costs"
    ),
    "2.06": Classification(
        FilingKind.IMPAIRMENT, Materiality.HIGH, "recorded a material impairment"
    ),
    "3.01": Classification(
        FilingKind.DELISTING, Materiality.HIGH, "delisting notice or listing rule failure"
    ),
    "3.02": Classification(
        FilingKind.DILUTION, Materiality.HIGH, "unregistered sale of equity securities"
    ),
    "4.01": Classification(
        FilingKind.AUDITOR_CHANGE, Materiality.HIGH, "changed certifying accountant"
    ),
    "4.02": Classification(
        FilingKind.RESTATEMENT,
        Materiality.HIGH,
        "previously issued financials cannot be relied upon",
    ),
    "5.01": Classification(FilingKind.MERGER, Materiality.HIGH, "change in control"),
    "5.02": Classification(
        FilingKind.MANAGEMENT_CHANGE, Materiality.MEDIUM, "director or principal officer change"
    ),
    "7.01": Classification(FilingKind.OTHER, Materiality.LOW, "regulation FD disclosure"),
    "8.01": Classification(FilingKind.OTHER, Materiality.LOW, "other events"),
    "9.01": Classification(FilingKind.OTHER, Materiality.LOW, "financial statements and exhibits"),
}

# ── Form types ───────────────────────────────────────────────────────────
FORM_MAP: dict[str, Classification] = {
    "424B1": Classification(FilingKind.DILUTION, Materiality.HIGH, "securities offering"),
    "424B2": Classification(FilingKind.DILUTION, Materiality.HIGH, "securities offering"),
    "424B3": Classification(FilingKind.DILUTION, Materiality.HIGH, "securities offering"),
    "424B4": Classification(FilingKind.DILUTION, Materiality.HIGH, "securities offering"),
    "424B5": Classification(
        FilingKind.DILUTION, Materiality.HIGH, "prospectus supplement — shares offered"
    ),
    "S-1": Classification(FilingKind.SHELF, Materiality.MEDIUM, "registration statement"),
    "S-3": Classification(
        FilingKind.SHELF, Materiality.MEDIUM, "shelf registration — capacity, not yet issuance"
    ),
    "S-3ASR": Classification(FilingKind.SHELF, Materiality.MEDIUM, "automatic shelf registration"),
    "SC 13D": Classification(
        FilingKind.ACTIVIST_STAKE, Materiality.HIGH, "activist stake — intent to influence"
    ),
    "SC 13G": Classification(FilingKind.PASSIVE_STAKE, Materiality.LOW, "passive stake"),
    "4": Classification(FilingKind.INSIDER_TRADE, Materiality.LOW, "insider transaction"),
    "10-K": Classification(FilingKind.PERIODIC_REPORT, Materiality.MEDIUM, "annual report"),
    "10-Q": Classification(FilingKind.PERIODIC_REPORT, Materiality.MEDIUM, "quarterly report"),
    "NT 10-K": Classification(
        FilingKind.LATE_FILING, Materiality.HIGH, "annual report will be late"
    ),
    "NT 10-Q": Classification(
        FilingKind.LATE_FILING, Materiality.HIGH, "quarterly report will be late"
    ),
    # A bare 25-NSE is not assumed to threaten the common stock — see
    # `classify_delisting`, which reads the class of securities being removed.
    "25-NSE": Classification(FilingKind.DELISTING, Materiality.MEDIUM, "exchange delisting notice"),
}

_MATERIALITY_ORDER = {Materiality.HIGH: 0, Materiality.MEDIUM: 1, Materiality.LOW: 2}

UNKNOWN = Classification(FilingKind.OTHER, Materiality.LOW, "unclassified filing")


def classify_items(item_codes: list[str]) -> list[Classification]:
    """Classify each 8-K item, dropping ones with no mapping."""
    out = []
    for raw in item_codes:
        code = raw.replace("Item", "").strip()
        found = ITEM_MAP.get(code)
        if found:
            out.append(found)
    return out


def classify_filing(form: str, item_codes: list[str] | None = None) -> Classification:
    """The single most material reading of a filing.

    An 8-K is classified by its *items*, not by being an 8-K — "8-K filed" is
    not information. When several items are present the most material wins,
    which is why an offering buried alongside two housekeeping items still
    surfaces.
    """
    candidates = classify_items(item_codes or [])
    form_class = FORM_MAP.get(form.upper().strip())
    if form_class and not (form.upper().startswith("8-K") and candidates):
        candidates.append(form_class)
    if not candidates:
        return UNKNOWN
    return sorted(candidates, key=lambda c: _MATERIALITY_ORDER[c.materiality])[0]


# Classes of security whose removal does not touch a common-stock position.
# T1 Energy's 25-NSE struck its $11.50 warrants while the common kept trading;
# reporting that as a Tier A DELISTING would have been a false alarm on a
# position that was never at risk.
_NON_EQUITY_CLASSES = ("warrant", "unit", "right", "preferred", "debenture")

# A derivative names the common stock it settles into, so simply looking for
# "common stock" says yes to a warrant. T1 Energy's delisted class reads
# "Warrants, each whole warrant exercisable to purchase one Common Stock at an
# exercise price of $11.50" — the common stock appears only as the thing the
# warrant converts *into*. These subordinate clauses are removed before the
# text is asked whether the common stock is itself being struck.
_DERIVATIVE_CLAUSE = re.compile(
    r"(to purchase|exercisable (for|to purchase)|convertible into|underlying|"
    r"issuable upon (exercise|conversion)|in exchange for)\s+[^.;]{0,80}?"
    r"common (stock|share)s?",
    re.IGNORECASE,
)


def classify_delisting(class_description: str | None) -> Classification:
    """Grade a 25-NSE by what is actually being removed from the exchange.

    Fails *safe* rather than closed: an unreadable or ambiguous class
    description keeps the filing at MEDIUM, because "something was delisted
    and I could not tell what" deserves a look but not an interrupt.
    """
    if not class_description:
        return FORM_MAP["25-NSE"]

    text = class_description.lower()
    mentions_other = any(word in text for word in _NON_EQUITY_CLASSES)
    # Strip the "...to purchase one common stock..." style references first.
    standalone = _DERIVATIVE_CLAUSE.sub(" ", text)
    mentions_common = "common stock" in standalone or "common share" in standalone

    if mentions_common and not mentions_other:
        return Classification(
            FilingKind.DELISTING, Materiality.HIGH, "the common stock is being delisted"
        )
    if mentions_other and not mentions_common:
        return Classification(
            FilingKind.OTHER,
            Materiality.LOW,
            "a non-equity class was delisted; the common stock is unaffected",
        )
    # Both, or neither — do not guess at a Tier A interrupt.
    return FORM_MAP["25-NSE"]
