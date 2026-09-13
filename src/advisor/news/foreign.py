"""Reading a 6-K, where the substance is never in the filing itself.

A foreign private issuer files no 8-K and no 10-Q. Everything it would
otherwise disclose — results, offerings, board changes, partnerships — arrives
as a 6-K, and a 6-K has no item codes. Worse, its body is a cover page: about
1,500 characters of SEC boilerplate that says only "a report is attached".
The disclosure lives in an EX-99 exhibit.

The cost of not reading it, measured on the live book: Nebius issued
**$5.75bn of convertible senior notes** across three August filings — 9.4% of
its market cap, on a position worth 14.4% of the portfolio — and every one was
archived as Tier C "other events", because the cover page genuinely says
nothing and nothing looked further.

Classification here is by the press-release headline, which is the closest a
6-K has to an item code. It is keyword matching and it is stated as such: an
unrecognised headline stays OTHER rather than being forced into a category.
"""

from __future__ import annotations

import logging
import re

from advisor.news.classify import Classification, FilingKind, Materiality

logger = logging.getLogger(__name__)

# Only the opening of the exhibit: a press release states its subject in the
# headline, and the body cites everything else the company has ever done.
HEADLINE_CHARS = 400

# Ordered by specificity. An offering announcement also mentions the quarter
# it closed in, so the offering patterns are tried first.
_PATTERNS: tuple[tuple[re.Pattern[str], Classification], ...] = (
    (
        re.compile(r"convertible (senior )?notes|convertible bond", re.IGNORECASE),
        Classification(FilingKind.DILUTION, Materiality.HIGH, "convertible notes offering"),
    ),
    (
        re.compile(
            r"(private |public |underwritten )?offering of .{0,40}(shares|ordinary|common)|"
            r"share (issuance|placement)|equity (offering|raise)",
            re.IGNORECASE,
        ),
        Classification(FilingKind.DILUTION, Materiality.HIGH, "share offering"),
    ),
    (
        re.compile(
            r"offering of .{0,30}(senior )?notes|notes offering|bond offering", re.IGNORECASE
        ),
        Classification(FilingKind.DEBT_ISSUANCE, Materiality.MEDIUM, "debt offering"),
    ),
    (
        re.compile(
            r"(reports|announces|publishes).{0,30}(financial results|results for|"
            r"quarter|full[- ]year|interim results)|earnings release",
            re.IGNORECASE,
        ),
        Classification(FilingKind.RESULTS, Materiality.HIGH, "reported results"),
    ),
    (
        re.compile(
            r"appoint(s|ment|ed)|steps down|resign(s|ation)|new (chief|ceo|cfo)|"
            r"chief (executive|financial) officer",
            re.IGNORECASE,
        ),
        Classification(FilingKind.MANAGEMENT_CHANGE, Materiality.MEDIUM, "management change"),
    ),
    (
        re.compile(r"acqui(re|res|sition)|merger|to be acquired|takeover", re.IGNORECASE),
        Classification(FilingKind.MERGER, Materiality.HIGH, "acquisition or merger"),
    ),
    (
        re.compile(r"partner(s|ship)?\b|collaborat|strategic agreement", re.IGNORECASE),
        Classification(FilingKind.MATERIAL_AGREEMENT, Materiality.MEDIUM, "partnership announced"),
    ),
    (
        re.compile(r"annual general meeting|extraordinary general meeting|agm\b", re.IGNORECASE),
        Classification(FilingKind.OTHER, Materiality.LOW, "shareholder meeting"),
    ),
    (
        re.compile(r"delisting|deregistration|notice of non[- ]compliance", re.IGNORECASE),
        Classification(FilingKind.DELISTING, Materiality.HIGH, "listing status"),
    ),
)

UNRECOGNISED = Classification(FilingKind.OTHER, Materiality.LOW, "foreign issuer report")


# A deal is announced, then priced, then closed — three filings for one
# transaction. Nebius's convertible ran 4.50bn proposed, 5.0bn priced, 5.75bn
# closed, and each raised its own interrupt. The proposal is an intention and
# is demoted; the pricing and the closing are facts.
_PROPOSED = re.compile(
    r"\bproposed\b|\bintends to (offer|issue)|\bplans to (offer|issue)", re.IGNORECASE
)


def is_proposal(text: str, *, headline_chars: int = HEADLINE_CHARS) -> bool:
    """Whether a headline announces an intention rather than a completed deal."""
    return bool(_PROPOSED.search(" ".join((text or "").split())[:headline_chars]))


def classify_headline(text: str, *, headline_chars: int = HEADLINE_CHARS) -> Classification:
    """Classify a 6-K from its press-release headline.

    Returns OTHER for anything unrecognised rather than guessing. A 6-K that
    cannot be read is still archived and still visible; it simply does not
    interrupt.
    """
    headline = " ".join((text or "").split())[:headline_chars]
    if not headline:
        return UNRECOGNISED
    for pattern, classification in _PATTERNS:
        if pattern.search(headline):
            return classification
    logger.info("foreign: unrecognised 6-K headline: %s", headline[:110])
    return UNRECOGNISED


def exhibit_text(filing, *, max_chars: int = 20000) -> str:
    """Text of a 6-K's first EX-99 exhibit, or an empty string.

    EX-99 is where a foreign issuer attaches its press release. The other
    exhibit types on a 6-K are indentures, legal opinions and images.
    """
    try:
        attachments = list(filing.attachments)
    except Exception as exc:  # noqa: BLE001
        logger.debug("foreign: no attachments on %s: %s", filing.accession_no, exc)
        return ""

    for attachment in attachments:
        if not str(getattr(attachment, "document_type", "")).upper().startswith("EX-99"):
            continue
        try:
            return " ".join(attachment.text().split())[:max_chars]
        except Exception as exc:  # noqa: BLE001
            logger.debug("foreign: could not read exhibit: %s", exc)
    return ""
