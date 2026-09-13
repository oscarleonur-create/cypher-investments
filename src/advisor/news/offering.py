"""What a prospectus is actually offering, and whether the deal is done.

Two errors an end-to-end run against AMD exposed, both of which made the
system say something false about a real filing:

**A 424B5 is not necessarily dilution.** AMD's August supplement offered
$4.75bn of 4.600% Senior Notes due 2029 and three further tranches — debt.
Not one share was issued. Classifying every 424B as DILUTION reported a
leverage decision as shareholder dilution, which is a different fact with a
different consequence.

**A preliminary supplement is not a completed offering.** AMD filed the
preliminary on 13 August with no amount, because the deal had not priced, and
the final on the 14th carrying $4,750,000,000. The system raised a Tier A
interrupt on the unpriced preliminary and quietly filed the priced final as a
digest item — exactly backwards, because the "an unsized offering still
interrupts" rule fired on the half that could not be sized by construction.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from enum import StrEnum

logger = logging.getLogger(__name__)


class SecurityType(StrEnum):
    EQUITY = "EQUITY"  # shares — genuinely dilutive
    DEBT = "DEBT"  # notes, bonds, debentures — leverage, not dilution
    CONVERTIBLE = "CONVERTIBLE"  # debt that becomes equity; dilutive if converted
    UNKNOWN = "UNKNOWN"


# Ordered: a convertible names itself, and must be recognised before the
# generic debt patterns claim it.
_CONVERTIBLE = re.compile(
    r"convertible (senior )?(notes|debentures|bonds|preferred)", re.IGNORECASE
)
_DEBT = re.compile(
    r"[\d.]+\s*%\s+(senior |subordinated |secured |unsecured )*notes due|"
    r"\bnotes due \d{4}|\bdebentures\b|\bbonds due \d{4}|"
    r"aggregate principal amount of.{0,40}notes",
    re.IGNORECASE,
)
_EQUITY = re.compile(
    r"shares of (our |the company'?s )?common (stock|shares)|"
    r"\bcommon stock,? (par value|no par)|depositary shares|"
    r"american depositary shares",
    re.IGNORECASE,
)
_PRELIMINARY = re.compile(
    r"information in this preliminary prospectus supplement is not complete|"
    r"subject to completion",
    re.IGNORECASE,
)

# Only the front of the document is read. A debt prospectus discusses common
# stock in its risk factors and a share prospectus mentions the notes it will
# repay; the cover page is where the offering is named.
COVER_CHARS = 4000


@dataclass(frozen=True)
class OfferingShape:
    security: SecurityType
    preliminary: bool

    @property
    def dilutive(self) -> bool:
        """Whether shareholders are diluted if this deal completes."""
        return self.security in (SecurityType.EQUITY, SecurityType.CONVERTIBLE)


def classify_offering(text: str, *, cover_chars: int = COVER_CHARS) -> OfferingShape:
    """Read the cover page for what is being sold and whether it is priced."""
    cover = " ".join((text or "").split())[:cover_chars]
    preliminary = bool(_PRELIMINARY.search(cover))

    if _CONVERTIBLE.search(cover):
        return OfferingShape(SecurityType.CONVERTIBLE, preliminary)

    debt, equity = _DEBT.search(cover), _EQUITY.search(cover)
    if debt and not equity:
        return OfferingShape(SecurityType.DEBT, preliminary)
    if equity and not debt:
        return OfferingShape(SecurityType.EQUITY, preliminary)
    if debt and equity:
        # Both named on the cover: whichever appears first is the offering,
        # the other is context (notes being repaid, shares underlying).
        first = SecurityType.DEBT if debt.start() < equity.start() else SecurityType.EQUITY
        return OfferingShape(first, preliminary)

    logger.info("offering: could not identify the security type from the cover")
    return OfferingShape(SecurityType.UNKNOWN, preliminary)


def offering_shape_for(accession: str) -> OfferingShape | None:
    """Fetch a filing and read its cover. None when it cannot be read."""
    try:
        from advisor.research.edgar import EdgarClient

        text = EdgarClient().get_filing_text(accession, as_markdown=False)
    except Exception as exc:  # noqa: BLE001
        logger.info("offering: could not read %s: %s", accession, exc)
        return None
    return classify_offering(text)
