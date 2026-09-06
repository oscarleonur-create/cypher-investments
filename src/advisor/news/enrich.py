"""Turning "a 424B5 was filed" into "6.7% potential dilution".

A classification alone is not advice. ``DILUTION / HIGH`` tells you to look;
*$600,000,000 against an $8.96bn market cap* tells you how much to care. That
second number is the difference between a notification and a decision.

The extraction is deliberately narrow and **fails closed**. It reads the
offering amount out of the filing's own text, keeps the sentence it came from
verbatim, and returns nothing when it cannot find an unambiguous figure. A
filing with no size is still reported — just without a size. This module never
guesses, because a fabricated dilution percentage is far worse than a missing
one, and this repo has shipped fabricated numbers before.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

logger = logging.getLogger(__name__)

# "$600,000,000" / "$600.0 million" / "$1.2 billion"
_AMOUNT = re.compile(
    r"\$\s?([\d,]+(?:\.\d+)?)\s*(billion|million|bn|mm)?",
    re.IGNORECASE,
)
_SCALE = {"billion": 1e9, "bn": 1e9, "million": 1e6, "mm": 1e6, None: 1.0}

# The phrases an offering uses to state its size. Requiring one of these keeps
# the parser away from every other dollar figure in a prospectus.
_CONTEXT = re.compile(
    r"(aggregate offering price|aggregate amount|aggregate gross sales price|"
    r"aggregate principal amount|having an aggregate|up to \$|"
    r"offering price of up to|principal amount of)",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class OfferingSize:
    """An extracted offering amount and the sentence that proves it.

    "Offering" covers equity and the debt that becomes equity. T1 Energy sold
    $120m of *convertible notes* — 8.9% of its market cap if converted — and
    an extractor that only understood "aggregate offering price" reported that
    Tier A event with no size at all.
    """

    amount_usd: float
    quote: str

    def dilution_pct(self, market_cap: float) -> float | None:
        if market_cap and market_cap > 0:
            return self.amount_usd / market_cap
        return None


def _sentences(text: str) -> list[str]:
    return re.split(r"(?<=[.;])\s+", " ".join(text.split()))


def extract_offering_size(text: str, *, max_chars: int = 20000) -> OfferingSize | None:
    """Largest stated offering amount in the opening of a prospectus, or None.

    Only the first ``max_chars`` are read: a prospectus states its size on the
    cover, and the body is full of unrelated dollar amounts that would poison
    a whole-document scan.
    """
    if not text:
        return None

    best: OfferingSize | None = None
    for sentence in _sentences(text[:max_chars]):
        if not _CONTEXT.search(sentence):
            continue
        for raw, unit in _AMOUNT.findall(sentence):
            try:
                value = float(raw.replace(",", "")) * _SCALE[(unit or "").lower() or None]
            except (ValueError, KeyError):
                continue
            # A real offering is not $5 and not a trillion dollars; anything
            # outside that band is a page number or a typo, not a size.
            if not (1e5 <= value <= 5e11):
                continue
            if best is None or value > best.amount_usd:
                best = OfferingSize(amount_usd=value, quote=sentence[:400])

    if best is None:
        logger.info("enrich: no unambiguous offering size found; reporting without one")
    return best


def offering_size_for(accession: str) -> OfferingSize | None:
    """Fetch a filing's text and extract its offering size, or None."""
    try:
        from advisor.research.edgar import EdgarClient

        text = EdgarClient().get_filing_text(accession, as_markdown=False)
    except Exception as exc:  # noqa: BLE001
        logger.info("enrich: could not read filing %s: %s", accession, exc)
        return None
    return extract_offering_size(text)
