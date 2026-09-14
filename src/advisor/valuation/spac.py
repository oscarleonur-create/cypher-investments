"""Valuing a blank-cheque company, where EV/revenue is a category error.

CCXI is Churchill Capital Corp XI. It has no revenue and never will under that
name — it is a SPAC holding cash in trust while it merges with Agility
Robotics. Running it through the revenue model produced "no usable filing",
which is true and useless.

The valuation a SPAC actually has is its trust: the cash per public share that
holders can redeem for if they decline the deal. That number is a **floor**,
and the gap between it and the price is what the market is paying for the
merger. For CCXI: $420.9m of trust across 41.9m Class A shares is $10.04 a
share against a price of $13.40 — a 33% premium, and a stated downside if the
deal fails, which no revenue multiple could ever express.

Only Class A shares are backed. Founder Class B shares do not participate in
the trust, and counting them would understate the floor by a third.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import date

logger = logging.getLogger(__name__)

TRUST_CONCEPTS = ("AssetsHeldInTrustNoncurrent", "AssetsHeldInTrust")
SHARE_CONCEPT = "EntityCommonStockSharesOutstanding"

# The trust backs the public shares, which are the Class A / ordinary line on
# the cover. A founder class is labelled B (occasionally F).
_PUBLIC_CLASS = re.compile(r"\bclass\s*a\b|\bordinary shares\b|\bsubunit", re.IGNORECASE)
_FOUNDER_CLASS = re.compile(r"\bclass\s*[bf]\b|founder", re.IGNORECASE)


@dataclass(frozen=True)
class TrustValue:
    """A SPAC's redemption floor and what the market pays above it."""

    symbol: str
    asof: date
    trust_total: float
    public_shares: float
    price: float
    source_accession: str

    @property
    def per_share(self) -> float:
        return self.trust_total / self.public_shares

    @property
    def premium(self) -> float:
        """Price over trust, as a fraction. Negative means below the floor."""
        return self.price / self.per_share - 1

    def describe(self) -> str:
        direction = "premium to" if self.premium >= 0 else "discount to"
        return (
            f"${self.price:,.2f} against ${self.per_share:,.2f} of trust per public "
            f"share — a {abs(self.premium):.0%} {direction} the redemption floor"
        )


def _undimensioned(xbrl, concept: str):
    try:
        df = xbrl.query().by_concept(concept).to_dataframe()
    except Exception:  # noqa: BLE001
        return None
    if df is None or df.empty or "is_dimensioned" not in df.columns:
        return None
    plain = df[~df["is_dimensioned"].astype(bool)].dropna(subset=["numeric_value"])
    return plain if not plain.empty else None


def is_spac(xbrl) -> bool:
    """Whether this filer holds money in trust pending a business combination."""
    return any(_undimensioned(xbrl, c) is not None for c in TRUST_CONCEPTS)


def _public_shares(xbrl) -> float | None:
    """Shares the trust actually backs — Class A, never the founder class."""
    try:
        df = xbrl.query().by_concept(SHARE_CONCEPT).to_dataframe()
    except Exception:  # noqa: BLE001
        return None
    if df is None or df.empty or "label" not in df.columns:
        return None

    labels = df["label"].astype(str)
    public = df[labels.apply(lambda x: bool(_PUBLIC_CLASS.search(x)))]
    if not public.empty:
        return float(public["numeric_value"].dropna().sum())

    # No class labels at all: a single-class filer, so every share is public.
    founder = df[labels.apply(lambda x: bool(_FOUNDER_CLASS.search(x)))]
    if founder.empty:
        return float(df["numeric_value"].dropna().sum())

    logger.info("spac: share classes present but no public class identified")
    return None


def trust_value(symbol: str, filing, price: float) -> TrustValue | None:
    """Trust per public share and the premium the price carries, or None.

    Returns None rather than approximating: a floor computed off the wrong
    share count is worse than no floor, because it would be quoted as the
    downside.
    """
    try:
        xbrl = filing.xbrl()
    except Exception as exc:  # noqa: BLE001
        logger.info("spac: no XBRL on %s: %s", filing.accession_no, exc)
        return None
    if xbrl is None or not is_spac(xbrl):
        return None

    trust = None
    for concept in TRUST_CONCEPTS:
        rows = _undimensioned(xbrl, concept)
        if rows is not None:
            # The current balance is the larger of the two reported periods
            # for a trust that accrues interest; ties are harmless.
            trust = float(rows["numeric_value"].max())
            break

    shares = _public_shares(xbrl)
    if not trust or not shares or price <= 0:
        logger.info("spac: %s — trust or public share count unavailable", symbol)
        return None

    period_end = getattr(filing, "period_of_report", None) or filing.filing_date
    if isinstance(period_end, str):
        period_end = date.fromisoformat(period_end[:10])

    return TrustValue(
        symbol=symbol.upper(),
        asof=period_end,
        trust_total=trust,
        public_shares=shares,
        price=price,
        source_accession=str(filing.accession_no),
    )


def latest_trust_value(symbol: str, price: float) -> TrustValue | None:
    """Trust value from the newest periodic filing, or None if not a SPAC."""
    from advisor.news.edgar import company_for

    company = company_for(symbol)
    if company is None:
        return None
    try:
        filings = company.get_filings(form=["10-Q", "10-K"]).head(3)
    except Exception as exc:  # noqa: BLE001
        logger.warning("spac: filing lookup failed for %s: %s", symbol, exc)
        return None
    for filing in filings:
        value = trust_value(symbol, filing, price)
        if value is not None:
            return value
    return None
