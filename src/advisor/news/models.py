"""The source contract.

One rule governs this module, and it is the reason it exists:

    **The tier of a source decides what its items are allowed to do.**

A filing is the company speaking under legal liability, so it may interrupt
you. A scraped headline that merely mentions your ticker may be logged and
nothing more. Without that rule every source is implicitly trusted equally,
which is how "These 3 AI Stocks Are Way Off Their Highs" ends up sitting in
the same queue as a $600M dilutive offering.
"""

from __future__ import annotations

import hashlib
from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel, Field

from advisor.daemon.market_calendar import now_et
from advisor.daemon.models import EventTier


class SourceTier(StrEnum):
    """How much a source's word is worth."""

    PRIMARY = "PRIMARY"  # SEC EDGAR, company IR — the issuer, under liability
    BROKER = "BROKER"  # TastyTrade — contractual data about your own account
    AGGREGATOR = "AGGREGATOR"  # Tavily news mode — dated, entity-tagged reporting
    UNTAGGED = "UNTAGGED"  # yfinance — thematic scrape, no entity guarantee


# The ceiling each tier may reach in the event stream. UNTAGGED can never
# interrupt and never reaches a digest; it exists so that when something else
# fires, there is context sitting next to it.
MAX_EVENT_TIER: dict[SourceTier, EventTier] = {
    SourceTier.PRIMARY: EventTier.A,
    SourceTier.BROKER: EventTier.A,
    SourceTier.AGGREGATOR: EventTier.B,
    SourceTier.UNTAGGED: EventTier.C,
}


def capped_tier(tier: SourceTier, proposed: EventTier) -> EventTier:
    """Lower ``proposed`` to whatever ``tier`` is permitted to reach."""
    ceiling = MAX_EVENT_TIER[tier]
    order = {EventTier.A: 0, EventTier.B: 1, EventTier.C: 2}
    return proposed if order[proposed] >= order[ceiling] else ceiling


class MatchMethod(StrEnum):
    """How an item was tied to a security, strongest first."""

    CIK = "CIK"  # regulatory identifier — exact
    PROVIDER_TAG = "PROVIDER_TAG"  # the provider asserted the ticker
    COMPANY_NAME = "COMPANY_NAME"  # registered name found in the text
    CASHTAG = "CASHTAG"  # "$AAOI"
    TICKER_TOKEN = "TICKER_TOKEN"  # bare uppercase token, weakest
    NONE = "NONE"  # no defensible link — the item is dropped


MATCH_CONFIDENCE: dict[MatchMethod, float] = {
    MatchMethod.CIK: 1.00,
    MatchMethod.PROVIDER_TAG: 0.90,
    MatchMethod.COMPANY_NAME: 0.80,
    MatchMethod.CASHTAG: 0.70,
    MatchMethod.TICKER_TOKEN: 0.60,
    MatchMethod.NONE: 0.0,
}


class EntityMatch(BaseModel):
    """The link between an item and a security, and how it was established."""

    symbol: str
    cik: int | None = None
    method: MatchMethod = MatchMethod.NONE

    @property
    def confidence(self) -> float:
        return MATCH_CONFIDENCE[self.method]

    @property
    def resolved(self) -> bool:
        return self.method is not MatchMethod.NONE


class SourceItem(BaseModel):
    """One external fact, with everything needed to audit it later.

    ``published_at`` is required and timezone-aware. An undated item cannot be
    aged, cannot be ordered against a price move, and cannot be shown to you
    honestly — so it is refused at the boundary rather than stored with a
    guessed timestamp.
    """

    tier: SourceTier
    provider: str
    url: str
    title: str
    published_at: datetime
    retrieved_at: datetime = Field(default_factory=now_et)
    entity: EntityMatch
    doc_type: str | None = None  # "8-K", "424B5", "NEWS"
    item_codes: list[str] = Field(default_factory=list)  # 8-K items, e.g. ["5.02"]
    accession: str | None = None  # SEC accession number — a perfect dedup key
    summary: str | None = None

    def dedup_key(self) -> str:
        """Stable identity for this item.

        An SEC accession number is globally unique and assigned by the
        regulator, so it is used verbatim when present. Otherwise the URL
        identifies the item — the same story republished under two URLs is a
        clustering problem, handled separately, not a dedup one.
        """
        if self.accession:
            return self.accession
        return hashlib.sha256(self.url.encode()).hexdigest()[:32]
