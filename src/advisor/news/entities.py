"""Tying an item to a security, defensibly.

This is the component whose absence produced the failure that started Phase
2c: three "AAOI" headlines from yfinance, none of which named AAOI. Two were
about Nvidia and one about HPE.

The rule is that a match must be *earned*. An item that cannot be tied to a
holding by an identifier, a provider's own tag, or the registered company name
is dropped. Dropping a real item is a missed signal; keeping a false one puts
a story about HPE under your AAOI thesis, which is worse.
"""

from __future__ import annotations

import logging
import re

from advisor.news.models import EntityMatch, MatchMethod

logger = logging.getLogger(__name__)

# Tickers that are also ordinary words, or too short to be distinctive. A bare
# token match on these is meaningless — the book holds TE (T1 Energy), and
# "te" appears in a great deal of text that has nothing to do with it. These
# must match on the company name instead.
AMBIGUOUS_TICKERS: frozenset[str] = frozenset(
    {
        "A",
        "ALL",
        "AN",
        "ANY",
        "ARE",
        "AS",
        "AT",
        "BE",
        "BIG",
        "BY",
        "CAR",
        "DO",
        "EAT",
        "FOR",
        "GO",
        "GOOD",
        "HAS",
        "HE",
        "HIS",
        "IT",
        "KEY",
        "LOW",
        "NEW",
        "NOW",
        "OLD",
        "ON",
        "ONE",
        "OR",
        "OUT",
        "PLAY",
        "RUN",
        "SO",
        "TE",
        "TWO",
        "UP",
        "US",
        "WE",
        "Y",
    }
)

# Corporate suffixes stripped before comparing names, so "Applied
# Optoelectronics, Inc." matches a headline saying "Applied Optoelectronics".
_SUFFIXES = re.compile(
    r"[,\.]?\s*\b(inc|incorporated|corp|corporation|co|company|ltd|limited|plc|"
    r"nv|n\.v|sa|s\.a|ag|holdings?|group|technologies|technology|systems)\b\.?",
    re.IGNORECASE,
)
_PUNCT = re.compile(r"[^a-z0-9 ]+")


def normalize_name(name: str) -> str:
    """Reduce a company name to a comparable core."""
    trimmed = _SUFFIXES.sub("", name or "")
    return _PUNCT.sub(" ", trimmed.lower()).strip()


def is_ambiguous(symbol: str) -> bool:
    """True when a bare ticker token is not evidence of anything."""
    upper = symbol.upper()
    return len(upper) <= 2 or upper in AMBIGUOUS_TICKERS


def resolve_entity(
    symbol: str,
    *,
    text: str = "",
    company_name: str | None = None,
    provider_tags: list[str] | None = None,
    cik: int | None = None,
) -> EntityMatch:
    """Establish the strongest defensible link between ``text`` and ``symbol``.

    Order matters: a regulatory identifier beats a provider's assertion, which
    beats a name found in prose, which beats a bare ticker token.
    """
    upper = symbol.upper()

    if cik is not None:
        return EntityMatch(symbol=upper, cik=cik, method=MatchMethod.CIK)

    if provider_tags and upper in {t.upper() for t in provider_tags}:
        return EntityMatch(symbol=upper, method=MatchMethod.PROVIDER_TAG)

    haystack = text or ""
    if company_name:
        core = normalize_name(company_name)
        # Require a substantive name, not "Co" or a single letter.
        if len(core) >= 4 and core in normalize_name(haystack):
            return EntityMatch(symbol=upper, method=MatchMethod.COMPANY_NAME)

    if re.search(rf"\${re.escape(upper)}\b", haystack, re.IGNORECASE):
        return EntityMatch(symbol=upper, method=MatchMethod.CASHTAG)

    # A bare token only counts for tickers distinctive enough to mean something.
    if not is_ambiguous(upper) and re.search(rf"(?<![A-Za-z0-9]){upper}(?![A-Za-z0-9])", haystack):
        return EntityMatch(symbol=upper, method=MatchMethod.TICKER_TOKEN)

    return EntityMatch(symbol=upper, method=MatchMethod.NONE)


def company_name_for(symbol: str) -> str | None:
    """Registered name from SEC company data, or None if it cannot be resolved.

    Uses the same identity and rate limiting as the rest of the EDGAR path.
    """
    try:
        from advisor.news.edgar import company_for

        company = company_for(symbol)
        return company.name if company else None
    except Exception as exc:  # noqa: BLE001
        logger.debug("entities: name lookup failed for %s: %s", symbol, exc)
        return None
