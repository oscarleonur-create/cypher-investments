"""SEC EDGAR as a live event source.

Distinct from ``advisor.research.edgar``, which reads five years of filings to
build a research report and caches aggressively. This module asks a different
question — *what is new since I last looked* — so it is watermark-driven and
must not serve a cached answer.

Two properties make EDGAR the strongest source available for free:

- **Entity resolution is exact.** A filing is attached to a CIK by the
  regulator. There is no keyword matching and no ambiguity.
- **Timestamps are exact.** Acceptance datetime is recorded to the second. The
  AAOI 424B5 was accepted 2026-08-21 20:09:22 UTC, minutes after Friday's
  close; the stock gapped down 12.6% at Monday's open. That ordering is
  checkable rather than inferred.

The known limitation, stated because it changes how much you can trust
silence: an 8-K may be filed up to four business days after the event it
reports. EDGAR is authoritative about *what* happened and not always first.
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any

from advisor.news.classify import classify_delisting, classify_filing
from advisor.news.foreign import classify_headline
from advisor.news.lead import eight_k_lead
from advisor.news.models import EntityMatch, MatchMethod, SourceItem, SourceTier

logger = logging.getLogger(__name__)

# Forms worth watching for a held position. Deliberately broader than the
# research module's set: dilution and distress live in forms a fundamental
# report never looks at.
WATCHED_FORMS: tuple[str, ...] = (
    "8-K",
    "424B1",
    "424B2",
    "424B3",
    "424B4",
    "424B5",
    "S-1",
    "S-3",
    "S-3ASR",
    "SC 13D",
    "SC 13G",
    "10-K",
    "10-Q",
    "NT 10-K",
    "NT 10-Q",
    "25-NSE",
    # Foreign private issuers never file a 10-Q or an 8-K. Nebius is a Dutch
    # N.V.: everything it discloses arrives as a 6-K, and its periodic report
    # is an annual 20-F. Without these it had nothing archived at all.
    "20-F",
    "40-F",
    "6-K",
    # Merger communications and the registration statement that carries a
    # deal. A SPAC holder's single most material event arrives as a stream of
    # 425s and an S-4, none of which were watched: Churchill Capital Corp XI
    # filed nine 425s and an S-4 for its Agility Robotics merger and the
    # system saw a 10-Q and an unrelated 8-K.
    "425",
    "S-4",
    "S-4/A",
    "DEFM14A",
    "PREM14A",
    "DEF 14A",
    # Notices of proposed insider sales. Individually routine; the cluster
    # detector in `insider.py` is what makes them mean anything.
    "144",
)

# Form 4 (insider transactions) is separated because a single company can file
# dozens in a week and they would drown everything else.
INSIDER_FORMS: tuple[str, ...] = ("4",)


def _client_ready() -> None:
    """Set the SEC identity and take a rate-limit slot, reusing research's."""
    from advisor.research.config import get_settings
    from advisor.research.edgar import RateLimiter, _ensure_identity

    settings = get_settings()
    _ensure_identity(settings.edgar_user_agent)
    RateLimiter(settings.edgar_rate_limit_per_sec).acquire()


def company_for(symbol: str) -> Any | None:
    """Resolve a ticker to an edgartools Company, or None."""
    try:
        _client_ready()
        from edgar import Company

        return Company(symbol.upper())
    except Exception as exc:  # noqa: BLE001
        logger.info("edgar: could not resolve %s: %s", symbol, exc)
        return None


def _acceptance(filing: Any) -> datetime:
    """Acceptance timestamp, falling back to the filing date at UTC midnight.

    The fallback is explicit rather than silent: a filing whose acceptance time
    is unavailable is still a real filing, but its intraday ordering against a
    price move cannot be trusted.
    """
    raw = getattr(filing, "acceptance_datetime", None)
    if isinstance(raw, datetime):
        return raw if raw.tzinfo else raw.replace(tzinfo=timezone.utc)
    filed = getattr(filing, "filing_date", None)
    if isinstance(filed, date):
        return datetime(filed.year, filed.month, filed.day, tzinfo=timezone.utc)
    return datetime.now(timezone.utc)


def _security_class(filing: Any) -> str | None:
    """The class of securities a Form 25-NSE removes, or None.

    Read for 25-NSE only. The form names the class in prose, and the
    difference between "common stock" and "warrants, each whole warrant
    exercisable to purchase one Common Stock at $11.50" is the difference
    between an interrupt and a footnote.
    """
    if str(getattr(filing, "form", "")).upper() != "25-NSE":
        return None
    try:
        text = " ".join(filing.text().split())
    except Exception as exc:  # noqa: BLE001
        logger.info("edgar: could not read 25-NSE %s: %s", filing.accession_no, exc)
        return None
    marker = text.lower().find("description of class")
    if marker < 0:
        return text[:1500]
    return text[max(0, marker - 400) : marker + 100]


def _headline(filing: Any, form: str) -> str:
    """A 6-K's press-release opening, or empty for every other form."""
    if not form.upper().startswith("6-K"):
        return ""
    from advisor.news.foreign import exhibit_text

    return exhibit_text(filing)


def _item_codes(filing: Any) -> list[str]:
    """8-K item numbers, empty for other forms or when parsing fails."""
    if not str(getattr(filing, "form", "")).startswith("8-K"):
        return []
    try:
        items = filing.obj().items
    except Exception as exc:  # noqa: BLE001
        logger.debug("edgar: items unavailable for %s: %s", filing.accession_no, exc)
        return []
    return [str(i).replace("Item", "").strip() for i in (items or [])]


def recent_filings(
    symbol: str,
    *,
    since: datetime | None = None,
    lookback_days: int = 14,
    forms: tuple[str, ...] = WATCHED_FORMS,
    limit: int = 40,
) -> list[SourceItem]:
    """Filings for ``symbol`` accepted after ``since``.

    ``since`` is the caller's watermark and ``lookback_days`` bounds the query
    window sent to EDGAR. **The window is widened to cover the watermark**: a
    laptop asleep for three weeks must not come back to a 14-day query and
    silently miss the fortnight it was off. Without this the AAOI 424B5, filed
    16 days before the run that looked for it, returned nothing at all.
    """
    if since is not None:
        gap = (datetime.now(timezone.utc) - since).days + 2
        lookback_days = max(lookback_days, gap)
    company = company_for(symbol)
    if company is None:
        return []

    start = (date.today() - timedelta(days=lookback_days)).isoformat()
    end = date.today().isoformat()
    try:
        _client_ready()
        raw = company.get_filings(form=list(forms), date=f"{start}:{end}")
    except Exception as exc:  # noqa: BLE001
        logger.warning("edgar: filing query failed for %s: %s", symbol, exc)
        return []

    cik = getattr(company, "cik", None)
    name = getattr(company, "name", None)
    out: list[SourceItem] = []
    for filing in list(raw)[:limit]:
        try:
            accepted = _acceptance(filing)
            if since is not None and accepted <= since:
                continue
            form = str(filing.form)
            codes = _item_codes(filing)
            security_class = _security_class(filing)
            headline = _headline(filing, form)
            # An 8-K's items say *that* something happened; the item text says
            # what. Read here because the filing is already in hand.
            lead = eight_k_lead(filing, codes) if form.upper().startswith("8-K") else None
            if security_class is not None:
                classification = classify_delisting(security_class)
            elif headline:
                # A 6-K has no item codes and a boilerplate body; its
                # press-release headline is the closest thing it has to one.
                classification = classify_headline(headline)
            else:
                classification = classify_filing(form, codes)
            out.append(
                SourceItem(
                    tier=SourceTier.PRIMARY,
                    provider="SEC EDGAR",
                    url=str(getattr(filing, "filing_url", "") or ""),
                    title=f"{form}: {name or symbol} {classification.label}",
                    published_at=accepted,
                    entity=EntityMatch(symbol=symbol.upper(), cik=cik, method=MatchMethod.CIK),
                    doc_type=form,
                    item_codes=codes,
                    accession=str(filing.accession_no),
                    summary=security_class or (headline[:900] or None) or lead,
                )
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("edgar: skipping unparseable filing for %s: %s", symbol, exc)

    out.sort(key=lambda i: i.published_at, reverse=True)
    return out
