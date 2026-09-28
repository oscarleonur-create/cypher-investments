"""Every US-listed security, from Nasdaq Trader's symbol directory, cut to common stock.

The directory is the exchanges' own list: Nasdaq's file and the "other
listed" file (NYSE, NYSE American, NYSE Arca, Cboe, IEX). Measured on
2026-09-27: 7,515 non-ETF issues, fetched in under a second. The same site
already supplies the trade-halts feed.

A listing is common stock when nothing marks it as something else. The
directory flags ETFs and test issues; everything else — warrants, units,
rights, preferreds, notes — is recognised from the security name, because
that is the only place the directory says it. Each exclusion carries its
reason, so a company wrongly cut (a firm called "Unit Corp") shows up in the
counts instead of vanishing.

Blank-check companies (SPACs) are cut by name. They have no business to
analyse until they merge, and their price sits on the trust value.

A listing must also be an SEC filer: every later family reads filings, and a
name the SEC cannot resolve has none to read.
"""

from __future__ import annotations

import csv
import io
import logging
import re
from collections.abc import Callable
from dataclasses import dataclass

logger = logging.getLogger(__name__)

NASDAQ_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/nasdaqlisted.txt"
OTHER_URL = "https://www.nasdaqtrader.com/dynamic/SymDir/otherlisted.txt"
SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"

_OTHER_EXCHANGES = {
    "A": "NYSE American",
    "N": "NYSE",
    "P": "NYSE Arca",
    "Z": "Cboe BZX",
    "V": "IEX",
}

# Nasdaq's financial-status codes. Delinquent (E) has no current financials;
# bankrupt (Q) is not an opportunity this system can analyse. The combined
# codes carry one or both. Deficient alone (D, usually a sub-$1 bid) is left
# to the price floor.
_DELINQUENT = {"E", "H", "J", "K"}
_BANKRUPT = {"Q", "G", "J", "K"}

# Security-name markers of anything that is not the common stock.
_NOT_COMMON: tuple[tuple[str, re.Pattern[str]], ...] = (
    ("warrant", re.compile(r"\bwarrants?\b", re.I)),
    ("unit", re.compile(r"\bunits?\b", re.I)),
    ("right", re.compile(r"\brights?\b", re.I)),
    ("preferred", re.compile(r"\bpreferred\b|\bpref\b|\bperpetual\b", re.I)),
    ("debt", re.compile(r"\bnotes?\b|\bdebentures?\b|\bbonds?\b|% senior|\bdue 20\d\d\b", re.I)),
    # Closed-end funds trade like common stock but hold a portfolio, not a business.
    ("fund", re.compile(r"\bfund\b|\bclosed[- ]end\b|\betns?\b|\bexchange[- ]traded\b", re.I)),
)
_SPAC = re.compile(
    r"\bacquisition (corp|corporation|company|co\b|inc\b|ltd\b|limited)|\bblank check\b", re.I
)


@dataclass(frozen=True)
class Listing:
    symbol: str  # as the exchange writes it: BRK.B
    name: str
    exchange: str
    etf: bool = False
    test: bool = False
    status: str = ""  # Nasdaq financial status; blank for other exchanges

    @property
    def yahoo(self) -> str:
        """Yahoo's and the SEC's spelling of a class share: BRK-B."""
        return self.symbol.replace(".", "-").replace("/", "-")


def _rows(text: str) -> list[dict[str, str]]:
    """Pipe-delimited rows, without the trailing ``File Creation Time`` line."""
    lines = [ln for ln in text.splitlines() if ln and not ln.startswith("File Creation Time")]
    return list(csv.DictReader(io.StringIO("\n".join(lines)), delimiter="|"))


def parse_nasdaq(text: str) -> list[Listing]:
    out = []
    for r in _rows(text):
        symbol = (r.get("Symbol") or "").strip()
        if not symbol:
            continue
        out.append(
            Listing(
                symbol=symbol,
                name=(r.get("Security Name") or "").strip(),
                exchange="Nasdaq",
                etf=(r.get("ETF") or "").strip() == "Y",
                test=(r.get("Test Issue") or "").strip() == "Y",
                status=(r.get("Financial Status") or "").strip(),
            )
        )
    return out


def parse_other(text: str) -> list[Listing]:
    out = []
    for r in _rows(text):
        symbol = (r.get("ACT Symbol") or "").strip()
        if not symbol:
            continue
        code = (r.get("Exchange") or "").strip()
        out.append(
            Listing(
                symbol=symbol,
                name=(r.get("Security Name") or "").strip(),
                exchange=_OTHER_EXCHANGES.get(code, code or "unknown"),
                etf=(r.get("ETF") or "").strip() == "Y",
                test=(r.get("Test Issue") or "").strip() == "Y",
            )
        )
    return out


def exclusion(listing: Listing) -> str | None:
    """Why a listing is not a common stock this system can analyse, or None. Pure."""
    if listing.etf:
        return "etf"
    if listing.test:
        return "test issue"
    if listing.status in _BANKRUPT:
        return "bankrupt"
    if listing.status in _DELINQUENT:
        return "delinquent in filings"
    # '$' marks a preferred series in the ACT symbology (ABR$D); '=' and '^'
    # are units and rights on some feeds. A dot is a share class (BRK.B), kept.
    if re.search(r"[$=^+#]", listing.symbol):
        return "preferred"
    for reason, pattern in _NOT_COMMON:
        if pattern.search(listing.name):
            return reason
    if _SPAC.search(listing.name):
        return "probable SPAC"
    return None


def parse_sec_tickers(payload: dict) -> dict[str, int]:
    """``{ticker: cik}`` from the SEC's company_tickers.json. Pure."""
    out: dict[str, int] = {}
    for row in (payload or {}).values():
        try:
            ticker = str(row["ticker"]).strip().upper()
            cik = int(row["cik_str"])
        except (KeyError, TypeError, ValueError):
            continue
        # The file lists a company's primary ticker first; keep it on collisions.
        out.setdefault(ticker, cik)
    return out


@dataclass(frozen=True)
class Directory:
    """Every listing, with its exclusion reason (None = common stock) and CIK."""

    listings: list[Listing]
    reasons: dict[str, str | None]  # by Yahoo symbol
    ciks: dict[str, int]  # by Yahoo symbol, only those the SEC resolves

    def common(self) -> list[Listing]:
        """Common stock of an SEC filer: the names E0 goes on to price."""
        return [x for x in self.listings if self.reasons.get(x.yahoo) is None]


def build_directory(nasdaq: str, other: str, sec: dict) -> Directory:
    """Parse both files and the SEC map into one directory. Pure.

    A symbol on both files (it happens during a listing transfer) is kept once,
    Nasdaq's row first.
    """
    tickers = parse_sec_tickers(sec)
    seen: set[str] = set()
    listings: list[Listing] = []
    reasons: dict[str, str | None] = {}
    ciks: dict[str, int] = {}
    for x in parse_nasdaq(nasdaq) + parse_other(other):
        key = x.yahoo.upper()
        if key in seen:
            continue
        seen.add(key)
        listings.append(x)
        reason = exclusion(x)
        cik = tickers.get(key)
        if cik is not None:
            ciks[key] = cik
        elif reason is None:
            reason = "no SEC filer"
        reasons[key] = reason
    return Directory(listings=listings, reasons=reasons, ciks=ciks)


def _get_text(url: str) -> str:
    import httpx

    r = httpx.get(url, headers={"User-Agent": "Mozilla/5.0 advisor"}, timeout=30)
    r.raise_for_status()
    return r.text


def _get_sec_json(url: str) -> dict:
    import httpx

    from advisor.news.edgar import _client_ready
    from advisor.research.config import get_settings

    _client_ready()
    r = httpx.get(url, headers={"User-Agent": get_settings().edgar_user_agent}, timeout=30)
    r.raise_for_status()
    return r.json()


def fetch_directory(
    get_text: Callable[[str], str] = _get_text,
    get_json: Callable[[str], dict] = _get_sec_json,
) -> Directory:
    """Today's directory. Raises if either exchange file is unreachable.

    A partial directory would silently drop half the market, so there is no
    fallback: the caller keeps yesterday's universe and reports the failure.
    """
    nasdaq, other = get_text(NASDAQ_URL), get_text(OTHER_URL)
    if "|" not in nasdaq or "|" not in other:
        raise ValueError("symbol directory: unexpected format (no pipe-delimited rows)")
    return build_directory(nasdaq, other, get_json(SEC_TICKERS_URL))
