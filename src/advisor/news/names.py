"""How a headline names each company: a list, reviewed by the holder.

The registered name is the wrong key for a headline, and the audit of
2026-09-27 measured it on real titles from this week:

- **COHR** ("COHERENT CORP." → "coherent"): Google Research's "Automating
  coherent long-form video generation", a Nokia piece on "coherent
  pluggables" and a Science paper were all kept as Coherent news.
- **SPCX** ("SPACE EXPLORATION TECHNOLOGIES CORP" → "space exploration"): six
  of seven kept items were about space exploration in general — ESA, a Boy
  Scouts merit badge, an astronomy club — while "SpaceX plans to launch…"
  was dropped.
- **AMZN** ("AMAZON COM INC" → "amazon com"): "Amazon Blocks Meta's Muse
  Agent" was dropped, because no headline writes "Amazon com".

So each name is listed as headlines write it, and matched as a whole word
with its capitals ("Coherent", never "coherent"):

- **strong** names identify the company on their own ("SpaceX", "T1 Energy").
- **weak** names are also ordinary words, even capitalised in a title-case
  headline ("Coherent Long-Form Video"). They count only beside something
  that pins them to the listed company in the same text: the ticker, a
  strong name, or a market word (stock, shares, earnings…).

A symbol missing from the list falls back to the registered name, as
before, and the resolver logs it, so a new watchlist name shows up as a gap
rather than silently matching on a common word.
"""

from __future__ import annotations

import re

# Proposed 2026-09-27 for the held book and the watchlists; the holder reviews it.
COMPANY_NAMES: dict[str, dict[str, tuple[str, ...]]] = {
    "SPCX": {"strong": ("SpaceX", "Space Exploration Technologies")},
    "AAOI": {"strong": ("Applied Optoelectronics",)},
    # The SPAC and the company it is merging with.
    "CCXI": {"strong": ("Churchill Capital Corp XI", "Agility Robotics")},
    "CRDO": {"strong": ("Credo Technology",), "weak": ("Credo",)},
    "COHR": {"strong": ("Coherent Corp",), "weak": ("Coherent",)},
    "CBRS": {"strong": ("Cerebras",)},
    "NBIS": {"strong": ("Nebius",)},
    "TE": {"strong": ("T1 Energy",)},
    "AMD": {"strong": ("Advanced Micro Devices",)},
    "AMZN": {"strong": ("Amazon", "Amazon.com", "Amazon Web Services", "AWS")},
    "INTC": {"strong": ("Intel",)},
    "JBL": {"strong": ("Jabil",)},
    "META": {"strong": ("Meta Platforms", "Meta")},
    "MSFT": {"strong": ("Microsoft",)},
    "PENG": {"strong": ("Penguin Solutions",)},
    "WOLF": {"strong": ("Wolfspeed",)},
}

# Words that pin a weak name to the listed company when they share its text.
MARKET_WORDS = (
    "stock",
    "stocks",
    "shares",
    "shareholders",
    "investors",
    "earnings",
    "revenue",
    "guidance",
    "analyst",
    "analysts",
    "Inc",
    "Corp",
    "Nasdaq",
    "NASDAQ",
    "NYSE",
)


def _word(term: str, text: str) -> bool:
    """``term`` as a whole word, capitals as written. Possessives count ("Credo's")."""
    return bool(re.search(rf"(?<![A-Za-z0-9]){re.escape(term)}(?![A-Za-z0-9])", text))


def names_company(symbol: str, text: str) -> bool | None:
    """Whether ``text`` names the listed company; None when the symbol is not listed."""
    names = COMPANY_NAMES.get(symbol.upper())
    if names is None:
        return None
    if any(_word(n, text) for n in names["strong"]):
        return True
    if not any(_word(n, text) for n in names.get("weak", ())):
        return False
    pins = (symbol.upper(), f"${symbol.upper()}", *MARKET_WORDS)
    return any(_word(p, text) for p in pins)


def query_names(symbol: str) -> tuple[str, ...]:
    """The names a search should ask for: the strong ones, then the weak."""
    names = COMPANY_NAMES.get(symbol.upper())
    return (*names["strong"], *names.get("weak", ())) if names else ()
