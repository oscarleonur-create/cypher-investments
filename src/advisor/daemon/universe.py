"""What the daemon watches.

Scope was settled in the design: open positions across both accounts, plus the
watchlist. Nothing else. A wider net (peers, sector cohorts) multiplies events
and therefore LLM cost, and buys nothing until the narrow case is proven.

Symbols carry their reason for being watched, because the relevance gate
treats them differently: a held name earns a Tier A interrupt, a watchlist name
generally does not.
"""

from __future__ import annotations

import logging
from enum import StrEnum

from pydantic import BaseModel, Field

from advisor.daemon.book import BookSnapshot

logger = logging.getLogger(__name__)


class WatchReason(StrEnum):
    HELD = "held"  # an open position
    WATCHLIST = "watchlist"  # on the watchlist, not held
    THESIS = "thesis"  # has a stated thesis, not held


class WatchedSymbol(BaseModel):
    symbol: str
    reasons: set[WatchReason] = Field(default_factory=set)

    @property
    def is_held(self) -> bool:
        return WatchReason.HELD in self.reasons

    model_config = {"arbitrary_types_allowed": True}


class Universe(BaseModel):
    """The monitored set, with each symbol's reason for inclusion."""

    symbols: dict[str, WatchedSymbol] = Field(default_factory=dict)

    def add(self, symbol: str, reason: WatchReason) -> None:
        key = symbol.strip().upper()
        if not key:
            return
        entry = self.symbols.get(key)
        if entry is None:
            entry = WatchedSymbol(symbol=key)
            self.symbols[key] = entry
        entry.reasons.add(reason)

    @property
    def held(self) -> list[str]:
        return sorted(s for s, w in self.symbols.items() if w.is_held)

    @property
    def watched_only(self) -> list[str]:
        """Symbols tracked but not currently held."""
        return sorted(s for s, w in self.symbols.items() if not w.is_held)

    @property
    def all_symbols(self) -> list[str]:
        return sorted(self.symbols)

    def reason_for(self, symbol: str) -> set[WatchReason]:
        entry = self.symbols.get(symbol.upper())
        return entry.reasons if entry else set()

    def __len__(self) -> int:
        return len(self.symbols)


def build_universe(book: BookSnapshot, *, include_theses: bool = True) -> Universe:
    """Resolve held symbols, the watchlist, and (optionally) thesis symbols.

    Watchlist and thesis lookups are best-effort: a store problem must degrade
    the universe to "what we hold", never fail the job outright.
    """
    universe = Universe()
    for symbol in book.symbols:
        universe.add(symbol, WatchReason.HELD)

    try:
        from advisor.research.config import get_settings
        from advisor.research.store import ResearchStore

        store = ResearchStore(get_settings().db_path)
        try:
            for row in store.load_watchlist():
                universe.add(str(row.get("symbol", "")), WatchReason.WATCHLIST)
            if include_theses:
                for row in store.list_theses():
                    universe.add(str(row.get("symbol", "")), WatchReason.THESIS)
        finally:
            store.close()
    except Exception as exc:  # noqa: BLE001
        logger.warning("universe: watchlist/theses unavailable, holdings only: %s", exc)

    return universe
