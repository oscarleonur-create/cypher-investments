"""Universe resolution: held + watchlist + thesis symbols."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from advisor.daemon.book import EQUITY, BookSnapshot, Position
from advisor.daemon.universe import WatchReason, build_universe


def pos(symbol: str) -> Position:
    return Position(
        account="A",
        symbol=symbol,
        underlying=symbol,
        instrument=EQUITY,
        quantity=1,
        avg_open_price=10.0,
        close_price=10.0,
    )


def patched_store(watchlist=(), theses=()):
    store = MagicMock()
    store.load_watchlist.return_value = [{"symbol": s} for s in watchlist]
    store.list_theses.return_value = [{"symbol": s} for s in theses]
    return patch("advisor.research.store.ResearchStore", return_value=store)


class TestUniverse:
    def test_held_symbols_come_from_the_book(self):
        with patched_store():
            u = build_universe(BookSnapshot(positions=[pos("AMD"), pos("COHR")]))
        assert u.held == ["AMD", "COHR"]

    def test_watchlist_and_thesis_symbols_are_watched_not_held(self):
        with patched_store(watchlist=["JBL"], theses=["PENG"]):
            u = build_universe(BookSnapshot(positions=[pos("AMD")]))
        assert u.held == ["AMD"]
        assert u.watched_only == ["JBL", "PENG"]

    def test_a_held_symbol_that_also_has_a_thesis_carries_both_reasons(self):
        with patched_store(theses=["CRDO"]):
            u = build_universe(BookSnapshot(positions=[pos("CRDO")]))
        assert u.reason_for("CRDO") == {WatchReason.HELD, WatchReason.THESIS}
        assert u.held == ["CRDO"]

    def test_symbols_are_uppercased_and_deduped(self):
        with patched_store(watchlist=["amd", "AMD"]):
            u = build_universe(BookSnapshot(positions=[pos("AMD")]))
        assert u.all_symbols == ["AMD"]

    def test_blank_symbols_are_ignored(self):
        with patched_store(watchlist=["", "  "]):
            u = build_universe(BookSnapshot(positions=[pos("AMD")]))
        assert u.all_symbols == ["AMD"]

    def test_theses_can_be_excluded(self):
        with patched_store(theses=["PENG"]):
            u = build_universe(BookSnapshot(positions=[pos("AMD")]), include_theses=False)
        assert u.all_symbols == ["AMD"]

    def test_empty_book_and_empty_watchlist_is_an_empty_universe(self):
        with patched_store():
            assert len(build_universe(BookSnapshot())) == 0


class TestDegradation:
    def test_store_failure_falls_back_to_holdings(self):
        """A store problem must narrow the universe, never fail the job."""
        with patch("advisor.research.store.ResearchStore", side_effect=RuntimeError("locked")):
            u = build_universe(BookSnapshot(positions=[pos("AMD")]))
        assert u.held == ["AMD"]
        assert u.watched_only == []
