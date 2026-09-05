"""Ingest pass: broker state in, deduped events out."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from advisor.daemon.book import EQUITY, BookSnapshot, Position
from advisor.daemon.ingest import ingest_book
from advisor.daemon.market_calendar import now_et
from advisor.daemon.models import EventSource
from advisor.daemon.store import DaemonStore


@pytest.fixture
def store(tmp_path: Path):
    s = DaemonStore(tmp_path / "research.db")
    yield s
    s.close()


def pos(symbol="AMD", *, qty=2, entry=100.0, price=100.0, account="A") -> Position:
    return Position(
        account=account,
        symbol=symbol,
        underlying=symbol,
        instrument=EQUITY,
        quantity=qty,
        multiplier=1,
        avg_open_price=entry,
        close_price=price,
    )


def book(*positions, net_liq=10_000.0, partial=False) -> BookSnapshot:
    return BookSnapshot(as_of=now_et(), positions=list(positions), net_liq=net_liq, partial=partial)


def patched(snapshot):
    return patch("advisor.daemon.ingest.fetch_book", return_value=snapshot)


class TestIngest:
    async def test_first_pass_stores_a_baseline_without_alerting(self, store):
        with patched(book(pos(price=60.0))):
            result = await ingest_book(store, include_standing=False)
        assert result.new == 0
        assert store.load_latest_book() is not None

    async def test_second_pass_detects_a_crossing(self, store):
        with patched(book(pos(price=97.0))):
            await ingest_book(store)
        with patched(book(pos(price=90.0))):
            result = await ingest_book(store)
        assert [e.kind for e in result.events] == ["STOP_BREACHED"]
        assert result.new == 1

    async def test_repeating_the_same_pass_emits_nothing_new(self, store):
        """Two sweeps in a row on an unchanged book must stay silent."""
        with patched(book(pos(price=60.0))):
            await ingest_book(store, include_standing=True)
            result = await ingest_book(store, include_standing=True)
        assert result.new == 0

    async def test_standing_conditions_only_when_asked(self, store):
        deep = book(pos(price=60.0))
        with patched(deep):
            await ingest_book(store)  # baseline
            sweep = await ingest_book(store, include_standing=False)
            digest = await ingest_book(store, include_standing=True)
        assert sweep.events == []
        assert "DEEP_DRAWDOWN" in [e.kind for e in digest.events]

    async def test_watermark_advances(self, store):
        with patched(book(pos())):
            await ingest_book(store)
        assert store.get_watermark(EventSource.BROKER).last_seen_ts is not None


class TestFailureModes:
    async def test_broker_failure_degrades_instead_of_raising(self, store):
        """A dead broker must not take the job down."""
        with patch("advisor.daemon.ingest.fetch_book", side_effect=RuntimeError("api down")):
            result = await ingest_book(store)
        assert result.book is None
        assert result.new == 0
        assert "unavailable" in result.summary()

    async def test_partial_snapshot_is_not_saved_as_baseline(self, store):
        """Saving a partial book would report the missing account's positions
        as closed on the next diff — a wave of false alerts."""
        with patched(book(pos("AMD"), pos("COHR"))):
            await ingest_book(store)
        good = store.load_latest_book()

        with patched(book(pos("AMD"), partial=True)):
            await ingest_book(store)

        still = store.load_latest_book()
        assert len(still.positions) == len(good.positions) == 2

    async def test_a_partial_snapshot_does_not_report_false_closures(self, store):
        with patched(book(pos("AMD"), pos("COHR"))):
            await ingest_book(store)
        with patched(book(pos("AMD"), partial=True)):
            result = await ingest_book(store)
        assert [e.kind for e in result.events if e.kind == "POSITION_CLOSED"] == []

    async def test_empty_book_is_handled(self, store):
        """One live account genuinely holds nothing."""
        with patched(book()):
            result = await ingest_book(store, include_standing=True)
        assert result.new == 0
        assert "0 positions" in result.summary()
