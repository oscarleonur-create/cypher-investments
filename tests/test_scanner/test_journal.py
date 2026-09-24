"""The trade journal: broker fills mark candidates taken; skips carry a reason."""

from __future__ import annotations

from datetime import date, datetime, timezone

import pytest
from advisor.daemon import market_calendar as mc
from advisor.scanner.journal import (
    Fill,
    fill_from_transaction,
    match,
    review,
    status,
    sync_fills,
)
from advisor.scanner.models import (
    Candidate,
    DecisionSource,
    Phase,
    Setup,
    SkipReason,
    TradeDecision,
)
from advisor.scanner.store import ScannerStore


def et(y, m, d, hh, mm):
    return datetime(y, m, d, hh, mm, tzinfo=mc.MARKET_TZ)


DAY = date(2026, 9, 24)


def cand(symbol="AMD", setup=Setup.CATALYST_GAP, phase=Phase.SESSION, outcomes=None):
    c = Candidate(
        session=DAY,
        setup=setup,
        symbol=symbol,
        detected_at=et(2026, 9, 24, 9, 35),
        phase=phase,
        price=100.0,
        prev_close=95.0,
        change=0.05,
    )
    c.outcomes = outcomes or {}
    return c


def fill(
    symbol="AMD",
    action="Buy to Open",
    when=None,
    price=101.0,
    qty=10,
    instrument="Equity",
    underlying=None,
):
    return Fill(
        underlying=underlying or symbol,
        symbol=symbol,
        action=action,
        quantity=qty,
        price=price,
        executed_at=when or et(2026, 9, 24, 9, 40),
        account="5WI30382",
        instrument=instrument,
    )


class TestMatch:
    def test_same_symbol_same_day_buy(self):
        (d,) = match([cand()], [fill()])
        assert d.taken and d.source is DecisionSource.BROKER and d.fill_price == 101.0

    def test_buying_before_the_scanner_saw_it_still_counts(self):
        (d,) = match([cand()], [fill(when=et(2026, 9, 24, 9, 31))])
        assert d.taken

    def test_first_fill_price_and_summed_quantity(self):
        fills = [
            fill(when=et(2026, 9, 24, 10, 0), price=103.0, qty=5),
            fill(when=et(2026, 9, 24, 9, 45), price=102.0, qty=5),
        ]
        (d,) = match([cand()], fills)
        assert d.fill_price == 102.0 and d.quantity == 10

    def test_other_day_or_symbol_does_not_match(self):
        assert match([cand()], [fill(when=et(2026, 9, 25, 9, 40))]) == []
        assert match([cand()], [fill(symbol="INTC")]) == []

    def test_selling_to_open_is_not_a_long_entry(self):
        assert match([cand()], [fill(action="Sell to Open")]) == []

    def test_a_call_on_the_underlying_counts_and_says_so(self):
        f = fill(symbol="AMD  261016C00200000", underlying="AMD", instrument="Equity Option")
        (d,) = match([cand()], [f])
        assert d.taken and d.note == "Equity Option"

    def test_utc_fill_after_8pm_et_belongs_to_its_et_day(self):
        """00:30 UTC on 09-25 is 20:30 ET on 09-24 — an after-hours fill that day."""
        f = fill(when=datetime(2026, 9, 25, 0, 30, tzinfo=timezone.utc))
        assert len(match([cand()], [f])) == 1

    def test_premarket_and_session_candidates_both_marked(self):
        cs = [cand(phase=Phase.PREMARKET), cand()]
        assert len(match(cs, [fill()])) == 2


@pytest.fixture
def store(tmp_path):
    s = ScannerStore(tmp_path / "j.db")
    yield s
    s.close()


def test_sync_is_idempotent(store):
    store.add(cand())
    assert sync_fills(store, [DAY], fetch=lambda a, b: [fill()]) == 1
    assert sync_fills(store, [DAY], fetch=lambda a, b: [fill()]) == 0


def test_no_candidates_never_calls_the_broker(store):
    called = []
    assert sync_fills(store, [DAY], fetch=lambda a, b: called.append(1) or []) == 0
    assert called == []


def test_broker_down_writes_nothing(store):
    store.add(cand())
    assert sync_fills(store, [DAY], fetch=lambda a, b: []) == 0
    assert store.latest_decisions() == {}


def test_a_fill_found_after_a_skip_stands(store):
    c = cand()
    store.add(c)
    store.record_decision(
        TradeDecision(
            candidate_id=c.id,
            taken=False,
            source=DecisionSource.USER,
            reason=SkipReason.LATE,
            decided_at=et(2026, 9, 24, 10, 0),
        )
    )
    sync_fills(store, [DAY], fetch=lambda a, b: [fill()])
    assert store.latest_decisions()[c.id].taken


def test_status_labels():
    c = cand()
    assert status(c, None) == "untagged"
    assert (
        status(c, TradeDecision(candidate_id=c.id, taken=True, source=DecisionSource.BROKER))
        == "taken"
    )
    skip = TradeDecision(
        candidate_id=c.id, taken=False, source=DecisionSource.USER, reason=SkipReason.NEWS
    )
    assert status(c, skip) == "skipped:news"


def test_review_separates_taken_skipped_and_untagged():
    taken = cand("AMD", outcomes={"close": 0.03})
    skipped = cand("INTC", outcomes={"close": 0.05})
    untagged = cand("META", outcomes={"close": -0.02})
    decisions = {
        taken.id: TradeDecision(candidate_id=taken.id, taken=True, source=DecisionSource.BROKER),
        skipped.id: TradeDecision(
            candidate_id=skipped.id, taken=False, source=DecisionSource.USER, reason=SkipReason.LATE
        ),
    }
    rows = {r["status"]: r for r in review([taken, skipped, untagged], decisions)}
    assert set(rows) == {"taken", "skipped:late", "untagged"}
    assert rows["skipped:late"]["close"]["mean"] == pytest.approx(0.05)


def test_review_empty():
    assert review([], {}) == []


class TestFromBroker:
    """The SDK hands back enums; a plain str() comparison matched nothing live."""

    def tx(self, action, ttype="Trade", instrument=None):
        from types import SimpleNamespace

        from tastytrade.order import InstrumentType, OrderAction

        return SimpleNamespace(
            transaction_type=ttype,
            action=OrderAction(action),
            instrument_type=instrument or InstrumentType.EQUITY,
            underlying_symbol="GOOGL",
            symbol="GOOGL",
            quantity=3,
            price=346.5,
            executed_at=datetime(2026, 9, 15, 19, 10, tzinfo=timezone.utc),
        )

    def test_buy_to_open_enum_becomes_a_fill(self):
        f = fill_from_transaction(self.tx("Buy to Open"), "5WI30382")
        assert f is not None and f.action == "Buy to Open" and f.instrument == "Equity"

    def test_closing_trades_are_not_entries(self):
        assert fill_from_transaction(self.tx("Sell to Close"), "5WI30382") is None

    def test_non_trade_rows_are_skipped(self):
        assert fill_from_transaction(self.tx("Buy to Open", ttype="Money Movement"), "x") is None

    def test_option_instrument_is_read_as_its_value(self):
        from tastytrade.order import InstrumentType

        tx = self.tx("Buy to Open", instrument=InstrumentType.EQUITY_OPTION)
        assert fill_from_transaction(tx, "x").instrument == "Equity Option"
