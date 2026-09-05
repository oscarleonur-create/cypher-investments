"""Book normalization — shaped by what the live broker actually returns."""

from __future__ import annotations

from advisor.daemon.book import (
    EQUITY,
    EQUITY_OPTION,
    BookSnapshot,
    Position,
    normalize_instrument,
    position_from_broker,
)


def equity(**kw) -> Position:
    base = dict(
        account="5WI30382",
        symbol="AMD",
        underlying="AMD",
        instrument=EQUITY,
        quantity=2,
        multiplier=1,
        avg_open_price=504.065,
        close_price=456.16,
        mark_price=0.0,
    )
    return Position(**{**base, **kw})


class TestInstrumentNormalization:
    def test_handles_the_enum_repr_the_broker_returns(self):
        """Live data arrives as InstrumentType.EQUITY, not the string 'Equity'."""
        assert normalize_instrument("InstrumentType.EQUITY") == EQUITY
        assert normalize_instrument("InstrumentType.EQUITY_OPTION") == EQUITY_OPTION

    def test_handles_plain_strings(self):
        assert normalize_instrument("Equity") == EQUITY
        assert normalize_instrument("Equity Option") == EQUITY_OPTION

    def test_unknown_type_degrades_to_a_label(self):
        assert normalize_instrument("InstrumentType.FUTURE") == "FUTURE"
        assert normalize_instrument("") == "UNKNOWN"


class TestSignNormalization:
    def test_long_stays_positive(self):
        p = position_from_broker(
            {"symbol": "AMD", "quantity": "2", "quantity_direction": "Long"}, "acct"
        )
        assert p.quantity == 2
        assert not p.is_short

    def test_short_becomes_negative(self):
        """A short put is long delta / short vega — every later calc needs the sign."""
        p = position_from_broker(
            {"symbol": "AMD", "quantity": "3", "quantity_direction": "Short"}, "acct"
        )
        assert p.quantity == -3
        assert p.is_short

    def test_missing_quantity_is_zero_not_an_error(self):
        assert position_from_broker({"symbol": "X"}, "acct").quantity == 0


class TestPriceFallback:
    def test_falls_back_to_close_when_unmarked(self):
        """mark_price is 0 outside the session; overnight jobs must still see a
        priced book rather than a portfolio marked to zero."""
        assert equity(mark_price=0.0).price == 456.16

    def test_prefers_mark_when_available(self):
        assert equity(mark_price=460.0).price == 460.0

    def test_unpriced_position_reports_zero_not_a_crash(self):
        p = equity(mark_price=0.0, close_price=0.0)
        assert p.price == 0.0
        assert p.notional == 0.0
        assert p.unrealized_pct == 0.0


class TestPnl:
    def test_long_underwater(self):
        p = equity(mark_price=456.16)
        assert round(p.unrealized_pct, 4) == -0.095
        assert round(p.unrealized_pnl, 2) == -95.81

    def test_short_profits_when_price_falls(self):
        """Direction must flip the sign of the return."""
        p = equity(quantity=-2, avg_open_price=100.0, mark_price=90.0)
        assert round(p.unrealized_pct, 4) == 0.10
        assert p.unrealized_pnl > 0

    def test_zero_entry_price_does_not_divide_by_zero(self):
        assert equity(avg_open_price=0.0).unrealized_pct == 0.0

    def test_option_multiplier_scales_notional(self):
        p = equity(instrument=EQUITY_OPTION, quantity=-1, multiplier=100, mark_price=2.50)
        assert p.notional == 250.0
        assert p.is_option


class TestSnapshot:
    def test_empty_book_is_valid(self):
        """One of the two live accounts holds nothing — this is the real case."""
        b = BookSnapshot()
        assert b.symbols == []
        assert b.gross_notional == 0.0
        assert b.exposure_by_underlying() == {}

    def test_weight_with_zero_net_liq_does_not_divide_by_zero(self):
        b = BookSnapshot(positions=[equity()], net_liq=0.0)
        assert b.weight(b.positions[0]) == 0.0

    def test_exposure_sums_across_accounts_for_one_underlying(self):
        b = BookSnapshot(
            positions=[
                equity(account="A", mark_price=100.0, quantity=1),
                equity(account="B", mark_price=100.0, quantity=2),
            ]
        )
        assert b.exposure_by_underlying() == {"AMD": 300.0}

    def test_key_separates_the_same_symbol_in_two_accounts(self):
        a, b = equity(account="A"), equity(account="B")
        assert a.key() != b.key()

    def test_equities_and_options_split(self):
        b = BookSnapshot(positions=[equity(), equity(instrument=EQUITY_OPTION, multiplier=100)])
        assert len(b.equities) == 1
        assert len(b.options) == 1
