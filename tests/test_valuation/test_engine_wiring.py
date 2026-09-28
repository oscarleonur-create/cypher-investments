"""The engine wired into the daemon: the weekly job, the switch from the old
arithmetic, the range shown across margins, the scorecard rows and the price."""

from __future__ import annotations

import asyncio
from datetime import date
from pathlib import Path
from unittest.mock import patch

import pytest
from advisor.daemon.store import DaemonStore
from advisor.story.scorecard import build_scorecard
from advisor.valuation.figures import Figures, last_price
from advisor.valuation.implied import build_snapshot, undiscounted_expectations
from advisor.valuation.ingest import refresh_valuations, shift_event
from advisor.valuation.margins import required_range
from advisor.valuation.models import OwnMargin, ValuationSnapshot


@pytest.fixture
def store(tmp_path: Path):
    s = DaemonStore(tmp_path / "research.db")
    yield s
    s.close()


def figures(symbol="MSFT", price=516.17, **overrides) -> Figures:
    """MSFT's figures as the SEC reported them on 2026-09-27."""
    base = dict(
        symbol=symbol,
        price=price,
        price_source="test",
        shares=7_425.5e6,
        net_cash=57.48e9,
        balance_asof=date(2026, 6, 30),
        source_accession="0001193125-26-323660",
        revenue_base=331.84e9,
        revenue_base_label="trailing 12 months to 2026-06-30 (SEC XBRL)",
        revenue_growth=0.178,
        revenue_growth_label="trailing 12 months to 2026-06-30 vs a year earlier",
        start_margin=0.202,
        start_margin_label="FCF, trailing 12 months to 2026-06-30",
        margins=[
            OwnMargin(kind="fcf", value=0.202, label="FCF, trailing 12 months to 2026-06-30"),
            OwnMargin(kind="median", value=0.254, label="FY2024–FY2026 median FCF"),
            OwnMargin(kind="nopat", value=0.370, label="operating margin after 21% tax"),
        ],
    )
    base.update(overrides)
    return Figures(**base)


def legacy_snapshot(symbol="MSFT", price=516.17, asof=date(2026, 9, 20)) -> ValuationSnapshot:
    """A row as stored before 2026-09-27: undiscounted, no method field."""
    shares, net_cash = 7_425.5e6, 57.48e9
    ev = shares * price - net_cash
    base = undiscounted_expectations(ev, 331.84e9, terminal_multiple=25, fcf_margin=0.25)
    return ValuationSnapshot(
        symbol=symbol, asof=asof, price=price, shares_outstanding=shares,
        market_cap=shares * price, net_cash=net_cash, enterprise_value=ev,
        revenue_runrate=331.84e9, ev_to_revenue=11.4, source_accession="acc",
        period_end=date(2026, 6, 30), scenarios=[base],
    )  # fmt: skip


class TestTheWeeklyJob:
    def run(self, store, symbols, prices, loader, asof=date(2026, 9, 27)):
        return asyncio.run(
            refresh_valuations(store, symbols, prices, asof=asof, figures_loader=loader)
        )

    def test_a_valued_symbol_is_stored_with_its_engine(self, store):
        result = self.run(store, ["MSFT"], {"MSFT": 516.17}, lambda s, p, **_: figures(s, p))
        assert result.summary() == "1 valued"
        stored = store.load_latest_valuation("MSFT")
        assert stored.method == "dcf"
        assert [v.name for v in stored.value] == ["bear", "base", "bull"]

    def test_no_price_is_skipped_before_any_fetch(self, store):
        called = []
        result = self.run(store, ["MSFT"], {}, lambda *a, **k: called.append(a))
        assert result.skipped == {"MSFT": "no price"} and called == []

    def test_a_loader_that_raises_is_skipped_with_the_reason(self, store):
        def boom(*_a, **_k):
            raise TimeoutError("SEC timed out")

        result = self.run(store, ["MSFT"], {"MSFT": 1.0}, boom)
        assert "timed out" in result.skipped["MSFT"]
        assert store.load_latest_valuation("MSFT") is None

    def test_incomplete_figures_say_what_is_missing(self, store):
        result = self.run(
            store, ["X"], {"X": 5.0}, lambda s, p, **_: figures(s, p, shares=None, net_cash=None)
        )
        assert result.skipped["X"] == "missing shares, balance sheet"

    def test_rerunning_the_same_day_replaces_and_emits_nothing_new(self, store):
        store.save_valuation(build_snapshot(figures(price=400.0), asof=date(2026, 9, 20)))
        first = self.run(store, ["MSFT"], {"MSFT": 516.17}, lambda s, p, **_: figures(s, p))
        again = self.run(store, ["MSFT"], {"MSFT": 516.17}, lambda s, p, **_: figures(s, p))
        assert len(first.events) == 1  # 400 → 516 moved the requirement
        assert again.events == []  # same symbol, same day: deduplicated
        assert len(store.valuation_history("MSFT")) == 2

    def test_the_switch_from_the_old_arithmetic_is_not_news(self, store):
        """Every requirement jumped when discounting arrived; nothing happened."""
        store.save_valuation(legacy_snapshot())
        result = self.run(store, ["MSFT"], {"MSFT": 516.17}, lambda s, p, **_: figures(s, p))
        assert result.events == []
        old = store.load_latest_valuation("MSFT", before=date(2026, 9, 27))
        new = store.load_latest_valuation("MSFT")
        assert old.method == "undiscounted" and new.method == "dcf"
        assert abs(new.base_case().implied_cagr - old.base_case().implied_cagr) > 0.02
        assert shift_event(old, new) is None


class TestTheRangeAcrossMargins:
    def test_generic_then_each_own_margin(self):
        snap = build_snapshot(figures())
        readings, notes = required_range(snap)
        assert [r.label for r in readings][0] == "generic"
        assert {r.margin for r in readings} == {0.25, 0.202, 0.254, 0.370}
        # A higher steady-state margin needs less growth.
        by_margin = sorted(readings, key=lambda r: r.margin)
        assert all(a.required > b.required for a, b in zip(by_margin, by_margin[1:]))
        assert notes == []

    def test_a_negative_own_margin_is_named_and_left_out(self):
        burning = [OwnMargin(kind="fcf", value=-0.015, label="FCF TTM")]
        readings, notes = required_range(build_snapshot(figures(margins=burning)))
        assert [r.label for r in readings] == ["generic"]
        assert any("FCF TTM" in n and "no steady state" in n for n in notes)

    def test_the_range_moves_with_a_new_price(self):
        snap = build_snapshot(figures())
        low = required_range(snap, 300.0)[0][0].required
        high = required_range(snap, 900.0)[0][0].required
        assert high > low

    def test_an_old_row_is_read_the_way_it_was_computed(self):
        readings, _ = required_range(legacy_snapshot())
        assert readings[0].required == pytest.approx(
            legacy_snapshot().base_case().implied_cagr, abs=1e-12
        )


class TestTheScorecard:
    def rows(self, store, snap):
        store.save_valuation(snap)
        card = build_scorecard(store, snap.symbol, consensus_loader=lambda *_: None)
        return {r.label: r for r in card.expectations}

    def test_the_value_range_is_labelled_an_opinion_with_its_assumptions(self, store):
        r = self.rows(store, build_snapshot(figures(), asof=date(2026, 9, 27)))
        value = r["Value range (opinion)"]
        assert value.value.startswith("$") and "base $" in value.value
        assert "discounted at 10%" in value.detail and "fading to 3%" in value.detail
        assert "discounted at 10%" in r["Price requires"].detail
        assert r["Price needs margin"].value.endswith("% FCF")

    def test_a_refused_range_says_why(self, store):
        snap = build_snapshot(figures(margins=[]), asof=date(2026, 9, 27))
        r = self.rows(store, snap)
        assert r["Value range (opinion)"].value == "none"
        assert "no positive margin" in r["Value range (opinion)"].detail

    def test_an_old_row_shows_no_value_range(self, store):
        r = self.rows(store, legacy_snapshot())
        assert "Value range (opinion)" not in r
        assert "undiscounted" in r["Price requires"].detail


class TestThePrice:
    def test_the_broker_close_comes_first(self):
        with patch(
            "advisor.valuation.figures._broker_close",
            return_value=(516.17, "TastyTrade final close 2026-09-25"),
        ):
            assert last_price("msft") == (516.17, "TastyTrade final close 2026-09-25")

    def test_both_sources_down_is_no_price_not_zero(self):
        with (
            patch("advisor.valuation.figures._broker_close", side_effect=ConnectionError("api")),
            patch("yfinance.Ticker", side_effect=RuntimeError("Too Many Requests")),
        ):
            assert last_price("JBL") is None

    def test_no_broker_answer_falls_back_to_yahoo(self):
        import pandas as pd

        frame = pd.DataFrame(
            {"Close": [310.0, 316.74]},
            index=pd.to_datetime(["2026-09-24", "2026-09-25"]),
        )
        with (
            patch("advisor.valuation.figures._broker_close", return_value=None),
            patch("yfinance.Ticker") as ticker,
        ):
            ticker.return_value.history.return_value = frame
            assert last_price("JBL") == (316.74, "Yahoo close 2026-09-25")

    def test_a_zero_or_missing_broker_close_is_no_close(self):
        from types import SimpleNamespace

        from advisor.valuation.figures import _broker_close

        async def session():
            return object()

        for close in (None, 0, -1):
            row = SimpleNamespace(
                symbol="SPCX", close=close, close_price_type="FINAL", summary_date="2026-09-25"
            )

            async def market(_s, equities, row=row):
                return [row]

            with (
                patch("advisor.market.tastytrade_client.get_session", session),
                patch("tastytrade.market_data.get_market_data_by_type", market),
            ):
                assert _broker_close("SPCX") is None

    def test_an_unknown_symbol_is_no_close(self):
        from advisor.valuation.figures import _broker_close

        async def session():
            return object()

        async def market(_s, equities):
            return []

        with (
            patch("advisor.market.tastytrade_client.get_session", session),
            patch("tastytrade.market_data.get_market_data_by_type", market),
        ):
            assert _broker_close("ZZZZ") is None
