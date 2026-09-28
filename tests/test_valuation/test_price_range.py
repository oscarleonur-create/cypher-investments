"""The price-range card: four cases, a verdict, and a rationale whose every
number comes from the snapshot."""

from __future__ import annotations

import re
from datetime import date
from pathlib import Path

import pytest
from advisor.daemon.store import DaemonStore
from advisor.valuation.implied import build_snapshot
from advisor.valuation.rationale import price_range

from tests.test_valuation.test_engine_wiring import figures, legacy_snapshot


def card(**overrides):
    return price_range(build_snapshot(figures(**overrides), asof=date(2026, 9, 27)))


class TestTheCases:
    def test_bear_base_bull_and_the_market(self):
        c = card()
        assert [x.name for x in c.cases] == ["bear", "base", "bull", "market"]
        market = c.cases[-1]
        assert market.value_per_share == c.price and market.upside == 0.0
        assert market.held_years == 3 and market.margin is not None

    def test_the_market_margin_is_what_makes_the_value_the_price(self):
        """Run forward on the market case's assumptions, the engine returns the price."""
        from advisor.valuation.dcf import path_for, per_share, project

        f = figures()
        c = card()
        m = c.cases[-1]
        p = project(f.revenue_base, f.start_margin, path_for(m.growth, m.held_years, m.margin))
        assert per_share(p.enterprise_value, f.net_cash, f.shares) == pytest.approx(
            c.price, rel=1e-5
        )

    def test_no_growth_comparison_means_no_market_case(self):
        c = card(revenue_growth=None)
        assert [x.name for x in c.cases] == []
        assert any("cannot be read" in line for line in c.rationale)

    def test_a_refused_range_keeps_the_market_case_and_says_why(self):
        c = card(margins=[])
        assert [x.name for x in c.cases] == ["market"]
        assert c.refused and c.verdict is None
        assert any(line.startswith("No value range:") for line in c.rationale)


class TestTheVerdict:
    @pytest.mark.parametrize(
        "price,verdict",
        [(5000.0, "above_bull"), (1.0, "below_bear")],
    )
    def test_the_extremes(self, price, verdict):
        assert card(price=price).verdict == verdict

    def test_between_the_cases(self):
        c = card()
        bear, base, bull = (
            next(x for x in c.cases if x.name == n) for n in ("bear", "base", "bull")
        )
        mid_low = (bear.value_per_share + base.value_per_share) / 2
        mid_high = (base.value_per_share + bull.value_per_share) / 2
        assert card(price=mid_low).verdict == "above_bear"
        assert card(price=mid_high).verdict == "above_base"

    def test_exactly_at_the_bull_is_not_above_it(self):
        bull = next(x for x in card().cases if x.name == "bull").value_per_share
        assert card(price=bull).verdict != "above_bull"


class TestTheRationale:
    def test_msft_reads_as_a_bet_on_margins_it_has_not_reported(self):
        c = card()  # MSFT at $516.17: needs ~42%, best filed 37%
        text = " ".join(c.rationale)
        assert "more than it has ever reported" in text
        assert c.verdict == "above_bull"
        assert "bet on economics the company has not yet reported" in text

    def test_every_number_in_it_is_in_the_snapshot_or_the_cases(self):
        c = card()
        known = {f"{c.price:,.2f}"} | {f"{x.value_per_share:,.2f}" for x in c.cases}
        for money in re.findall(r"\$([\d,]+\.\d{2})(?!bn)", " ".join(c.rationale)):
            assert money in known, money

    def test_all_negative_margins_are_named(self):
        from advisor.valuation.models import OwnMargin

        burning = [
            OwnMargin(kind="fcf", value=-2.0, label="FCF, 6 months"),
            OwnMargin(kind="nopat", value=-0.132, label="operating, 6 months"),
        ]
        text = " ".join(card(margins=burning, start_margin=-2.0).rationale)
        assert "Every margin it has filed is negative" in text and "-200.0% FCF" in text

    def test_net_debt_is_called_debt(self):
        assert "debt" in card(net_cash=-2.02e9).rationale[0]

    def test_a_stale_balance_sheet_says_so(self):
        snap = build_snapshot(figures(balance_asof=date(2025, 12, 31)), asof=date(2026, 9, 27))
        c = price_range(snap, today=date(2026, 9, 27))
        assert c.stale and "may describe a different business" in c.rationale[-1]


@pytest.fixture
def store(tmp_path: Path):
    s = DaemonStore(tmp_path / "research.db")
    yield s
    s.close()


class TestTheEndpoint:
    @pytest.fixture(autouse=True)
    def _fresh_settings(self):
        from advisor.api.routers import daemon
        from advisor.research.config import get_settings

        daemon._live_ranges.clear()
        yield
        daemon._live_ranges.clear()
        get_settings.cache_clear()  # never leave the next test on a tmp database

    def client(self, tmp_path, monkeypatch):
        from advisor.research.config import get_settings
        from fastapi.testclient import TestClient

        monkeypatch.setenv("ADVISOR_RESEARCH_DB_PATH", str(tmp_path / "research.db"))
        get_settings.cache_clear()
        from advisor.api.app import create_app

        return TestClient(create_app())

    def test_a_stored_engine_snapshot_is_served_without_the_network(self, tmp_path, monkeypatch):
        s = DaemonStore(tmp_path / "research.db")
        s.save_valuation(build_snapshot(figures(), asof=date(2026, 9, 27)))
        s.close()

        def no_network(*_a, **_k):
            raise AssertionError("must not compute live")

        monkeypatch.setattr("advisor.valuation.figures.load_figures", no_network)
        r = self.client(tmp_path, monkeypatch).get("/api/daemon/symbol/msft/price-range")
        assert r.status_code == 200
        assert r.json()["live"] is False and len(r.json()["cases"]) == 4

    def test_an_old_row_is_recomputed_live_and_not_stored(self, tmp_path, monkeypatch):
        s = DaemonStore(tmp_path / "research.db")
        s.save_valuation(legacy_snapshot())
        s.close()
        monkeypatch.setattr(
            "advisor.valuation.figures.load_figures", lambda sym, *a, **k: figures(sym)
        )
        r = self.client(tmp_path, monkeypatch).get("/api/daemon/symbol/MSFT/price-range")
        assert r.status_code == 200 and r.json()["live"] is True
        s = DaemonStore(tmp_path / "research.db")
        try:
            assert s.load_latest_valuation("MSFT").method == "undiscounted"  # untouched
        finally:
            s.close()

    def test_nothing_to_value_is_a_404_with_the_reason(self, tmp_path, monkeypatch):
        monkeypatch.setattr(
            "advisor.valuation.figures.load_figures",
            lambda sym, *a, **k: figures(sym, shares=None, net_cash=None),
        )
        r = self.client(tmp_path, monkeypatch).get("/api/daemon/symbol/ZZZZ/price-range")
        assert r.status_code == 404 and "share count" in r.json()["detail"]
        assert "rate-limiting" in r.json()["detail"]

    def test_a_live_range_is_kept_an_hour_and_refresh_recomputes(self, tmp_path, monkeypatch):
        calls = []

        def load(sym, *a, **k):
            calls.append(sym)
            return figures(sym)

        monkeypatch.setattr("advisor.valuation.figures.load_figures", load)
        c = self.client(tmp_path, monkeypatch)
        for _ in range(3):
            assert c.get("/api/daemon/symbol/MSFT/price-range").status_code == 200
        assert calls == ["MSFT"]
        assert c.get("/api/daemon/symbol/MSFT/price-range?refresh=true").status_code == 200
        assert calls == ["MSFT", "MSFT"]
