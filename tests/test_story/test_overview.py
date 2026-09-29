"""The company in a few bullets at the top of the ticker page.

User, 2026-09-29: "no tenemos un resumen al principio que dé un outline de la
empresa ... unos bullets que den una visión general del estado de la compañía
a día de hoy".
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from types import SimpleNamespace

import pytest
from advisor.daemon.market_calendar import MARKET_TZ
from advisor.entry.health import Health
from advisor.story import overview as O

NOW = datetime(2026, 9, 29, 9, 0, tzinfo=MARKET_TZ)


def mdb_health(**kw):
    base = dict(
        symbol="MDB", computed_on=NOW.date(), period_end=date(2026, 7, 31), source="SEC XBRL",
        revenue_ttm=2.78e9, growth=0.255, growth_two_q=0.228, growth_year=0.219,
        fcf_margin=0.237, fcf_label="FCF", fcf_median=0.069, fcf_median_label="median",
        net_cash=2.41e9, balance_asof=date(2026, 7, 31), market_cap=27e9, dilution=0.011,
    )  # fmt: skip
    base.update(kw)
    return Health(**base)


def proposal(action="ADD", **features):
    f = {"ps": 10.07, "ps_median": 10.49, "ps_percentile": 0.448, "in_zone": True}
    f.update(features)
    return {
        "action": action,
        "built_at": "2026-09-28T14:58:00-04:00",
        "features": f,
        "triggers": ["scanner setup today: 2026-09-28:C:MDB"],
        "reasons": [
            {"text": "Why ADD: fell -18.5% today (-4.5σ) inside the zone"},
            {"text": "next results 2026-12-01 (in 45 sessions)"},
        ],
        "exits": [],
    }


def judgment(direction="NEGATIVE", materiality="MEDIUM", title="CEO leaves for Meta"):
    return SimpleNamespace(
        published_at=NOW - timedelta(days=1), title=title, provider="reuters.com",
        direction=SimpleNamespace(value=direction), materiality=SimpleNamespace(value=materiality),
    )  # fmt: skip


def position(qty=1, cost=339.38, pnl=1.0):
    return SimpleNamespace(quantity=qty, cost_basis=cost * qty, unrealized_pnl=pnl,
                           signed_notional=(cost * qty + pnl), underlying="MDB")  # fmt: skip


def topics(ov):
    return {b.topic: b for b in ov.bullets}


def test_the_whole_outline_in_order_each_with_a_source():
    ov = O.assemble(
        "mdb", NOW, business=("A document database platform.", "deep research, 2026-09-28"),
        health=mdb_health(), proposal=proposal(), judgments=[judgment(), judgment()],
        held=[position()], net_liq=7700.0, book_as_of=NOW.date(),
    )  # fmt: skip
    assert [b.topic for b in ov.bullets] == [
        "Business", "Growth", "Cash", "Filings", "Valuation", "News", "Position", "Call",
    ]  # fmt: skip
    assert all(b.source for b in ov.bullets) and ov.gaps == []
    t = topics(ov)
    assert t["Growth"].text == (
        "Revenue $2.78bn over 12 months, +25.5% a year, accelerating (from +21.9% a year before)."
    )
    assert t["Growth"].tone == "pos"
    assert "net cash $2.41bn (8.9% of market cap)" in t["Cash"].text
    assert t["Filings"].text == "No concern flagged in the filings to 2026-07-31."
    assert "at the 45th percentile of its last 2 years: in its zone" in t["Valuation"].text
    assert t["News"].tone == "neg" and t["News"].text.endswith("— and 1 more this week")
    assert t["Position"].text.startswith("You hold 1 share,")  # singular
    assert t["Call"].text.startswith("ADD: fell -18.5% today")
    assert "next results 2026-12-01" in t["Call"].text


def test_a_scanner_id_in_a_call_is_named_as_its_setup():
    p = proposal()
    p["reasons"] = []
    assert O.call_bullet(p).text.startswith("ADD: scanner setup today: setup C")


def test_an_exit_says_why_from_the_exit_call():
    p = proposal("EXIT")
    p["reasons"], p["triggers"] = [], []
    p["exits"] = [{"why": "past its stop: -34.3% from your average cost $5.63"}]
    b = O.call_bullet(p)
    assert b.tone == "neg" and b.text == "EXIT: past its stop: -34.3% from your average cost $5.63."


@pytest.mark.parametrize(
    "kw,tone",
    [({}, "pos"), ({"growth": 0.20, "growth_two_q": 0.25}, "warn"),
     ({"growth": -0.05}, "neg"), ({"growth": 0.20, "growth_two_q": 0.20}, "neutral")],
)  # fmt: skip
def test_growth_tone_follows_the_trend(kw, tone):
    assert O.health_bullets(mdb_health(**kw))[0].tone == tone


def test_concerns_turn_the_filings_bullet_to_a_warning():
    b = topics(O.assemble("TE", NOW, health=mdb_health(fcf_margin=-0.185, dilution=0.796)))
    assert b["Cash"].tone == "neg"
    assert b["Filings"].tone == "warn"
    assert "burning cash, diluting more than 5% a year" in b["Filings"].text


def test_a_bank_is_not_described_by_its_cash_flow():
    b = topics(O.assemble("NU", NOW, health=mdb_health(financial=True, fcf_margin=-0.127)))
    assert b["Cash"].text.startswith("A financial company") and "-12.7" not in b["Cash"].text


def test_valuation_above_the_zone_and_rich_for_itself():
    b = O.valuation_bullet(proposal(in_zone=False, ps=12.8, ps_median=5.4, ps_percentile=0.9))
    assert b.tone == "warn" and "above its zone" in b.text and "90th percentile" in b.text
    assert O.valuation_bullet(proposal(in_zone=False, ps_percentile=0.5)).tone == "neutral"
    assert O.valuation_bullet({"features": {}}) is None


def test_low_materiality_news_does_not_colour_the_outline():
    assert O.news_bullet([judgment(materiality="LOW")], 7).tone == "neutral"
    assert O.news_bullet([judgment("POSITIVE", "HIGH")], 7).tone == "pos"


def test_nothing_on_file_is_said_not_left_out():
    ov = O.assemble("ZZZ", NOW)
    t = topics(ov)
    assert set(t) == {"News", "Position"}
    assert t["News"].text == "No news about the company judged in the last 7 days."
    assert t["Position"].text == "Not held."
    assert len(ov.gaps) == 2  # no health, no proposal: each says why


def test_a_losing_position_reads_negative():
    b = O.position_bullet([position(qty=50, cost=5.63, pnl=-102.0)], 7700.0, NOW.date())
    assert b.tone == "neg" and "-36.2% ($-102)" in b.text


def test_first_sentence_does_not_stop_at_inc():
    text = "Meta Platforms, Inc. engages in building products. It was founded in 2004."
    assert O.first_sentence(text) == "Meta Platforms, Inc. engages in building products."
    long = "Word " * 100
    assert O.first_sentence(long, limit=40).endswith("…")


def test_load_overview_from_the_store_without_network(tmp_path, monkeypatch):
    from advisor.daemon.book import BookSnapshot
    from advisor.daemon.store import DaemonStore
    from advisor.entry import health as H

    db = tmp_path / "research.db"
    H.save_health(db, mdb_health())
    monkeypatch.setattr(H, "_COMPUTED", {("MDB", NOW.date())})  # no SEC call
    store = DaemonStore(db)
    store.save_book(BookSnapshot(net_liq=7700.0))
    ov = O.load_overview(store, "mdb", NOW, profile=lambda *a: ("A database.", "test"))
    store.close()
    t = topics(ov)
    assert t["Business"].text == "A database." and "+25.5%" in t["Growth"].text
    assert t["Position"].text == "Not held."
    assert ov.gaps == ["no proposal on file: evaluate the name to place its price and call"]


def test_the_endpoint_refuses_something_that_is_not_a_symbol():
    from advisor.api.routers import daemon as router
    from fastapi import HTTPException

    with pytest.raises(HTTPException):
        router.symbol_overview("../etc/passwd")
