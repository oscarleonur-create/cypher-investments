"""The daily news sweep over every watched name (user decision, 2026-10-04)."""

from __future__ import annotations

from datetime import datetime, timedelta

import pytest
from advisor.daemon import market_calendar as mc
from advisor.daemon.store import DaemonStore
from advisor.news import google_news
from advisor.news import sweep as S
from advisor.news.models import EntityMatch, MatchMethod, SourceItem, SourceTier

NOW = datetime(2026, 10, 5, 8, 0, tzinfo=mc.MARKET_TZ)


def item(symbol, title, hours=3, tier=SourceTier.AGGREGATOR):
    return SourceItem(
        tier=tier,
        provider="reuters.com",
        url=f"https://example.com/{symbol}/{abs(hash(title))}",
        title=title,
        published_at=NOW - timedelta(hours=hours),
        entity=EntityMatch(symbol=symbol, method=MatchMethod.COMPANY_NAME),
    )


@pytest.fixture
def store(tmp_path):
    s = DaemonStore(tmp_path / "research.db")
    yield s
    s.close()


class Fakes:
    def __init__(self, google=None, tavily=None, fail=()):
        self.google, self.tavily, self.fail = google or {}, tavily or {}, set(fail)
        self.google_calls, self.tavily_calls = [], []

    def search(self, symbol, query, **kw):
        self.google_calls.append((symbol, query, kw.get("days")))
        if ("google", symbol) in self.fail:
            raise RuntimeError("rss down")
        return list(self.google.get(symbol, []))

    async def explain(self, store, symbol, *, reason, company_name, days):
        self.tavily_calls.append((symbol, reason, days))
        if ("tavily", symbol) in self.fail:
            raise RuntimeError("tavily 429")
        items = list(self.tavily.get(symbol, []))
        for i in items:
            store.save_source_item(i)
        return items

    @staticmethod
    def verify(items, store):
        return items


def run(store, symbols, fakes, **kw):
    return S.sweep_universe(
        store, symbols, names=lambda s: f"{s} Corp", websites=lambda s: None,
        search=fakes.search, explain=fakes.explain, verify=fakes.verify, **kw,
    )  # fmt: skip


def news_events(store, symbol):
    return [e for e in store.recent_events(symbol=symbol, since=NOW - timedelta(days=3))
            if e.kind == "NEWS_CONTEXT"]  # fmt: skip


def test_every_name_gets_google_and_tavily_archived_as_tier_c_context(store):
    f = Fakes(google={"AMZN": [item("AMZN", "Amazon wins cloud deal")]},
              tavily={"AMZN": [item("AMZN", "Amazon stock rises")]})  # fmt: skip
    r = run(store, ["AMZN", "INTC"], f)
    assert r.found == {"AMZN": 2, "INTC": 0} and r.tavily == 2
    assert [c[0] for c in f.google_calls] == ["AMZN", "INTC"]
    assert all(days == S.SWEEP_DAYS for *_, days in f.google_calls)
    assert [(s, reason) for s, reason, _ in f.tavily_calls] == [("AMZN", "DAILY_NEWS"),
                                                               ("INTC", "DAILY_NEWS")]  # fmt: skip
    evs = news_events(store, "AMZN")
    assert len(evs) == 2 and all(e.tier.value == "C" for e in evs)
    assert {e.payload["explains"] for e in evs} == {"DAILY_NEWS"}
    assert "nothing for INTC" in r.detail()


def test_a_primary_item_is_still_only_context(store):
    f = Fakes(google={"CRDO": [item("CRDO", "Credo announces results", tier=SourceTier.PRIMARY)]})
    run(store, ["CRDO"], f)
    [e] = news_events(store, "CRDO")
    assert e.tier.value == "C"


def test_the_budget_runs_out_on_the_tail_never_the_book(store):
    f = Fakes()
    r = run(store, ["SPCX", "CRDO", "AMZN", "INTC"], f, budget=2)
    assert [c[0] for c in f.tavily_calls] == ["SPCX", "CRDO"]
    assert r.over_budget == ["AMZN", "INTC"] and r.tavily == 2
    assert len(f.google_calls) == 4  # Google is free: every name
    assert "over budget: AMZN, INTC" in r.detail()


def test_one_failing_name_does_not_stop_the_rest(store):
    f = Fakes(google={"META": [item("META", "Meta ships glasses")]},
              fail={("google", "AMD"), ("tavily", "AMD")})  # fmt: skip
    r = run(store, ["AMD", "META"], f)
    assert r.found == {"AMD": 0, "META": 1}
    assert len(r.errors) == 2 and r.errors[0].startswith("AMD: google news failed")


def test_a_name_lookup_failure_still_searches_by_ticker(store):
    f = Fakes()

    def boom(symbol):
        raise RuntimeError("yahoo down")

    r = S.sweep_universe(store, ["WOLF"], names=boom, websites=boom, search=f.search,
                         explain=f.explain, verify=f.verify)  # fmt: skip
    assert f.google_calls and f.google_calls[0][0] == "WOLF"
    assert r.errors and "name lookup failed" in r.errors[0]


def test_running_twice_archives_once(store):
    same = item("NBIS", "Nebius signs Microsoft deal")
    f = Fakes(google={"NBIS": [same]}, tavily={"NBIS": [same]})
    run(store, ["NBIS"], f)
    run(store, ["NBIS"], f)
    assert len(news_events(store, "NBIS")) == 1


def test_an_empty_universe_does_nothing(store):
    r = run(store, [], Fakes())
    assert r.found == {} and r.tavily == 0 and r.detail().startswith("0 names")


def test_items_go_through_the_date_check(store):
    """Whatever verify drops is neither archived nor emitted."""
    f = Fakes(google={"PENG": [item("PENG", "Penguin Solutions stock jumps")]})
    f.verify = staticmethod(lambda items, store: [])
    r = run(store, ["PENG"], f)
    assert r.found == {"PENG": 0} and news_events(store, "PENG") == []


def test_google_keeps_the_strongest_eight_per_name(store):
    by_ticker = [item("AMZN", f"AMZN ticker item {n}", hours=n) for n in range(1, 6)]
    for i in by_ticker:
        i.entity = EntityMatch(symbol="AMZN", method=MatchMethod.TICKER_TOKEN)
    by_name = [item("AMZN", f"Amazon item {n}", hours=10 + n) for n in range(1, 8)]
    f = Fakes(google={"AMZN": by_ticker + by_name})
    r = run(store, ["AMZN"], f, budget=0)
    assert r.found == {"AMZN": S.GOOGLE_PER_NAME}
    titles = [e.payload["title"] for e in news_events(store, "AMZN")]
    assert sum(t.startswith("Amazon item") for t in titles) == 7  # every name match kept
    assert sum(t.startswith("AMZN ticker") for t in titles) == 1  # the newest ticker match


def test_wolf_is_a_word_and_matches_on_the_company_name_only():
    from advisor.news.entities import is_ambiguous, resolve_entity

    assert is_ambiguous("WOLF")
    hockey = resolve_entity("WOLF", text="HARTFORD WOLF PACK ANNOUNCE ROSTER",
                            company_name="Wolfspeed, Inc.")  # fmt: skip
    assert not hockey.resolved
    ok = resolve_entity("WOLF", text="Wolfspeed jumps 11%", company_name="Wolfspeed, Inc.")
    assert ok.resolved
    assert "WOLF" not in google_news.who("WOLF", "Wolfspeed, Inc.")


def test_the_query_names_the_company_without_keywords():
    q = google_news.who("AMZN", "Amazon.com, Inc.")
    assert "AMZN" in q and "bankruptcy" not in q
    assert google_news.distress_query("AMZN", "Amazon.com, Inc.").startswith(q + " (")
