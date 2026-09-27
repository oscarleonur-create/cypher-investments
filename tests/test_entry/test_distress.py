"""Distress news read for an exit: outlets counted without the model, citations checked."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import pytest
from advisor.daemon import market_calendar as mc
from advisor.daemon.models import Event, EventSource, EventTier
from advisor.daemon.store import DaemonStore
from advisor.entry.distress import (
    STICKY_DAYS,
    DistressReading,
    Draft,
    Item,
    Situation,
    Verdict,
    check,
    distress_all,
    latest_distress,
    news_items,
    outlet,
    read_distress,
    reading_event,
)
from advisor.entry.exits import exit_calls
from advisor.entry.proposal import Action, build_proposal
from advisor.entry.sheet import EventLine, Holding, Move, Sheet

NOW = datetime(2026, 9, 26, 8, 15, tzinfo=mc.MARKET_TZ)


@pytest.fixture
def store(tmp_path: Path):
    s = DaemonStore(tmp_path / "research.db")
    yield s
    s.close()


def news(
    store, title, provider, days_ago=1, symbol="TE", kind="NEWS_CONTEXT", verified="CONFIRMED"
):
    store.emit(
        Event(
            ts=NOW - timedelta(days=days_ago),
            source=EventSource.CALENDAR,
            kind=kind,
            tier=EventTier.C,
            symbol=symbol,
            dedup_key=f"{provider}:{title}",
            payload={
                "title": title,
                "provider": provider,
                "url": f"https://{provider}/x",
                "published_at": (NOW - timedelta(days=days_ago)).isoformat(),
                "verified": verified,
            },
        )
    )


def says(verdict, situation, ids, reason=""):
    def complete(system, user):
        return Draft(verdict=verdict, situation=situation, fact_ids=ids, reason=reason)

    return complete


class TestOutlets:
    @pytest.mark.parametrize(
        "provider,expected",
        [
            ("www.fool.com", "fool"),
            ("The Motley Fool", "fool"),
            ("Simply Wall St.", "simplywall"),
            ("simplywall.st", "simplywall"),
            ("finance.yahoo.com", "yahoo"),
            ("Yahoo Finance", "yahoo"),
            ("https://www.reuters.com/markets/x", "reuters"),
            ("Reuters", "reuters"),
            ("bbc.co.uk", "bbc"),
            ("Zacks", "zacks"),
        ],
    )
    def test_one_outlet_is_counted_once(self, provider, expected):
        assert outlet(provider) == expected


ITEMS = [
    Item(id="N1", title="T1 Energy hires restructuring advisers", provider="reuters.com", date="d"),
    Item(id="N2", title="T1 Energy weighs Chapter 11", provider="Bloomberg", date="d"),
    Item(id="N3", title="Could T1 Energy go bankrupt?", provider="www.fool.com", date="d"),
]


class TestCheck:
    def test_invented_ids_are_dropped(self):
        d, problems = check(
            Draft(verdict="EXIT_GRADE", situation="BANKRUPTCY", fact_ids=["N1", "N9"]), ITEMS
        )
        assert d.fact_ids == ["N1"] and problems

    def test_a_verdict_with_no_valid_citation_is_none(self):
        d, _ = check(Draft(verdict="EXIT_GRADE", situation="BANKRUPTCY", fact_ids=["N9"]), ITEMS)
        assert d.verdict is Verdict.NONE and d.situation is Situation.NONE and d.fact_ids == []

    def test_none_carries_no_citations(self):
        d, _ = check(Draft(verdict="NONE", situation="BANKRUPTCY", fact_ids=["N1"]), ITEMS)
        assert d.situation is Situation.NONE and d.fact_ids == []

    def test_graded_without_a_situation_is_other(self):
        d, _ = check(Draft(verdict="EXIT_GRADE", situation="NONE", fact_ids=["N1"]), ITEMS)
        assert d.situation is Situation.OTHER

    def test_a_reason_with_a_number_is_dropped(self):
        """The model may not introduce a figure no source carries."""
        d, problems = check(
            Draft(
                verdict="EXIT_GRADE",
                situation="DEFAULT",
                fact_ids=["N1"],
                reason="It missed a $40 million coupon",
            ),
            ITEMS,
        )
        assert d.reason == "" and any("number" in p for p in problems)

    def test_duplicate_ids_counted_once(self):
        d, _ = check(
            Draft(verdict="EXIT_GRADE", situation="BANKRUPTCY", fact_ids=["N1", "N1", "N2"]), ITEMS
        )
        assert d.fact_ids == ["N1", "N2"]


class TestRead:
    def test_no_news_no_model_call(self, store):
        def boom(s, u):
            raise AssertionError("must not be called")

        r = read_distress(store, "TE", NOW, complete=boom)
        assert r.status == "NO_ITEMS"

    def test_outlets_come_from_the_cited_items_not_the_model(self, store):
        news(store, "T1 Energy hires restructuring advisers", "www.reuters.com")
        news(store, "T1 Energy weighs Chapter 11 filing", "Bloomberg", days_ago=2)
        news(store, "Solar stocks slide", "www.fool.com", days_ago=3)
        items = news_items(store, "TE", NOW)
        ids = [i.id for i in items if "T1" in i.title]
        r = read_distress(
            store, "TE", NOW, complete=says("EXIT_GRADE", "BANKRUPTCY", ids, "Restructuring.")
        )
        assert r.status == "OK" and r.verdict is Verdict.EXIT_GRADE
        assert r.outlets == ["bloomberg", "reuters"] and len(r.cited) == 2

    def test_news_older_than_a_week_is_not_read(self, store):
        news(store, "Old story", "reuters.com", days_ago=8)
        assert read_distress(store, "TE", NOW, complete=says("NONE", "NONE", [])).status == (
            "NO_ITEMS"
        )

    def test_the_same_title_from_two_pulls_is_one_item(self, store):
        news(store, "T1 Energy weighs Chapter 11", "Bloomberg", days_ago=1)
        news(store, "T1 Energy weighs Chapter 11", "bloomberg.com", days_ago=2)
        assert len(news_items(store, "TE", NOW)) == 1

    def test_model_failure_is_unavailable(self, store):
        news(store, "x", "reuters.com")

        def boom(s, u):
            raise TimeoutError("openrouter")

        assert read_distress(store, "TE", NOW, complete=boom).status == "UNAVAILABLE"

    def test_a_malformed_answer_is_invalid(self, store):
        news(store, "x", "reuters.com")
        assert read_distress(store, "TE", NOW, complete=lambda s, u: "text").status == "INVALID"

    def test_other_symbols_news_is_not_read(self, store):
        news(store, "Some other company files for bankruptcy", "reuters.com", symbol="CCXI")
        assert read_distress(store, "TE", NOW, complete=says("NONE", "NONE", [])).status == (
            "NO_ITEMS"
        )


def graded(outlets=("reuters", "bloomberg"), as_of=NOW, verdict="EXIT_GRADE"):
    return DistressReading(
        symbol="TE",
        as_of=as_of,
        status="OK",
        verdict=verdict,
        situation="BANKRUPTCY" if verdict != "NONE" else "NONE",
        cited=[
            Item(id=f"N{k}", title=f"story {k}", provider=o, date="2026-09-25")
            for k, o in enumerate(outlets if verdict != "NONE" else ())
        ],
        outlets=sorted(outlets) if verdict != "NONE" else [],
        reason="Restructuring advisers hired.",
    )


class TestStored:
    def test_reading_event_is_idempotent(self, store):
        assert store.emit(reading_event(graded())) is True
        assert store.emit(reading_event(graded())) is False

    def test_an_exit_grade_reading_outlives_a_later_none(self, store):
        """A bankruptcy report does not expire because the next sweep found nothing new."""
        store.emit(reading_event(graded(as_of=NOW - timedelta(days=3))))
        store.emit(reading_event(graded(as_of=NOW, verdict="NONE")))
        r = latest_distress(store, "TE", NOW)
        assert r.verdict is Verdict.EXIT_GRADE

    def test_it_expires_after_the_sticky_window(self, store):
        old = NOW - timedelta(days=STICKY_DAYS + 1)
        e = reading_event(graded(as_of=old)).model_copy(update={"ts": old})
        store.emit(e)
        assert latest_distress(store, "TE", NOW) is None

    def test_without_an_exit_grade_the_newest_wins(self, store):
        store.emit(reading_event(graded(as_of=NOW - timedelta(days=2), verdict="WATCH")))
        store.emit(reading_event(graded(as_of=NOW, verdict="NONE")))
        assert latest_distress(store, "TE", NOW).verdict is Verdict.NONE


def held_sheet(distress=None, filings=()):
    return Sheet(
        symbol="TE",
        built_at=NOW,
        move=Move(price=3.78, asof=NOW.date(), day=0.0, sigma=0.069, z=0.0),
        holding=Holding(quantity=50, weight=0.024, unrealized=0.0, cost=3.80),
        distress=distress.model_dump(mode="json") if distress else None,
        filings=list(filings),
    )


class TestExitFromNews:
    def test_two_outlets_is_an_unconfirmed_exit(self):
        """User decision 2026-09-26: 2+ independent outlets, no filing yet → EXIT."""
        (c,) = exit_calls(held_sheet(graded()), net_liq=7_957.51)[0]
        assert c.action == "EXIT" and c.rule == "news (unconfirmed)" and c.shares == 50
        assert "2 independent outlets report bankruptcy or a restructuring" in c.why
        assert "no SEC filing confirms it yet" in c.why
        assert [e.text for e in c.evidence] == ["2026-09-25 story 0", "2026-09-25 story 1"]

    def test_one_outlet_is_a_review(self):
        (c,) = exit_calls(held_sheet(graded(outlets=("reuters",))), net_liq=7_957.51)[0]
        assert c.action == "REVIEW" and c.shares is None
        assert "one outlet is a REVIEW" in c.why

    @pytest.mark.parametrize("verdict", ["WATCH", "NONE"])
    def test_watch_and_none_are_not_calls(self, verdict):
        assert exit_calls(held_sheet(graded(verdict=verdict)), net_liq=7_957.51)[0] == []

    def test_a_filing_confirms_it(self):
        f = EventLine(ts=NOW, kind="FILING_BANKRUPTCY", tier="A", text="8-K", items=["1.03"])
        calls, _, _ = exit_calls(held_sheet(graded(), filings=[f]), net_liq=7_957.51)
        news_call = next(c for c in calls if c.rule.startswith("news"))
        assert news_call.rule == "news" and "confirmed by a filing" in news_call.why

    def test_the_proposal_exits_with_the_sources(self):
        p = build_proposal(held_sheet(graded()), net_liq=7_957.51)
        assert p.action is Action.EXIT
        assert p.reasons[0].source == "exit rule: news (unconfirmed)"
        assert "Reading: Restructuring advisers hired." in p.reasons[0].text


class TestSweep:
    def test_every_name_is_read_and_graded_ones_stored(self, store):
        searched = []

        def searcher(st, sym, company=None, website=None):
            searched.append(sym)
            return 1

        def reader(st, sym, now, company=None):
            return (
                graded()
                if sym == "TE"
                else DistressReading(symbol=sym, as_of=now, status="NO_ITEMS")
            )

        readings, errors = distress_all(
            store,
            ["TE", "CRDO"],
            NOW,
            searcher=searcher,
            reader=reader,
            names=lambda s: None,
            websites=lambda s: None,
        )
        assert searched == ["TE", "CRDO"] and errors == []
        assert [r.status for r in readings] == ["OK", "NO_ITEMS"]
        assert latest_distress(store, "TE", NOW).verdict is Verdict.EXIT_GRADE
        assert latest_distress(store, "CRDO", NOW) is None

    def test_a_failed_search_still_reads_what_is_archived(self, store):
        def searcher(st, sym, company=None, website=None):
            raise TimeoutError("tavily")

        readings, errors = distress_all(
            store,
            ["TE"],
            NOW,
            searcher=searcher,
            reader=lambda st, sym, now, company=None: graded(),
            names=lambda s: None,
            websites=lambda s: None,
        )
        assert readings[0].status == "OK" and "distress search failed: tavily" in errors[0]

    def test_an_unavailable_model_is_an_error(self, store):
        readings, errors = distress_all(
            store,
            ["TE"],
            NOW,
            search=False,
            reader=lambda st, sym, now, company=None: DistressReading(
                symbol=sym, as_of=now, status="UNAVAILABLE"
            ),
            names=lambda s: None,
            websites=lambda s: None,
        )
        assert "distress reading UNAVAILABLE" in errors[0]
        assert latest_distress(store, "TE", NOW) is None


class TestWires:
    @pytest.mark.parametrize(
        "provider",
        [
            "businesswire.com",
            "PR Newswire",
            "www.globenewswire.com",
            "accesswire.com",
            "EIN Presswire",
        ],
    )
    def test_every_press_release_wire_is_one_outlet(self, provider):
        """A wire distributes whoever pays, law firms' investor alerts included."""
        assert outlet(provider) == "wire"

    def test_three_wires_are_a_review_not_an_exit(self):
        r = graded(outlets=("businesswire.com", "prnewswire.com", "globenewswire.com"))
        r = r.model_copy(update={"outlets": sorted({outlet(i.provider) for i in r.cited})})
        (c,) = exit_calls(held_sheet(r), net_liq=7_957.51)[0]
        assert r.outlets == ["wire"] and c.action == "REVIEW"


def own_word(outlets=("t1energy.com",)):
    r = graded(outlets=outlets)
    cited = [i.model_copy(update={"issuer": i.provider == "t1energy.com"}) for i in r.cited]
    return r.model_copy(update={"cited": cited})


class TestCompanyStatement:
    def test_the_company_s_own_site_is_an_exit_on_its_own(self):
        """User decision 2026-09-27: the company's own word needs no second outlet."""
        (c,) = exit_calls(held_sheet(own_word()), net_liq=7_957.51)[0]
        assert c.action == "EXIT" and c.rule == "news (company statement)" and c.shares == 50
        assert "the company itself reports bankruptcy or a restructuring" in c.why
        assert "t1energy.com" in c.why and "no SEC filing confirms it yet" in c.why

    def test_a_filing_still_labels_it_confirmed(self):
        f = EventLine(ts=NOW, kind="FILING_BANKRUPTCY", tier="A", text="8-K", items=["1.03"])
        calls, _, _ = exit_calls(held_sheet(own_word(), filings=[f]), net_liq=7_957.51)
        assert next(c for c in calls if c.rule.startswith("news")).rule == "news"

    def test_one_outlet_that_is_not_the_company_stays_a_review(self):
        (c,) = exit_calls(held_sheet(graded(outlets=("reuters",))), net_liq=7_957.51)[0]
        assert c.action == "REVIEW" and not graded(outlets=("reuters",)).company_said

    def test_the_issuer_flag_reaches_the_reading(self, store):
        store.emit(
            Event(
                ts=NOW - timedelta(days=1),
                source=EventSource.CALENDAR,
                kind="NEWS_CONTEXT",
                tier=EventTier.C,
                symbol="TE",
                dedup_key="own",
                payload={
                    "title": "T1 Energy files voluntary Chapter 11 petitions",
                    "provider": "t1energy.com",
                    "published_at": (NOW - timedelta(days=1)).isoformat(),
                    "issuer": True,
                    "verified": "CONFIRMED",
                },
            )
        )
        seen = {}

        def complete(system, user):
            seen["user"] = user
            return Draft(verdict="EXIT_GRADE", situation="BANKRUPTCY", fact_ids=["N1"])

        r = read_distress(store, "TE", NOW, company="T1 Energy", complete=complete)
        assert r.company_said and r.cited[0].issuer
        assert "(t1energy.com, the company itself)" in seen["user"]


class TestGoogleInTheSweep:
    def test_the_sweep_adds_one_free_google_query_and_archives_it(self, store, monkeypatch):
        from advisor.entry import distress
        from advisor.news import ingest
        from advisor.news.models import EntityMatch, MatchMethod, SourceItem, SourceTier

        async def no_tavily(st, sym, **kw):
            return []

        monkeypatch.setattr(ingest, "explain_symbol", no_tavily)
        asked = {}

        def google(sym, query, *, company_name=None, website=None, days=3, title_filter=None):
            asked.update(sym=sym, query=query, website=website, days=days)
            return [
                SourceItem(
                    tier=SourceTier.PRIMARY,
                    provider="t1energy.com",
                    url="https://news.google.com/rss/articles/abc",
                    title="T1 Energy files voluntary Chapter 11 petitions",
                    published_at=NOW - timedelta(hours=3),
                    entity=EntityMatch(symbol="TE", method=MatchMethod.COMPANY_NAME),
                    doc_type="COMPANY_STATEMENT",
                )
            ]

        page = '<script>{"datePublished": "%s"}</script>' % (NOW - timedelta(hours=3)).isoformat()
        n = distress.sweep(
            store,
            "TE",
            company="T1 Energy Inc.",
            website="https://www.t1energy.com",
            google=google,
            verify_kwargs={
                "fetch": lambda url: page,
                "resolve": lambda link: "https://t1energy.com/pr",
            },
        )
        assert n == 1 and asked["website"] == "https://www.t1energy.com" and asked["days"] == 3
        assert asked["query"].startswith('"T1 Energy" (bankruptcy OR')
        (item,) = news_items(store, "TE", NOW)
        assert item.issuer and item.provider == "t1energy.com"

    def test_a_failed_website_lookup_is_reported_and_the_search_still_runs(self, store):
        searched = []

        def boom(sym):
            raise TimeoutError("yfinance")

        _, errors = distress_all(
            store,
            ["TE"],
            NOW,
            searcher=lambda st, sym, company=None, website=None: searched.append(website),
            reader=lambda st, sym, now, company=None: DistressReading(
                symbol=sym, as_of=now, status="NO_ITEMS"
            ),
            names=lambda s: "T1 Energy Inc.",
            websites=boom,
        )
        assert searched == [None] and "website lookup failed: yfinance" in errors[0]


class TestCheckedNewsOnly:
    def test_a_syndicated_story_is_one_outlet(self, store):
        """In the store 2026-09-27: one Zacks piece also ran on TradingView."""
        news(store, "T1 Energy misses interest payment, enters grace period", "zacks.com")
        news(store, "T1 Energy misses interest payment, enters grace period", "tradingview.com")
        r = read_distress(
            store, "TE", NOW, complete=says(Verdict.EXIT_GRADE, Situation.DEFAULT, ["N1", "N2"])
        )
        assert r.outlets == ["zacks"] or r.outlets == ["tradingview"]
        (c,) = exit_calls(held_sheet(r), net_liq=7_957.51)[0]
        assert c.action == "REVIEW"

    def test_two_independent_stories_are_two_outlets(self, store):
        news(store, "T1 Energy misses interest payment, enters grace period", "reuters.com")
        news(store, "T1 Energy hires restructuring advisers after skipped coupon", "bloomberg.com")
        r = read_distress(
            store, "TE", NOW, complete=says(Verdict.EXIT_GRADE, Situation.DEFAULT, ["N1", "N2"])
        )
        assert r.outlets == ["bloomberg", "reuters"]

    @pytest.mark.parametrize("verified", ["UNVERIFIED", None])
    def test_unchecked_news_is_not_read(self, store, verified):
        news(store, "T1 Energy files for Chapter 11", "reuters.com", verified=verified)
        assert news_items(store, "TE", NOW) == []

    def test_a_story_whose_checked_date_is_old_is_not_read(self, store):
        """The TradingKey case: claimed yesterday, the page says three months ago."""
        store.emit(
            Event(
                ts=NOW - timedelta(days=1),  # a legacy row still dated at ingestion
                source=EventSource.CALENDAR,
                kind="NEWS_CONTEXT",
                tier=EventTier.C,
                symbol="TE",
                dedup_key="stale",
                payload={
                    "title": "T1 Energy faces default",
                    "provider": "tradingkey.com",
                    "published_at": (NOW - timedelta(days=90)).isoformat(),
                    "verified": "CORRECTED",
                },
            )
        )
        assert news_items(store, "TE", NOW) == []
