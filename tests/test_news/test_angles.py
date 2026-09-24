"""Angles: products and segments a holding is a bet on.

Grok 4.7 shipped on 2026-09-21 and the advisor never saw it. Tavily, asked
directly, returned six articles on it; the entity matcher dropped all six,
because "SpaceXAI Releases Grok 4.7" names neither Space Exploration
Technologies Corp nor SPCX. These tests pin the fix and its limits.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest
from advisor.daemon.models import EventTier
from advisor.daemon.store import DaemonStore
from advisor.news.angles import (
    AngleStatus,
    angle_event,
    refresh_suggestions,
    scan_angles,
    suggest_angles,
)
from advisor.news.entities import mentions
from advisor.news.models import EntityMatch, MatchMethod, SourceItem, SourceTier

SPCX_NAME = "SPACE EXPLORATION TECHNOLOGIES CORP"

# The holder's real SPCX claims, verbatim.
CLAIMS = [
    ("c1", "El precio implica que X+xAI valen ~$1.5 bn a 144x ingresos: el segmento IA debe "
           "crecer >100% interanual para sostenerlo"),
    ("c2", "Starlink es el motor de caja (55% de ingresos, $3.2 bn EBITDA aj.): si su "
           "crecimiento cae bajo 40% se acaba el subsidio a IA"),
    ("c3", "Cualquier ampliación de capital sobre 3% de la capitalización tras haber levantado "
           "$86 bn en la OPV rompe la tesis"),
    ("c4", "Litigios sobre generación de imágenes de Grok, revelados en el 10-Q"),
    ("c5", "Un movimiento que macro no explica avisa de algo que no sé"),
]  # fmt: skip


class TestSuggestions:
    def test_the_real_claims_suggest_the_products(self):
        terms = [t for t, _ in suggest_angles(CLAIMS, symbol="SPCX", company_name=SPCX_NAME)]
        assert {"xAI", "Starlink", "Grok"} <= set(terms)

    def test_finance_vocabulary_and_openers_are_not_suggested(self):
        terms = {t for t, _ in suggest_angles(CLAIMS, symbol="SPCX", company_name=SPCX_NAME)}
        assert not terms & {"EBITDA", "OPV", "IA", "10-Q", "El", "Cualquier", "Un", "X"}

    def test_each_suggestion_names_the_claim_it_came_from(self):
        found = dict(suggest_angles(CLAIMS, symbol="SPCX", company_name=SPCX_NAME))
        assert found["Grok"] == "c4" and found["Starlink"] == "c2"

    def test_the_company_and_ticker_are_not_angles(self):
        claims = [(None, "Space Exploration and SPCX need Starship to fly")]
        terms = {t for t, _ in suggest_angles(claims, symbol="SPCX", company_name=SPCX_NAME)}
        assert terms == {"Starship"}

    def test_no_claims_no_suggestions(self):
        assert suggest_angles([], symbol="SPCX") == []


class TestMentions:
    def test_a_capitalised_angle_does_not_match_the_verb(self):
        assert mentions("SpaceXAI releases Grok 4.7", "Grok")
        assert not mentions("hard to grok the numbers", "Grok")

    def test_word_boundaries(self):
        assert not mentions("Starlinked", "Starlink")
        assert mentions("xAI's new model", "xAI")

    def test_a_lowercase_angle_matches_any_case(self):
        assert mentions("STARSHIP flight 12", "starship")

    def test_empty_term(self):
        assert not mentions("anything", "  ")


@pytest.fixture
def store(tmp_path: Path):
    s = DaemonStore(tmp_path / "research.db")
    yield s
    s.close()


class TestDecisions:
    def test_a_suggestion_never_overrides_a_decision(self, store):
        store.set_angle_status("SPCX", "Grok", AngleStatus.REJECTED.value)
        assert not store.add_angle("SPCX", "grok", AngleStatus.SUGGESTED.value, source="c4")
        assert store.list_angles("SPCX")[0]["status"] == "REJECTED"

    def test_refreshing_suggestions_is_idempotent(self, store):
        from advisor.thesis.models import Claim, ClaimKind

        store.save_claim("SPCX", Claim(kind=ClaimKind.RISK, text=CLAIMS[3][1]))
        # "Grok", and "Litigios" — a sentence opener the holder will reject.
        assert refresh_suggestions(store, "SPCX", company_name=SPCX_NAME) == 2
        assert refresh_suggestions(store, "SPCX", company_name=SPCX_NAME) == 0

    def test_a_manual_angle_is_recorded_as_manual(self, store):
        store.set_angle_status("SPCX", "Starship", AngleStatus.CONFIRMED.value)
        (angle,) = store.list_angles("SPCX")
        assert (angle["term"], angle["status"], angle["source"]) == (
            "Starship",
            "CONFIRMED",
            "manual",
        )

    def test_confirmed_are_listed_first(self, store):
        store.add_angle("SPCX", "Litigios", "SUGGESTED", source="c4")
        store.set_angle_status("SPCX", "Grok", "CONFIRMED")
        assert [a["term"] for a in store.list_angles("SPCX")] == ["Grok", "Litigios"]


def article(title="SpaceXAI Releases Grok 4.7 for Coding", url="https://unite.ai/grok-47"):
    return SourceItem(
        tier=SourceTier.AGGREGATOR,
        provider="www.unite.ai",
        url=url,
        title=title,
        published_at=datetime(2026, 9, 21, 14, tzinfo=timezone.utc),
        entity=EntityMatch(symbol="SPCX", method=MatchMethod.ALIAS),
        doc_type="NEWS",
        summary="SpaceXAI released Grok 4.7, a larger base model at the same $2 price.",
    )


class FakeSearch:
    def __init__(self, results=None, fail=()):
        self.calls: list[tuple[str, str]] = []
        self.results = results or {}
        self.fail = set(fail)

    def __call__(self, symbol, query, **kw):
        self.calls.append((symbol, query))
        assert kw["aliases"] == [query]
        if query in self.fail:
            raise TimeoutError("tavily timed out")
        return self.results.get(query, [])


class TestScan:
    def test_only_confirmed_angles_are_searched(self, store):
        store.add_angle("SPCX", "Litigios", "SUGGESTED", source="c4")
        store.set_angle_status("SPCX", "Grok", "CONFIRMED")
        store.set_angle_status("SPCX", "Starlink", "REJECTED")
        search = FakeSearch()
        scan_angles(store, ["SPCX"], search=search)
        assert search.calls == [("SPCX", "Grok")]

    def test_an_article_becomes_one_tier_c_event(self, store):
        store.set_angle_status("SPCX", "Grok", "CONFIRMED")
        result = scan_angles(store, ["SPCX"], search=FakeSearch({"Grok": [article()]}))
        (event,) = result.events
        assert event.kind == "NEWS_ANGLE" and event.tier is EventTier.C
        assert event.payload["angle"] == "Grok"
        assert event.payload["lead"].startswith("SpaceXAI released Grok 4.7")

    def test_a_second_day_does_not_duplicate(self, store):
        store.set_angle_status("SPCX", "Grok", "CONFIRMED")
        search = FakeSearch({"Grok": [article()]})
        scan_angles(store, ["SPCX"], search=search)
        again = scan_angles(store, ["SPCX"], search=search)
        assert (again.items_stored, again.events) == (0, [])

    def test_one_article_found_by_two_angles_is_one_event(self, store):
        store.set_angle_status("SPCX", "Grok", "CONFIRMED")
        store.set_angle_status("SPCX", "xAI", "CONFIRMED")
        same = article()
        result = scan_angles(store, ["SPCX"], search=FakeSearch({"Grok": [same], "xAI": [same]}))
        assert len(result.events) == 1

    def test_the_budget_starves_the_last_symbols_not_the_first(self, store):
        for sym in ("SPCX", "AAOI", "TE"):
            store.set_angle_status(sym, f"{sym}-product", "CONFIRMED")
        search = FakeSearch()
        result = scan_angles(store, ["SPCX", "AAOI", "TE"], search=search, budget=2)
        assert [s for s, _ in search.calls] == ["SPCX", "AAOI"]
        assert result.skipped_for_budget == ["TE:TE-product"]

    def test_zero_budget_searches_nothing(self, store):
        store.set_angle_status("SPCX", "Grok", "CONFIRMED")
        search = FakeSearch()
        assert scan_angles(store, ["SPCX"], search=search, budget=0).queries == 0
        assert search.calls == []

    def test_one_failing_angle_does_not_stop_the_rest(self, store):
        store.set_angle_status("SPCX", "Grok", "CONFIRMED")
        store.set_angle_status("SPCX", "Starlink", "CONFIRMED")
        result = scan_angles(store, ["SPCX"], search=FakeSearch(fail={"Grok"}))
        assert result.queries == 2
        assert "tavily timed out" in result.errors[0]

    def test_no_angles_no_queries(self, store):
        assert scan_angles(store, ["SPCX"], search=FakeSearch()).queries == 0

    def test_an_empty_book(self, store):
        assert scan_angles(store, [], search=FakeSearch()).queries == 0


class TestTavilyAliasFallback:
    """The six live Grok results were all dropped; with the angle, they are kept."""

    @pytest.fixture
    def tavily(self, monkeypatch):
        import advisor.news.tavily as mod

        class Config:
            tavily_api_key = "test"
            search_endpoint = "https://example.invalid"
            http_timeout_seconds = 1

        rows = [
            {
                "title": "SpaceXAI Releases Grok 4.7 for Coding and Knowledge",
                "url": "https://www.unite.ai/spacexai-releases-grok-4-7",
                "published_date": "Mon, 21 Sep 2026 14:00:00 GMT",
                "score": 0.9,
                "content": "Grok 4.7 is a larger base model at the same $2 price.",
            },
            {
                "title": "Why nobody can grok this market",
                "url": "https://example.com/verb",
                "published_date": "Mon, 21 Sep 2026 14:00:00 GMT",
                "score": 0.9,
                "content": "Hard to grok.",
            },
        ]

        class Response:
            def raise_for_status(self):
                pass

            def json(self):
                return {"results": rows}

        monkeypatch.setattr("research_agent.config.ResearchConfig", lambda: Config())
        monkeypatch.setattr(mod.httpx, "post", lambda *a, **k: Response())
        return mod

    def test_without_the_angle_the_article_is_dropped(self, tavily):
        assert tavily.search_news("SPCX", "Grok", company_name=SPCX_NAME) == []

    def test_with_the_angle_it_is_kept_as_an_alias_match(self, tavily):
        (item,) = tavily.search_news("SPCX", "Grok", company_name=SPCX_NAME, aliases=["Grok"])
        assert item.entity.method is MatchMethod.ALIAS
        assert "Grok 4.7" in item.title

    def test_the_verb_is_not_the_product(self, tavily):
        items = tavily.search_news("SPCX", "Grok", company_name=SPCX_NAME, aliases=["Grok"])
        assert all("grok this market" not in i.title for i in items)


def test_an_alias_is_the_weakest_match():
    from advisor.news.models import MATCH_CONFIDENCE

    ranked = sorted((m for m in MatchMethod if m is not MatchMethod.NONE), key=MATCH_CONFIDENCE.get)
    assert ranked[0] is MatchMethod.ALIAS


def test_an_angle_event_never_interrupts():
    event = angle_event(article(), "Grok")
    assert event.tier is EventTier.C
