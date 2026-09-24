"""The ticker reading: model prose, admitted only through an arithmetic gate.

Every rejection case here is one the live model actually produced or one the
gate got wrong against it: "38 311,8" rejected as 311.8, a 5.8% requirement
computed at $104.90 paired with a later price, "dilution already executed"
for an at-the-market facility that is only a ceiling.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pytest
from advisor.daemon.market_calendar import now_et
from advisor.daemon.models import Event, EventSource, EventTier
from advisor.daemon.store import DaemonStore
from advisor.story.reading import (
    Draft,
    Fact,
    ReadingStatus,
    Sentence,
    Stance,
    _readings,
    check,
    facts_hash,
    gather_facts,
    numbers_in,
    position_moved,
    read_symbol,
    tidy,
)

FACTS = [
    Fact(id="F1", kind="POSITION", text="Held: 12 shares, 15.4% of net liq, unrealized -22.6%"),
    Fact(id="F2", kind="VALUATION", text="At $104.90 the price requires 5.8% revenue CAGR"),
    Fact(id="F3", kind="EVENT", text="dilution (tier A, 424B5) $600M  = 6.7% dilution"),
    Fact(id="F4", kind="EVENT", text="lease of approximately 38,311.8 square meters in Ningbo"),
    Fact(id="F5", kind="EVENT", text="insider selling cluster 3 insiders selling  $6M"),
    Fact(id="F6", kind="CLAIM", text="The holder's claim: any raise above 5% of market cap"),
    Fact(
        id="F7",
        kind="NEWS",
        text="news context (third-party commentary from www.benzinga.com) — shares fell on an "
        "imminent post-IPO share unlock",
        source="www.benzinga.com",
    ),
]


def draft(*sentences: tuple[str, list[str]], stance=Stance.CAUTIOUS) -> Draft:
    return Draft(stance=stance, sentences=[Sentence(text=t, facts=f) for t, f in sentences])


class TestNumbers:
    @pytest.mark.parametrize(
        "raw, value",
        [("38 311,8", 38311.8), ("38.311,8", 38311.8), ("38,311.8", 38311.8), ("21,5", 21.5)],
    )
    def test_spanish_and_english_forms_read_the_same(self, raw, value):
        assert value in [v for v, _ in _readings(raw)]

    def test_an_ambiguous_grouping_keeps_both_readings(self):
        assert {v for v, _ in _readings("1.234")} == {1234.0, 1.234}

    def test_dates_years_and_form_names_are_not_quantities(self):
        text = "El 10 de septiembre de 2026 un 8-K (Item 1.01) y el 424B5 del Q2, on Sep 4, 2026"
        assert numbers_in(text) == []


class TestGate:
    def test_a_grounded_reading_passes(self):
        ok = draft(
            ("El ATM de hasta $600 millones equivale a 6.7% de dilución.", ["F3"]),
            ("Eso supera el 5% que fijó el titular.", ["F3", "F6"]),
        )
        assert check(ok, FACTS) == []

    def test_a_number_in_no_fact_is_rejected(self):
        problems = check(draft(("La dilución es de 9.1%.", ["F3"])), FACTS)
        assert "9.1%" in problems[0]

    def test_a_number_from_an_uncited_fact_is_rejected(self):
        """Grounded somewhere is not enough; it must be in what the sentence cites."""
        problems = check(draft(("El precio requiere 5.8% de CAGR.", ["F3"])), FACTS)
        assert problems and "5.8%" in problems[0]

    def test_rounding_is_allowed_recomputing_is_not(self):
        assert check(draft(("Pérdida de 23%.", ["F1"])), FACTS) == []
        assert check(draft(("Pérdida de 24%.", ["F1"])), FACTS) != []

    def test_units_convert(self):
        assert check(draft(("Hasta $0.6bn.", ["F3"])), FACTS) == []
        assert check(draft(("Hasta 600 millones.", ["F3"])), FACTS) == []

    @pytest.mark.parametrize(
        "written", ["$2 billones", "$2 billón", "$2 trillion", "$2T", "$2,000 bn"]
    )
    def test_the_long_scale(self, written):
        """Live SPCX: '$2 billones' for Fool's '$2 trillion' was rejected as 10^9."""
        facts = [Fact(id="F1", kind="EVENT", text="market value near $2 trillion today")]
        assert check(draft((f"Vale cerca de {written}.", ["F1"])), facts) == []

    def test_a_spanish_billion_is_not_an_english_billion(self):
        facts = [Fact(id="F1", kind="EVENT", text="proceeds of $2 billion")]
        assert check(draft(("Recaudó $2 billones.", ["F1"])), facts) != []
        assert check(draft(("Recaudó $2 mil millones.", ["F1"])), facts) == []

    def test_the_space_grouped_area_is_grounded(self):
        """The live false positive: '38 311,8' was read as 311.8 and rejected."""
        assert check(draft(("Una nave de 38 311,8 m² en Ningbo.", ["F4"])), FACTS) == []

    def test_a_sign_is_carried_in_words(self):
        assert check(draft(("Cayó 22.6% desde la entrada.", ["F1"])), FACTS) == []

    def test_small_counts_are_allowed(self):
        assert check(draft(("Tres insiders vendieron; 3 en total.", ["F5"])), FACTS) == []

    @pytest.mark.parametrize(
        "text",
        [
            "El valor justo es mayor al precio.",
            "The stock looks undervalued.",
            "Está sobrevalorada frente a pares.",
            "Our price target stands.",
        ],
    )
    def test_valuation_opinions_are_rejected(self, text):
        assert any("valuation opinion" in p for p in check(draft((text, ["F2"])), FACTS))

    def test_commentary_stated_as_fact_is_rejected(self):
        """Live SPCX draft: a Benzinga unlock story written as established fact."""
        problems = check(draft(("Un desbloqueo post-OPV crea presión de oferta.", ["F7"])), FACTS)
        assert any("without saying who said it" in p for p in problems)

    @pytest.mark.parametrize(
        "text",
        [
            "Según Benzinga, un desbloqueo post-OPV crea presión de oferta.",
            "Benzinga atribuye la caída a un desbloqueo post-OPV.",
            "Press reports tie the drop to a post-IPO unlock.",
        ],
    )
    def test_attributed_commentary_passes(self, text):
        assert check(draft((text, ["F7"])), FACTS) == []

    def test_a_filing_needs_no_attribution(self):
        assert check(draft(("El ATM es de hasta $600M.", ["F3"])), FACTS) == []

    def test_a_sentence_must_cite(self):
        assert check(draft(("Algo pasa.", [])), FACTS) == ["sentence 1 cites no fact"]

    def test_a_citation_must_exist(self):
        problems = check(draft(("Algo pasa.", ["F1", "F99"])), FACTS)
        assert "F99" in problems[0]

    def test_sentence_count_is_bounded(self):
        many = draft(*[("Hecho.", ["F1"])] * 5)
        assert "not 5" in check(many, FACTS)[0]
        assert "not 0" in check(draft(), FACTS)[0]


def position(weight: float, ret: float, price: float = 100.13) -> Fact:
    return Fact(
        id="F1",
        kind="POSITION",
        text=f"Held: 12 shares, {weight}% of net liq, unrealized {ret:+}%, price ${price}",
    )


class TestTidy:
    def test_inline_citations_are_removed(self):
        out = tidy(draft(("Supera el umbral del 5% [F1, F3, F7]. Y más (F4; F6).", ["F1"])))
        assert out.sentences[0].text == "Supera el umbral del 5%. Y más."
        assert out.sentences[0].facts == ["F1"]

    def test_prose_in_brackets_survives(self):
        text = "El ATM (hasta $600M) y el [anexo 10.1] quedan."
        assert tidy(draft((text, ["F3"]))).sentences[0].text == text


class TestCacheKey:
    def test_a_price_tick_keeps_the_key(self):
        assert facts_hash([position(15.4, -22.6)]) == facts_hash([position(15.3, -22.1, 100.8)])

    def test_a_new_event_changes_it(self):
        assert facts_hash(FACTS[:3]) != facts_hash(FACTS[:4])

    def test_selling_out_changes_it(self):
        gone = Fact(id="F1", kind="POSITION", text="Not held in the book as of 2026-09-25.")
        assert facts_hash([position(15.4, -22.6)]) != facts_hash([gone])


class TestPositionMoved:
    def test_hovering_at_a_boundary_is_not_a_move(self):
        """Bucketing thrashed here: -22.6% and -22.1% straddle a 5pp edge."""
        assert not position_moved([position(15.4, -22.6)], [position(15.3, -22.1)])

    def test_five_points_of_return_is(self):
        assert position_moved([position(15.4, -22.6)], [position(15.4, -27.6)])

    def test_one_point_of_weight_is(self):
        assert position_moved([position(15.4, -22.6)], [position(16.4, -22.6)])

    def test_just_under_both_thresholds_is_not(self):
        assert not position_moved([position(15.4, -22.6)], [position(16.3, -18.7)])

    def test_no_position_line_on_either_side(self):
        assert not position_moved(FACTS[1:2], FACTS[1:2])


@pytest.fixture
def store(tmp_path: Path):
    s = DaemonStore(tmp_path / "research.db")
    yield s
    s.close()


def filing(
    store, *, days_ago=3, lead="On August 31, 2026, AAOI entered into two leases in Houston."
):
    event = Event(
        source=EventSource.EDGAR,
        kind="FILING_MATERIAL_AGREEMENT",
        tier=EventTier.B,
        symbol="AAOI",
        dedup_key=f"acc-{days_ago}",
        payload={"form": "8-K", "items": ["1.01"], "lead": lead, "accession": f"acc-{days_ago}"},
    )
    event.ts = now_et() - timedelta(days=days_ago)
    store.emit(event)


class FakeModel:
    def __init__(self, *drafts: Draft):
        self.drafts = list(drafts)
        self.prompts: list[str] = []

    def __call__(self, system: str, user: str) -> Draft:
        self.prompts.append(user)
        return self.drafts.pop(0)


GOOD = draft(("La empresa firmó dos arrendamientos en Houston.", ["F1"]))
BAD = draft(("La empresa firmó 7 arrendamientos por $90M.", ["F1"]))


class TestReadSymbol:
    def test_no_events_means_no_model_call(self, store):
        model = FakeModel()
        reading = read_symbol(store, "AAOI", complete=model)
        assert reading.status is ReadingStatus.NO_FACTS
        assert model.prompts == []

    def test_events_outside_the_window_are_not_facts(self, store):
        filing(store, days_ago=90)
        assert read_symbol(store, "AAOI", complete=FakeModel()).status is ReadingStatus.NO_FACTS

    def test_a_passing_draft_is_shown_and_cached(self, store):
        filing(store)
        model = FakeModel(GOOD)
        first = read_symbol(store, "AAOI", complete=model, model="fake")
        assert first.status is ReadingStatus.OK
        assert first.sentences[0].text.startswith("La empresa firmó")
        again = read_symbol(store, "aaoi", complete=FakeModel())  # would raise if called
        assert again == first

    def test_a_failed_draft_is_retried_with_the_objections(self, store):
        filing(store)
        model = FakeModel(BAD, GOOD)
        reading = read_symbol(store, "AAOI", complete=model)
        assert reading.status is ReadingStatus.OK
        assert "rejected by the checker" in model.prompts[1]
        assert "$90" in model.prompts[1] or "90" in model.prompts[1]

    def test_two_failures_show_nothing_but_keep_the_draft(self, store):
        filing(store)
        reading = read_symbol(store, "AAOI", complete=FakeModel(BAD, BAD))
        assert reading.status is ReadingStatus.REJECTED
        assert reading.sentences == []
        assert reading.rejected_draft and reading.problems
        assert reading.facts  # the facts are still there to read

    def test_the_model_erroring_is_unavailable_and_not_cached(self, store):
        filing(store)

        def boom(_s, _u):
            raise TimeoutError("OpenRouter timed out")

        assert read_symbol(store, "AAOI", complete=boom).status is ReadingStatus.UNAVAILABLE
        assert read_symbol(store, "AAOI", complete=FakeModel(GOOD)).status is ReadingStatus.OK

    def test_refresh_bypasses_the_cache(self, store):
        filing(store)
        read_symbol(store, "AAOI", complete=FakeModel(GOOD))
        model = FakeModel(GOOD)
        read_symbol(store, "AAOI", complete=model, refresh=True)
        assert len(model.prompts) == 1

    def test_a_new_filing_invalidates_the_cache(self, store):
        filing(store)
        read_symbol(store, "AAOI", complete=FakeModel(GOOD))
        filing(store, days_ago=1, lead="On September 10, 2026, AAOI leased a factory in Ningbo.")
        model = FakeModel(GOOD)
        read_symbol(store, "AAOI", complete=model)
        assert len(model.prompts) == 1


class TestGatherFacts:
    def test_a_standing_condition_is_one_fact(self, store):
        for days_ago in (1, 2, 3):
            event = Event(
                source=EventSource.COMPUTED,
                kind="DEEP_DRAWDOWN",
                tier=EventTier.B,
                symbol="AAOI",
                dedup_key=f"dd-{days_ago}",
                payload={"unrealized_pct": -0.21, "threshold": -0.2},
            )
            event.ts = now_et() - timedelta(days=days_ago)
            store.emit(event)
        facts = gather_facts(store, "AAOI")
        assert sum("drawdown" in f.text for f in facts) == 1

    def test_every_state_event_the_mechanics_emit_collapses(self):
        """The set once said CONCENTRATION; the emitter says CONCENTRATION_WARNING."""
        from advisor.daemon.book import EQUITY, BookSnapshot, Position
        from advisor.daemon.mechanics import state_events
        from advisor.story.reading import _STANDING_KINDS

        # 60% of net liq and down 30%: concentrated and in deep drawdown.
        held = Position(
            account="A", symbol="SPCX", underlying="SPCX", instrument=EQUITY,
            quantity=60, avg_open_price=100.0, close_price=70.0, mark_price=70.0,
        )  # fmt: skip
        snapshot = BookSnapshot(as_of=now_et(), net_liq=7_000.0, positions=[held])
        kinds = {e.kind for e in state_events(snapshot)}
        assert kinds, "the fixture should trip at least one standing condition"
        assert kinds <= _STANDING_KINDS

    def test_an_atm_is_marked_as_a_maximum(self, store):
        event = Event(
            source=EventSource.EDGAR,
            kind="FILING_DILUTION",
            tier=EventTier.A,
            symbol="AAOI",
            dedup_key="atm",
            payload={
                "form": "424B5",
                "offering_usd": 600e6,
                "dilution_pct": 0.067,
                "quote": "up to $600,000,000 from time to time through Raymond James",
            },
        )
        store.emit(event)
        (fact,) = [f for f in gather_facts(store, "AAOI") if "dilution" in f.text]
        assert "not an amount already raised" in fact.text

    def test_angle_coverage_cannot_crowd_out_context(self, store):
        """Live SPCX: ten Grok/xAI articles pushed the unlock story out."""

        def news(kind, key, angle=None, hours=0):
            payload = {"title": key, "url": f"https://x/{key}", "provider": "www.benzinga.com"}
            if angle:
                payload["angle"] = angle
            event = Event(
                source=EventSource.CALENDAR, kind=kind, tier=EventTier.C, symbol="SPCX",
                dedup_key=key, payload=payload,
            )  # fmt: skip
            event.ts = now_et() - timedelta(hours=hours)
            store.emit(event)

        for i in range(8):
            news("NEWS_ANGLE", f"grok-{i}", angle="Grok", hours=i)
        for i in range(3):
            news("NEWS_ANGLE", f"xai-{i}", angle="xAI", hours=i)
        news("NEWS_CONTEXT", "unlock", hours=48)
        texts = [f.text for f in gather_facts(store, "SPCX") if f.kind == "NEWS"]
        assert any("unlock" in t for t in texts)
        assert sum("grok-" in t for t in texts) <= 3
        assert sum("xai-" in t for t in texts) <= 3
        assert sum(("grok-" in t) or ("xai-" in t) for t in texts) == 5

    def test_a_form_144_is_a_notice_not_a_sale(self, store):
        event = Event(
            source=EventSource.EDGAR, kind="FILING_INSIDER_TRADE", tier=EventTier.C,
            symbol="SPCX", dedup_key="144", payload={"form": "144", "items": []},
        )  # fmt: skip
        store.emit(event)
        (fact,) = [f for f in gather_facts(store, "SPCX") if f.kind == "EVENT"]
        assert "not a completed sale" in fact.text

    def test_other_symbols_do_not_leak_in(self, store):
        filing(store)
        assert all(f.kind != "EVENT" for f in gather_facts(store, "CRDO"))
