"""The scorecard: numbers first, each from a stored source, none from a model.

Figures are SPCX's, live on 2026-09-24: $31.3bn run-rate, +91.9% YoY,
consensus FY2026 $44.8bn and FY2027 $108.3bn, 25.4% required at $147.60.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import pytest
from advisor.daemon.book import EQUITY, BookSnapshot, Position
from advisor.daemon.market_calendar import now_et
from advisor.daemon.models import Event, EventSource, EventTier
from advisor.daemon.store import DaemonStore
from advisor.story.reading import Fact, check, facts_hash
from advisor.story.scorecard import build_scorecard, scorecard_facts
from advisor.thesis.models import Claim, ClaimKind, Comparator, Trigger
from advisor.valuation.consensus import (
    Consensus,
    RevenueEstimate,
    load_consensus,
    remaining_cagr,
)
from advisor.valuation.implied import implied_expectations
from advisor.valuation.models import Fundamentals, ValuationSnapshot


@pytest.fixture
def store(tmp_path: Path):
    s = DaemonStore(tmp_path / "research.db")
    yield s
    s.close()


def valuation(*, yoy: float | None = 0.9194, price: float = 147.60) -> ValuationSnapshot:
    # EV = required revenue × margin × multiple: $301.6bn × 25% × 25x.
    base = implied_expectations(1.885e12, 31.256e9, terminal_multiple=25, fcf_margin=0.25)
    return ValuationSnapshot(
        symbol="SPCX", asof=date(2026, 9, 24), price=price, shares_outstanding=1.3e10,
        market_cap=1.95e12, net_cash=None, enterprise_value=1.885e12,
        revenue_runrate=31.256e9, ev_to_revenue=60.3, source_accession="0001628280-26-052535",
        period_end=date(2026, 6, 30), revenue_yoy=yoy, scenarios=[base],
    )  # fmt: skip


CONSENSUS = Consensus(
    symbol="SPCX",
    asof=now_et(),
    years=[
        RevenueEstimate(label="FY2026", fiscal_year_end=date(2026, 12, 31), avg=44.8e9,
                        low=39.1e9, high=51.3e9, analysts=19),
        RevenueEstimate(label="FY2027", fiscal_year_end=date(2027, 12, 31), avg=108.3e9,
                        low=59.4e9, high=152.0e9, analysts=19, growth=1.417),
    ],
)  # fmt: skip


def spcx_book(weight: float = 0.206) -> BookSnapshot:
    held = Position(
        account="A", symbol="SPCX", underlying="SPCX", instrument=EQUITY,
        quantity=11, avg_open_price=128.16, close_price=147.74, mark_price=147.74,
    )  # fmt: skip
    return BookSnapshot(as_of=now_et(), net_liq=11 * 147.74 / weight, positions=[held])


def labels(rows):
    return {r.label: r for r in rows}


class TestRemainingGrowth:
    def test_the_spcx_arithmetic(self):
        cagr, years = remaining_cagr(301.6e9, date(2036, 6, 30), CONSENSUS.years[-1])
        assert years == pytest.approx(8.5, abs=0.01)
        assert cagr == pytest.approx(0.128, abs=0.001)

    def test_an_estimate_past_the_horizon_cannot_be_used(self):
        late = RevenueEstimate(label="FY2040", fiscal_year_end=date(2040, 1, 1), avg=1e9)
        assert remaining_cagr(301.6e9, date(2036, 6, 30), late) is None

    def test_zero_or_negative_estimates_cannot_be_used(self):
        zero = CONSENSUS.years[-1].model_copy(update={"avg": 0.0})
        assert remaining_cagr(301.6e9, date(2036, 6, 30), zero) is None


class TestExpectations:
    def test_every_row_says_where_it_came_from(self, store):
        store.save_valuation(valuation())
        card = build_scorecard(store, "SPCX", consensus_loader=lambda *_: CONSENSUS)
        rows = labels(card.expectations)
        assert rows["Price requires"].value == "25.4%/yr × 10y"
        assert rows["Growing"].value == "+91.9% YoY"
        assert rows["Consensus FY2027"].value == "$108.3bn"
        assert "19 analysts" in rows["Consensus FY2027"].detail
        assert rows["If FY2027 holds"].value == "12.8%/yr"
        assert all(r.source for r in card.expectations)

    def test_no_comparative_period_means_no_growth_row(self, store):
        """A first report after an IPO has no prior year; never estimated."""
        store.save_valuation(valuation(yoy=None))
        card = build_scorecard(store, "SPCX", consensus_loader=lambda *_: CONSENSUS)
        assert "Growing" not in labels(card.expectations)

    def test_no_consensus_says_so_and_skips_the_arithmetic(self, store):
        store.save_valuation(valuation())
        card = build_scorecard(store, "SPCX", consensus_loader=lambda *_: None)
        rows = labels(card.expectations)
        assert rows["Consensus"].value == "unavailable"
        assert not any(label.startswith("If ") for label in rows)

    def test_no_valuation_is_said_not_guessed(self, store):
        card = build_scorecard(store, "SPCX", consensus_loader=lambda *_: CONSENSUS)
        assert card.expectations[0].value == "no valuation"

    def test_no_row_publishes_a_value_for_the_business(self, store):
        store.save_valuation(valuation())
        card = build_scorecard(store, "SPCX", consensus_loader=lambda *_: CONSENSUS)
        text = " ".join(f"{r.label} {r.value} {r.detail}" for r in card.expectations).lower()
        assert "fair value" not in text and "target" not in text and "worth" not in text


def claim(field, threshold, kinds, comparator=Comparator.ABOVE, text="the holder's words"):
    return Claim(
        kind=ClaimKind.INVALIDATION,
        text=text,
        trigger=Trigger(event_kinds=kinds, field=field, comparator=comparator, threshold=threshold),
    )


class TestThresholds:
    def test_a_standing_breach(self, store):
        store.save_valuation(valuation())
        store.save_book(spcx_book())
        store.save_claim("SPCX", claim("implied_cagr", 0.25, ["IMPLIED_EXPECTATIONS_SHIFT"]))
        rows = labels(build_scorecard(store, "SPCX", consensus_loader=lambda *_: None).thresholds)
        row = rows["Required growth ≤ 25.0%"]
        assert (row.value, row.status) == ("25.4%", "TRIPPED")
        assert row.claim == "the holder's words"

    def test_an_event_claim_with_no_event_is_within_and_says_since_when(self, store):
        store.save_book(spcx_book())
        old = Event(
            source=EventSource.COMPUTED, kind="POSITION_OPENED", tier=EventTier.B,
            symbol="SPCX", dedup_key="o", payload={},
        )  # fmt: skip
        old.ts = datetime(2026, 9, 13, 10, tzinfo=now_et().tzinfo)
        store.emit(old)
        store.save_claim("SPCX", claim("dilution_pct", 0.03, ["FILING_DILUTION"]))
        row = labels(build_scorecard(store, "SPCX", consensus_loader=lambda *_: None).thresholds)[
            "Dilution ≤ 3.0%"
        ]
        assert (row.value, row.status) == ("none", "OK")
        assert "since 2026-09-13" in row.detail

    def test_the_latest_qualifying_event_decides(self, store):
        store.save_book(spcx_book())
        store.save_claim("SPCX", claim("dilution_pct", 0.05, ["FILING_DILUTION"]))
        for days, pct in ((10, 0.02), (1, 0.067)):
            event = Event(
                source=EventSource.EDGAR, kind="FILING_DILUTION", tier=EventTier.A,
                symbol="SPCX", dedup_key=f"d{days}", payload={"dilution_pct": pct},
            )  # fmt: skip
            event.ts = now_et() - timedelta(days=days)
            store.emit(event)
        row = build_scorecard(store, "SPCX", consensus_loader=lambda *_: None).thresholds[0]
        assert (row.value, row.status) == ("6.7%", "TRIPPED")

    def test_an_unreachable_claim_is_never_a_pass(self, store):
        """SPCX's residual claim can never fire: 59 sessions against a 120 floor."""
        store.save_book(spcx_book())
        store.save_claim("SPCX", claim("residual_z", 2.0, ["RESIDUAL_DIVERGENCE"]))
        row = build_scorecard(store, "SPCX", consensus_loader=lambda *_: None).thresholds[0]
        assert (row.value, row.status) == ("can't be checked", "UNCHECKABLE")
        assert "factor estimate" in row.detail

    def test_claims_without_a_number_are_not_rows(self, store):
        store.save_claim("SPCX", claim(None, None, ["FILING_RESULTS"]))
        assert build_scorecard(store, "SPCX", consensus_loader=lambda *_: None).thresholds == []

    @pytest.mark.parametrize("weight, status", [(0.206, "TRIPPED"), (0.20, "OK"), (0.12, "OK")])
    def test_the_book_limit_at_and_around_the_line(self, store, weight, status):
        store.save_book(spcx_book(weight))
        row = labels(build_scorecard(store, "SPCX", consensus_loader=lambda *_: None).thresholds)[
            "Position weight ≤ 20.0%"
        ]
        assert row.status == status

    def test_not_held_has_no_weight_row(self, store):
        assert build_scorecard(store, "SPCX", consensus_loader=lambda *_: None).thresholds == []


class TestConsensusCache:
    def test_a_fresh_cache_is_not_refetched(self, store):
        calls = []
        load_consensus(store, "SPCX", fetch=lambda s: calls.append(s) or CONSENSUS)
        load_consensus(store, "SPCX", fetch=lambda s: calls.append(s) or CONSENSUS)
        assert calls == ["SPCX"]

    def test_a_stale_cache_is_refetched(self, store):
        store.save_consensus(
            "SPCX",
            CONSENSUS.model_copy(update={"asof": now_et() - timedelta(days=2)}).model_dump_json(),
        )
        calls = []
        load_consensus(store, "SPCX", fetch=lambda s: calls.append(s) or CONSENSUS)
        assert calls == ["SPCX"]

    def test_a_failed_fetch_falls_back_to_the_dated_cache(self, store):
        stale = CONSENSUS.model_copy(update={"asof": now_et() - timedelta(days=5)})
        store.save_consensus("SPCX", stale.model_dump_json())
        got = load_consensus(store, "SPCX", fetch=lambda s: None)
        assert got is not None and got.asof == stale.asof

    def test_nothing_anywhere(self, store):
        assert load_consensus(store, "SPCX", fetch=lambda s: None) is None


class TestPriorYear:
    """The comparative column, from the same 10-Q."""

    class Xbrl:
        def __init__(self, rows):
            import pandas as pd

            self.df = pd.DataFrame(rows)

        def query(self):
            outer = self

            class Q:
                def by_concept(self, concept):
                    class R:
                        def to_dataframe(self):
                            return outer.df

                    return R()

            return Q()

    def rows(self, prior_end="2025-06-30", prior_start="2025-04-01", dimensioned=False):
        return [
            dict(period_start="2026-04-01", period_end="2026-06-30", numeric_value=7.814e9,
                 is_dimensioned=False),
            dict(period_start=prior_start, period_end=prior_end, numeric_value=4.071e9,
                 is_dimensioned=dimensioned),
            dict(period_start="2025-01-01", period_end="2025-06-30", numeric_value=7.9e9,
                 is_dimensioned=False),  # six months: same end, wrong length
        ]  # fmt: skip

    def test_spcx_q2(self):
        from advisor.valuation.fundamentals import REVENUE_CONCEPTS, _prior_year_value

        got = _prior_year_value(
            self.Xbrl(self.rows()), REVENUE_CONCEPTS[:1], date(2026, 4, 1), date(2026, 6, 30)
        )
        assert got == 4.071e9

    def test_a_52_53_week_calendar_is_a_day_off(self):
        from advisor.valuation.fundamentals import REVENUE_CONCEPTS, _prior_year_value

        xbrl = self.Xbrl(self.rows(prior_end="2025-06-28", prior_start="2025-03-30"))
        got = _prior_year_value(xbrl, REVENUE_CONCEPTS[:1], date(2026, 3, 29), date(2026, 6, 27))
        assert got == 4.071e9

    def test_a_segment_fact_is_not_the_total(self):
        from advisor.valuation.fundamentals import REVENUE_CONCEPTS, _prior_year_value

        xbrl = self.Xbrl(self.rows(dimensioned=True))
        assert (
            _prior_year_value(xbrl, REVENUE_CONCEPTS[:1], date(2026, 4, 1), date(2026, 6, 30))
            is None
        )

    def test_growth_is_none_without_a_comparative(self):
        f = Fundamentals(
            symbol="SPCX", source_accession="a", period_end=date(2026, 6, 30), revenue=7.8e9
        )
        assert f.revenue_yoy is None
        assert f.model_copy(update={"prior_revenue": 0.0}).revenue_yoy is None


class TestReadingOverTheScorecard:
    BREACHED = Fact(
        id="F9", kind="SCORECARD",
        text="Threshold Position weight ≤ 20.0%: 20.6%, BREACHED (book limit)",
    )  # fmt: skip

    @pytest.mark.parametrize(
        "text",
        [
            "La posición roza el límite del 20.0% con 20.6%.",
            "El peso de 20.6% está cerca del límite del 20.0%.",
            "At 20.6% the position is approaching its 20.0% limit.",
        ],
    )
    def test_a_breach_cannot_be_softened(self, text):
        from advisor.story.reading import Draft, Sentence, Stance

        d = Draft(stance=Stance.CAUTIOUS, sentences=[Sentence(text=text, facts=["F9"])])
        assert any("past the line" in p for p in check(d, [self.BREACHED]))

    def test_saying_it_plainly_passes(self):
        from advisor.story.reading import Draft, Sentence, Stance

        d = Draft(
            stance=Stance.CAUTIOUS,
            sentences=[
                Sentence(text="La posición supera el límite: 20.6% vs 20.0%.", facts=["F9"])
            ],
        )
        assert check(d, [self.BREACHED]) == []

    def test_three_sentences_are_too_many(self):
        from advisor.story.reading import Draft, Sentence, Stance

        d = Draft(stance=Stance.NEUTRAL, sentences=[Sentence(text="x.", facts=["F9"])] * 3)
        assert "not 3" in check(d, [self.BREACHED])[0]

    def test_the_weight_row_ticking_does_not_rewrite_the_reading(self):
        a = [self.BREACHED]
        b = [self.BREACHED.model_copy(update={"text": self.BREACHED.text.replace("20.6", "20.9")})]
        assert facts_hash(a) == facts_hash(b)

    def test_crossing_back_under_the_line_does(self):
        within = self.BREACHED.model_copy(
            update={"text": "Threshold Position weight ≤ 20.0%: 19.8%, within (book limit)"}
        )
        assert facts_hash([self.BREACHED]) != facts_hash([within])

    def test_facts_carry_every_row(self, store):
        store.save_valuation(valuation())
        card = build_scorecard(store, "SPCX", consensus_loader=lambda *_: CONSENSUS)
        lines = scorecard_facts(card)
        assert any(line.startswith("If FY2027 holds: 12.8%/yr") for line in lines)
