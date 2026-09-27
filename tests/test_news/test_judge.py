"""The news agent: judged, gated, stored once, measured — and never acting."""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest
from advisor.daemon import market_calendar as mc
from advisor.daemon.store import DaemonStore
from advisor.news import judge
from advisor.news.judge import (
    ClaimLink,
    Direction,
    Draft,
    ItemCall,
    Judgment,
    Materiality,
    NewsJudgmentStore,
    check,
    judge_symbol,
    outcomes_for,
    quoted,
    summary,
)
from advisor.news.models import EntityMatch, MatchMethod, SourceItem, SourceTier
from advisor.thesis.models import Claim, ClaimKind

ET = mc.MARKET_TZ
NOW = datetime(2026, 9, 27, 9, 0, tzinfo=ET)


def item(title="Nebius to hike prices next week", summary="GPU prices rise 10% on 1,250 racks",
         url="https://x.com/a", days_ago=1, tier=SourceTier.AGGREGATOR, symbol="NBIS"):  # fmt: skip
    return SourceItem(
        tier=tier,
        provider="www.zacks.com",
        url=url,
        title=title,
        summary=summary,
        published_at=(NOW - timedelta(days=days_ago)).astimezone(timezone.utc),
        entity=EntityMatch(symbol=symbol, method=MatchMethod.COMPANY_NAME),
        doc_type="NEWS",
    )


def call(id="N1", **kw):
    base = dict(
        id=id,
        about_company=True,
        event_type="PRODUCT",
        direction="POSITIVE",
        materiality="MEDIUM",
        novelty="NEW",
        basis="REPORTED",
        quote="GPU prices rise 10%",
        why="Prices rise 10% on 1250 racks.",
    )
    return ItemCall(**{**base, **kw})


class TestGate:
    def test_a_supported_call_passes(self):
        kept, problems = check(call(), item(), {})
        assert kept is not None and problems == []

    def test_a_quote_the_item_lacks_voids_it(self):
        kept, problems = check(call(quote="prices collapse"), item(), {})
        assert kept is None and "quote is not in the item" in problems[0]

    def test_quote_fragments_joined_by_an_ellipsis(self):
        """The model stitched title and summary: each fragment must be there."""
        text = "Nebius to hike prices next week GPU prices rise 10% on 1,250 racks"
        assert quoted("Nebius to hike prices... GPU prices rise", text)
        assert quoted("hike prices […] rise 10%", text)
        assert not quoted("hike prices... prices collapse", text)

    def test_curly_apostrophes_match_straight_ones(self):
        assert quoted("Coleman's Top 2", "Billionaire Chase Coleman’s Top 2 New AI Stock Picks")

    def test_a_trivial_quote_is_not_a_quote(self):
        assert not quoted("...", "anything")
        assert not quoted("a", "a b c")

    def test_a_number_not_in_the_item_voids_it(self):
        kept, problems = check(call(why="Prices rise 25%."), item(), {})
        assert kept is None and "['25']" in problems[0]

    def test_thousands_separators_and_rounding_are_the_same_number(self):
        """$51.96 million written as ~$52M; 1,250 as 1250."""
        it = item(summary="valued at approximately $51.96 million over 1,250 racks")
        c = call(quote="approximately $51.96 million", why="A ~$52M sale over 1250 racks.")
        kept, _ = check(c, it, {})
        assert kept is not None

    def test_rounding_is_bounded(self):
        it = item(summary="valued at approximately $51.96 million")
        c = call(quote="approximately $51.96 million", why="A $55M sale.")
        assert check(c, it, {})[0] is None

    def test_unknown_claims_are_dropped_and_known_ones_get_their_kind(self):
        c = call(claims=[ClaimLink(claim_id="c1", effect="supports"),
                         ClaimLink(claim_id="nope", effect="SUPPORTS")])  # fmt: skip
        kept, problems = check(c, item(), {"c1": "INVALIDATION"})
        assert [(x.claim_id, x.effect, x.kind) for x in kept.claims] == [
            ("c1", "SUPPORTS", "INVALIDATION")
        ]
        assert "unknown claims" in problems[0]

    def test_off_topic_keeps_no_call(self):
        c = call(about_company=False, direction="NEGATIVE", materiality="HIGH",
                 claims=[ClaimLink(claim_id="c1", effect="CONTRADICTS")])  # fmt: skip
        kept, _ = check(c, item(), {"c1": "DRIVER"})
        assert kept.direction is Direction.NEUTRAL and kept.materiality is Materiality.LOW
        assert kept.claims == []


class TestThesisDirection:
    @pytest.mark.parametrize(
        "effect,kind,against",
        [
            ("SUPPORTS", "INVALIDATION", True),  # the kill condition is coming true
            ("SUPPORTS", "RISK", True),
            ("SUPPORTS", "DRIVER", False),
            ("CONTRADICTS", "DRIVER", True),  # what must go right is not
            ("CONTRADICTS", "KPI", True),
            ("CONTRADICTS", "INVALIDATION", False),
        ],
    )
    def test_against_thesis(self, effect, kind, against):
        assert ClaimLink(claim_id="c", effect=effect, kind=kind).against_thesis is against


# ── Judging, with the store ───────────────────────────────────────────────


@pytest.fixture
def db(tmp_path: Path):
    return tmp_path / "research.db"


@pytest.fixture
def stores(db):
    s, conn = DaemonStore(db), sqlite3.connect(str(db))
    yield s, conn
    s.close()
    conn.close()


def fake(calls):
    seen = []

    def complete(system, user):
        seen.append(user)
        return Draft(judgments=calls)

    complete.seen = seen
    return complete


class TestJudgeSymbol:
    def test_judges_stores_and_never_twice(self, stores):
        s, conn = stores
        s.save_source_item(item())
        c = fake([call()])
        js, problems = judge_symbol(s, conn, "NBIS", NOW, complete=c, model="m")
        assert len(js) == 1 and problems == [] and js[0].model == "m"
        again, _ = judge_symbol(s, conn, "NBIS", NOW, complete=c, model="m")
        assert again == [] and len(c.seen) == 1  # no second model call

    def test_nothing_to_judge_calls_no_model(self, stores):
        s, conn = stores
        c = fake([])
        assert judge_symbol(s, conn, "NBIS", NOW, complete=c) == ([], [])
        assert c.seen == []

    def test_old_items_are_left_alone(self, stores):
        s, conn = stores
        s.save_source_item(item(days_ago=judge.JUDGE_WINDOW_DAYS + 1))
        assert judge_symbol(s, conn, "NBIS", NOW, complete=fake([call()]))[0] == []

    def test_a_filing_without_a_lead_is_not_judged(self, stores):
        s, conn = stores
        s.save_source_item(item(tier=SourceTier.PRIMARY, summary=None))
        assert judge_symbol(s, conn, "NBIS", NOW, complete=fake([call()]))[0] == []

    def test_the_same_headline_twice_is_judged_once(self, stores):
        s, conn = stores
        s.save_source_item(item(url="https://a"))
        s.save_source_item(item(url="https://b"))
        c = fake([call()])
        js, _ = judge_symbol(s, conn, "NBIS", NOW, complete=c)
        assert len(js) == 1 and c.seen[0].count("Nebius to hike prices") == 1

    def test_an_item_the_model_skipped_is_named(self, stores):
        s, conn = stores
        s.save_source_item(item())
        js, problems = judge_symbol(s, conn, "NBIS", NOW, complete=fake([]))
        assert js == [] and "not judged by the model" in problems[0]

    def test_a_rejected_judgment_is_retried_next_run(self, stores):
        s, conn = stores
        s.save_source_item(item())
        judge_symbol(s, conn, "NBIS", NOW, complete=fake([call(quote="invented")]))
        js, _ = judge_symbol(s, conn, "NBIS", NOW, complete=fake([call()]))
        assert len(js) == 1

    def test_a_model_failure_is_a_problem_not_a_crash(self, stores):
        s, conn = stores
        s.save_source_item(item())

        def boom(system, user):
            raise TimeoutError("slow")

        js, problems = judge_symbol(s, conn, "NBIS", NOW, complete=boom)
        assert js == [] and "model failed" in problems[0]

    def test_batches_of_ten(self, stores):
        s, conn = stores
        for n in range(12):
            s.save_source_item(item(title=f"story {n}", url=f"https://x/{n}",
                                    summary=f"GPU prices rise 10% case {n}"))  # fmt: skip
        c = fake([call(id=f"N{n}", why="Prices rise 10%.") for n in range(1, 11)])
        js, _ = judge_symbol(s, conn, "NBIS", NOW, complete=c)
        assert len(c.seen) == 2 and len(js) == 12

    def test_claims_are_offered_with_their_kind(self, stores):
        s, conn = stores
        s.save_source_item(item())
        s.save_claim("NBIS", Claim(kind=ClaimKind.INVALIDATION, text="raises over 10%"))
        c = fake([call()])
        judge_symbol(s, conn, "NBIS", NOW, complete=c)
        assert "[INVALIDATION]: raises over 10%" in c.seen[0]

    def test_a_new_prompt_rejudges(self, stores, monkeypatch):
        s, conn = stores
        s.save_source_item(item())
        judge_symbol(s, conn, "NBIS", NOW, complete=fake([call()]))
        monkeypatch.setattr(judge, "SYSTEM_PROMPT", judge.SYSTEM_PROMPT + " v2")
        js, _ = judge_symbol(s, conn, "NBIS", NOW, complete=fake([call()]))
        assert len(js) == 1


# ── What followed ─────────────────────────────────────────────────────────


def judgment(published, judged=None, **kw):
    base = dict(
        key="k",
        symbol="NBIS",
        published_at=published,
        title="t",
        provider="p",
        tier="AGGREGATOR",
        about_company=True,
        event_type="PRODUCT",
        direction="NEGATIVE",
        materiality="HIGH",
        novelty="NEW",
        basis="REPORTED",
        quote="q",
        why="w",
        prompt_version="v",
        judged_at=judged or published,
    )
    return Judgment(**{**base, **kw})


def closes(start=date(2026, 9, 1), n=40):
    out, d, px = {}, start, 100.0
    while len(out) < n:
        if mc.is_trading_day(d):
            out[d] = px
            px += 1
        d += timedelta(days=1)
    return out


class TestOutcomes:
    def test_after_the_close_the_entry_is_the_next_close(self):
        """Published Tuesday 18:00: the first close it could be acted on is Wednesday's."""
        c = closes()
        j = judgment(datetime(2026, 9, 8, 18, 0, tzinfo=ET))
        o = outcomes_for(j, c)
        assert o["d0"] == pytest.approx(c[date(2026, 9, 9)] / c[date(2026, 9, 8)] - 1)
        assert o["d1"] == pytest.approx(c[date(2026, 9, 10)] / c[date(2026, 9, 9)] - 1)

    def test_judged_later_than_published_counts_from_the_judgment(self):
        c = closes()
        j = judgment(
            datetime(2026, 9, 8, 10, 0, tzinfo=ET), judged=datetime(2026, 9, 10, 8, 30, tzinfo=ET)
        )
        o = outcomes_for(j, c)
        assert o["d1"] == pytest.approx(c[date(2026, 9, 11)] / c[date(2026, 9, 10)] - 1)

    def test_published_on_a_weekend(self):
        c = closes()
        j = judgment(datetime(2026, 9, 12, 12, 0, tzinfo=ET))  # Saturday
        o = outcomes_for(j, c)
        assert o["d0"] == pytest.approx(c[date(2026, 9, 14)] / c[date(2026, 9, 11)] - 1)

    def test_not_enough_history_leaves_the_horizon_empty(self):
        c = closes(n=10)
        o = outcomes_for(judgment(datetime(2026, 9, 2, 18, 0, tzinfo=ET)), c)
        assert "d5" in o and "d20" not in o

    def test_no_prices_after_it(self):
        assert outcomes_for(judgment(datetime(2026, 12, 1, tzinfo=ET)), closes()) == {}

    def test_fill_is_idempotent(self, stores):
        s, conn = stores
        st = NewsJudgmentStore(conn)
        st.add(judgment(datetime(2026, 9, 2, 18, 0, tzinfo=ET)))
        assert judge.fill_outcomes(conn, lambda sym: closes()) == 1
        assert judge.fill_outcomes(conn, lambda sym: closes()) == 0
        assert st.list()[0].outcomes["d20"] is not None

    def test_a_symbol_without_prices_is_skipped(self, stores):
        s, conn = stores
        NewsJudgmentStore(conn).add(judgment(datetime(2026, 9, 2, 18, 0, tzinfo=ET)))
        assert judge.fill_outcomes(conn, lambda sym: None) == 0


class TestEvaluateAndSummary:
    def test_too_little_is_undetermined(self):
        class B:
            def get(self, sym, k):
                return 0.0

        js = [judgment(datetime(2026, 9, 2, 18, 0, tzinfo=ET), outcomes={"d1": -0.05, "d5": -0.1})]
        cells = judge.evaluate(js, B())
        assert {c["verdict"] for c in cells} == {"UNDETERMINED"}
        assert cells[0]["group"] == "NEGATIVE/HIGH"

    def test_off_topic_is_its_own_group(self):
        assert judgment(NOW, about_company=False).group == "off-topic"

    def test_summary_counts_material_calls_only(self):
        js = [
            judgment(NOW, direction="NEGATIVE", materiality="HIGH",
                     claims=[ClaimLink(claim_id="c", effect="SUPPORTS", kind="INVALIDATION")]),
            judgment(NOW, direction="POSITIVE", materiality="LOW"),
            judgment(NOW, direction="POSITIVE", materiality="MEDIUM",
                     claims=[ClaimLink(claim_id="d", effect="SUPPORTS", kind="DRIVER")]),
            judgment(NOW, about_company=False),
        ]  # fmt: skip
        s = summary(js)
        assert s["items"] == 4 and s["about"] == 3
        assert s["negative"] == 1 and s["positive"] == 1 and s["high"] == 1
        assert s["against_thesis"] == 1 and s["for_thesis"] == 1


def test_the_job_is_scheduled_twice_a_day():
    from advisor.daemon.supervisor import build_registry

    names = {j.name: j for j in build_registry()}
    assert names["news_judge"].trigger.trading_days_only is False
    assert names["news_judge_close"].trigger.at.hour == 17
