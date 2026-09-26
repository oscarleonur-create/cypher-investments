"""Hypotheses: the model proposes in a fixed form; the code decides what is true."""

from __future__ import annotations

import random
import sqlite3
from datetime import date, timedelta

import pytest
from advisor.learning.evaluate import Baselines, Record
from advisor.learning.hypotheses import (
    MAX_HYPOTHESES,
    Draft,
    Hypothesis,
    HypothesisStore,
    Status,
    examine,
    generate_and_test,
    prompt_version,
    summary_for_model,
)

DAYS = [date(2025, 1, 2) + timedelta(days=i) for i in range(600) if (i % 7) < 5][:400]


def flat(symbol):
    return {d: 100.0 for d in DAYS}


def recs(effect_cheap: float, effect_rich: float, horizon="next_close", seed=1):
    rng = random.Random(seed)
    out = []
    for d in DAYS:
        for pct, eff in ((0.1, effect_cheap), (0.9, effect_rich)):
            out.append(
                Record(
                    "entry", "v", "action ENTER", d, "AAA",
                    {horizon: eff + rng.gauss(0, 0.002)},
                    extra={"features": {"ps_percentile": pct, "thesis": "intact"}},
                )
            )  # fmt: skip
    return out


def hyp(**kw) -> Hypothesis:
    base = dict(
        group="action ENTER",
        feature="ps_percentile",
        op="<=",
        value=0.25,
        horizon="next_close",
        expect="better",
        why="cheap entries looked better in the cells",
    )
    base.update(kw)
    return Hypothesis(**base)


class TestExamine:
    def test_supported(self):
        r = examine(hyp(), recs(0.01, 0.0), Baselines(flat))
        assert r.status is Status.SUPPORTED
        assert r.matching["excess"] > r.rest["excess"]

    def test_refuted_when_it_is_the_other_way(self):
        r = examine(hyp(), recs(-0.01, 0.0), Baselines(flat))
        assert r.status is Status.REFUTED

    def test_worse_expectation_supported(self):
        r = examine(hyp(expect="worse"), recs(-0.01, 0.0), Baselines(flat))
        assert r.status is Status.SUPPORTED

    def test_inconclusive_when_no_difference(self):
        r = examine(hyp(), recs(0.0, 0.0), Baselines(flat))
        assert r.status is Status.INCONCLUSIVE

    def test_invented_feature_is_invalid(self):
        r = examine(hyp(feature="analyst_mood"), recs(0.01, 0.0), Baselines(flat))
        assert r.status is Status.INVALID and "not recorded" in r.reason

    def test_unknown_group_is_invalid(self):
        r = examine(hyp(group="action BUY_EVERYTHING"), recs(0.01, 0.0), Baselines(flat))
        assert r.status is Status.INVALID

    def test_ordering_a_string_is_invalid(self):
        r = examine(hyp(feature="thesis", op=">", value="intact"), recs(0.01, 0.0), Baselines(flat))
        assert r.status is Status.INVALID

    def test_equality_on_a_string_works(self):
        r = examine(
            hyp(feature="thesis", op="==", value="intact"), recs(0.01, 0.0), Baselines(flat)
        )
        # every record matches: nothing to compare against
        assert r.status is Status.INCONCLUSIVE and r.rest["n"] == 0

    def test_records_without_the_feature_are_left_out(self):
        rs = recs(0.01, 0.0)
        for r in rs[::2]:
            r.extra["features"].pop("ps_percentile")
        out = examine(hyp(), rs, Baselines(flat))
        assert out.matching["n"] + out.rest["n"] == len(rs) // 2

    def test_overlapping_horizons_are_blocked(self):
        # 400 sessions of d20 returns are 20 independent windows, not 400.
        r = examine(hyp(horizon="d20"), recs(0.02, 0.0, horizon="d20"), Baselines(flat))
        assert r.matching["windows"] == 20


class FakeModel:
    def __init__(self, draft: Draft):
        self.draft, self.seen = draft, []

    def __call__(self, system, user):
        self.seen.append(user)
        return self.draft


@pytest.fixture
def store(tmp_path):
    conn = sqlite3.connect(str(tmp_path / "r.db"))
    yield HypothesisStore(conn)
    conn.close()


class TestRound:
    def test_the_verdict_is_the_codes_not_the_models(self, store):
        model = FakeModel(Draft(hypotheses=[hyp(why="I am certain this works")]))
        out = generate_and_test(recs(-0.01, 0.0), [], [], Baselines(flat), store,
                                complete=model, model="m")  # fmt: skip
        assert out[0]["result"]["status"] == "REFUTED"
        (row,) = store.list()
        assert row["status"] == "REFUTED" and row["prompt_version"] == prompt_version()

    def test_at_most_the_cap_is_tested(self, store):
        many = Draft.model_construct(hypotheses=[hyp()] * (MAX_HYPOTHESES + 3))
        out = generate_and_test(recs(0.01, 0.0), [], [], Baselines(flat), store,
                                complete=FakeModel(many), model="m")  # fmt: skip
        assert len(out) == MAX_HYPOTHESES

    def test_the_model_sees_aggregates_not_records(self, store):
        model = FakeModel(Draft(hypotheses=[]))
        generate_and_test(recs(0.01, 0.0), [], [], Baselines(flat), store,
                          complete=model, model="m")  # fmt: skip
        (seen,) = model.seen
        assert "AAA" not in seen and "action ENTER (800)" in seen

    def test_no_model_configured(self, store, monkeypatch):
        from advisor.learning import hypotheses

        monkeypatch.setattr(hypotheses, "_default_model", lambda: None)
        with pytest.raises(RuntimeError, match="no LLM"):
            generate_and_test([], [], [], Baselines(flat), store)

    def test_draft_caps_the_count(self):
        with pytest.raises(ValueError):
            Draft(hypotheses=[hyp()] * (MAX_HYPOTHESES + 1))


def test_summary_lists_groups_features_and_cells():
    s = summary_for_model([], [], {"action ENTER": 10})
    assert "action ENTER (10)" in s and "ps_percentile" in s
