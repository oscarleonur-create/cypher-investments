"""Whether a claim's trigger can fire at all, for this symbol, today.

The gap these close was found writing the NBIS thesis by hand. Two of its six
claims read a figure out of a results filing — `revenue_growth_yoy` and
`ai_cloud_adj_ebitda`. No `FILING_RESULTS` payload has ever carried a figure:
the classifier stores the form, the label and the URL, and nothing parses the
statements. The CLI still printed them under "checked by", because
`KNOWN_FIELDS` had no entry for that kind and an absent entry abstains.

The two unreachable claims were the two that carried the thesis. A thesis that
believes it is watched when it is not is worse than one known to be unwatched.
"""

from __future__ import annotations

from pathlib import Path

import pytest
from advisor.daemon.store import DaemonStore
from advisor.thesis.models import Claim, ClaimKind, Comparator, Trigger
from advisor.thesis.reachability import KNOWN_FIELDS, audit, claim_reachability


@pytest.fixture
def store(tmp_path: Path):
    s = DaemonStore(tmp_path / "research.db")
    yield s
    s.close()


def claim(**trigger) -> Claim:
    return Claim(
        symbol="NBIS",
        text="a claim",
        kind=ClaimKind.INVALIDATION,
        trigger=Trigger(**trigger),
    )


# --- the live failure -------------------------------------------------------


def test_results_filing_cannot_carry_a_revenue_figure(store):
    """The NBIS claim that reported itself as monitored and never could fire."""
    r = claim_reachability(
        store,
        "NBIS",
        claim(
            event_kinds=["FILING_RESULTS"],
            field="revenue_growth_yoy",
            comparator=Comparator.BELOW,
            threshold=1.0,
        ),
    )
    assert r.blocked
    assert "revenue_growth_yoy" in r.reason
    assert "FILING_RESULTS" in r.reason


def test_results_filing_is_reachable_on_a_field_it_does_carry(store):
    r = claim_reachability(
        store,
        "NBIS",
        claim(event_kinds=["FILING_RESULTS"], field="form", comparator=Comparator.HAPPENS),
    )
    assert r.reachable


def test_a_bare_results_trigger_stays_reachable(store):
    """No field means "any results filing", which needs no payload key."""
    assert claim_reachability(store, "NBIS", claim(event_kinds=["FILING_RESULTS"])).reachable


# --- the multi-kind lookup --------------------------------------------------


def test_a_field_on_two_kinds_is_still_checked(store):
    """The old table keyed on the whole kind set, so this combination missed
    it entirely and went unchecked."""
    r = claim_reachability(
        store,
        "NBIS",
        claim(
            event_kinds=["FILING_RESULTS", "FILING_MERGER"],
            field="revenue_growth_yoy",
            comparator=Comparator.BELOW,
            threshold=1.0,
        ),
    )
    assert r.blocked


def test_one_carrier_among_several_kinds_is_enough(store):
    """Dilution carries `dilution_pct` and results does not. A claim watching
    both is reachable — the dilution arm can fire."""
    r = claim_reachability(
        store,
        "NBIS",
        claim(
            event_kinds=["FILING_RESULTS", "FILING_DILUTION"],
            field="dilution_pct",
            comparator=Comparator.ABOVE,
            threshold=0.10,
        ),
    )
    assert r.reachable


def test_one_unknown_kind_makes_the_whole_check_abstain(store):
    """An absent entry means "not checked", never "invalid" — so a kind this
    table does not know must not convict the kinds beside it."""
    r = claim_reachability(
        store,
        "NBIS",
        claim(
            event_kinds=["FILING_RESULTS", "NEWS_CONTEXT"],
            field="revenue_growth_yoy",
            comparator=Comparator.BELOW,
            threshold=1.0,
        ),
    )
    assert r.reachable


# --- fields that only appear sometimes --------------------------------------


def test_conditional_fields_count_as_carried(store):
    """`pct_of_cap` is written only when a market cap was available. The
    question is whether the kind can ever carry it, not whether it always
    does — otherwise a real threshold would be called unreachable."""
    r = claim_reachability(
        store,
        "NBIS",
        claim(
            event_kinds=["INSIDER_SELLING_CLUSTER"],
            field="pct_of_cap",
            comparator=Comparator.ABOVE,
            threshold=0.002,
        ),
    )
    assert r.reachable


def test_insider_cluster_rejects_a_field_it_never_carries(store):
    r = claim_reachability(
        store,
        "NBIS",
        claim(
            event_kinds=["INSIDER_SELLING_CLUSTER"],
            field="unrealized_pct",
            comparator=Comparator.BELOW,
            threshold=-0.2,
        ),
    )
    assert r.blocked


def test_dilution_keeps_the_shared_filing_fields(store):
    """The old dilution entry listed only its three sizing extras, so a claim
    on `form` — which every filing payload carries — was called unreachable."""
    r = claim_reachability(store, "NBIS", claim(event_kinds=["FILING_DILUTION"], field="form"))
    assert r.reachable


# --- boundaries between position kinds --------------------------------------


def test_drawdown_does_not_carry_previous_pct(store):
    """Only the two crossings compare against a prior reading; a drawdown is
    a level, so `previous_pct` on it can never fire."""
    assert claim_reachability(
        store,
        "NBIS",
        claim(event_kinds=["DEEP_DRAWDOWN"], field="previous_pct", threshold=0.0),
    ).blocked
    assert claim_reachability(
        store,
        "NBIS",
        claim(event_kinds=["STOP_BREACHED"], field="previous_pct", threshold=0.0),
    ).reachable


def test_concentration_is_a_book_property_and_carries_no_entry_price(store):
    assert claim_reachability(
        store, "NBIS", claim(event_kinds=["CONCENTRATION_WARNING"], field="entry", threshold=1.0)
    ).blocked
    assert claim_reachability(
        store, "NBIS", claim(event_kinds=["CONCENTRATION_WARNING"], field="weight", threshold=0.2)
    ).reachable


# --- the table itself -------------------------------------------------------


def test_every_known_kind_is_a_bare_string_not_a_set():
    """The key type is load-bearing: a frozenset key silently never matches."""
    assert all(isinstance(k, str) for k in KNOWN_FIELDS)


def test_audit_returns_one_verdict_per_claim(store):
    claims = [
        claim(event_kinds=["FILING_RESULTS"], field="revenue_growth_yoy", threshold=1.0),
        claim(event_kinds=["FILING_DILUTION"], field="dilution_pct", threshold=0.1),
        claim(),
    ]
    verdicts = audit(store, "NBIS", claims)
    assert [v.reachable for v in verdicts] == [False, True, False]


def test_an_empty_claim_list_audits_to_nothing(store):
    assert audit(store, "NBIS", []) == []


# --- what the CLI says a factor claim watches -------------------------------


def test_a_factor_claim_describes_the_condition_that_trips_it():
    """A MACRO_DRIVER claim states the exposure the thesis *wants* and fires
    when the measured loading contradicts it. The description named the wanted
    direction, so the NBIS claim leaning on negative BREADTH — which fires on
    a flip to positive — printed "BREADTH loading turns negative"."""
    leans_negative = Trigger(factor="BREADTH", comparator=Comparator.HAPPENS)
    assert leans_negative.describe() == "BREADTH loading turns positive (thesis leans negative)"

    leans_positive = Trigger(factor="DURATION", comparator=Comparator.ABOVE)
    assert leans_positive.describe() == "DURATION loading turns negative (thesis leans positive)"
