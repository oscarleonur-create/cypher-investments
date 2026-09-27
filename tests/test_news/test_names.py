"""How a headline names the company: the real titles the 2026-09-27 audit misread."""

from __future__ import annotations

import pytest
from advisor.news.entities import resolve_entity
from advisor.news.names import COMPANY_NAMES, names_company, query_names

SEC = {
    "COHR": "COHERENT CORP.",
    "SPCX": "SPACE EXPLORATION TECHNOLOGIES CORP",
    "AMZN": "AMAZON COM INC",
    "TE": "T1 Energy Inc.",
    "CRDO": "Credo Technology Group Holding Ltd",
}


def kept(symbol, title):
    return resolve_entity(symbol, text=title, company_name=SEC.get(symbol)).resolved


@pytest.mark.parametrize(
    "symbol,title",
    [
        # Kept by the registered name before; none is about the company.
        ("COHR", "Automating coherent long-form video generation"),
        ("COHR", "How are coherent pluggables used in subsea networks"),
        ("COHR", "Electro-optic modulation of coherent and incoherent mid-IR radiation"),
        ("COHR", "Automating Coherent Long-Form Video Generation"),  # title case: weak, unpinned
        ("SPCX", "A call to boost European space exploration"),
        ("SPCX", "New Space Exploration merit badge digital resource guide launches"),
        (
            "SPCX",
            "Popular Astronomy Club of the Quad Cities celebrating 90 years of space exploration",
        ),
        ("TE", "TE Stock Drifts Lower As Losses Weigh On Sentiment"),  # TE alone is a word
    ],
)
def test_common_words_are_not_the_company(symbol, title):
    assert not kept(symbol, title)


@pytest.mark.parametrize(
    "symbol,title",
    [
        # Dropped by the registered name before; each is about the company.
        ("SPCX", "SpaceX plans to launch one of its most ambitious flights in South Texas"),
        ("AMZN", "Amazon Blocks Meta's Muse Agent From Shopping On Its Platform"),
        # Kept before and still kept.
        ("COHR", "Coherent Has Ripped 59% in 2026: Is It Too Late to Buy COHR Stock Now?"),
        ("COHR", "CUbIQ Technologies and Coherent Corp. demonstrate physical-layer security"),
        ("TE", "T1 Energy falls 13% as investors rotate to defensive sectors"),
        (
            "CRDO",
            "Credo Technology Crashed for 3 Months: This Wall Street Pro Says It's About to Double",
        ),
        ("CRDO", "Credo's stock jumps as earnings beat"),  # weak name, pinned, possessive
    ],
)
def test_the_company_under_the_names_headlines_use(symbol, title):
    assert kept(symbol, title)


def test_a_weak_name_alone_is_not_enough():
    """The trade-off, stated: a real Credo headline with nothing to pin it is missed."""
    assert names_company("CRDO", "Why Credo's New 1.6T Optics Matter") is False


def test_an_unlisted_symbol_falls_back_to_the_registered_name():
    assert names_company("ZZZZ", "anything") is None
    assert resolve_entity(
        "ZZZZ", text="Zeta Widgets beats", company_name="Zeta Widgets Inc."
    ).resolved


def test_names_match_as_written_not_inside_words():
    assert names_company("SPCX", "SpaceXAI ships Grok") is False
    assert names_company("META", "A meta-analysis of trials") is False


def test_the_query_asks_for_strong_names_then_weak():
    assert query_names("COHR") == ("Coherent Corp", "Coherent")
    assert query_names("ZZZZ") == ()


def test_every_listed_name_is_non_empty_text():
    for symbol, names in COMPANY_NAMES.items():
        assert names["strong"], symbol
        assert all(n.strip() for n in (*names["strong"], *names.get("weak", ()))), symbol
