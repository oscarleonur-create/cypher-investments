"""Entity resolution — the component whose absence started Phase 2c.

yfinance returned three headlines filed under AAOI. None named AAOI; two were
about Nvidia and one about HPE. Everything here exists so that cannot recur,
and so that the opposite failure — attaching a story to the wrong holding —
cannot happen either.
"""

from __future__ import annotations

import pytest
from advisor.news.entities import (
    AMBIGUOUS_TICKERS,
    is_ambiguous,
    normalize_name,
    resolve_entity,
)
from advisor.news.models import MatchMethod

AAOI_NAME = "APPLIED OPTOELECTRONICS, INC."


class TestNameNormalisation:
    def test_strips_corporate_suffixes(self):
        assert normalize_name("Applied Optoelectronics, Inc.") == "applied optoelectronics"
        assert normalize_name("Nebius Group N.V.") == "nebius"
        assert normalize_name("Ouster, Inc.") == "ouster"

    def test_is_case_and_punctuation_insensitive(self):
        assert normalize_name("APPLIED OPTOELECTRONICS, INC.") == normalize_name(
            "applied optoelectronics inc"
        )

    def test_handles_empty_and_none_safely(self):
        assert normalize_name("") == ""
        assert normalize_name(None) == ""


class TestMatchStrength:
    def test_cik_wins_over_everything(self):
        match = resolve_entity("AAOI", text="unrelated text", cik=1158114)
        assert match.method is MatchMethod.CIK
        assert match.confidence == 1.0
        assert match.cik == 1158114

    def test_provider_tag_beats_prose(self):
        match = resolve_entity("AAOI", text="Applied Optoelectronics", provider_tags=["AAOI"])
        assert match.method is MatchMethod.PROVIDER_TAG

    def test_company_name_in_the_title_resolves(self):
        match = resolve_entity(
            "AAOI",
            text="Applied Optoelectronics Launches Up to $600 Million ATM Program",
            company_name=AAOI_NAME,
        )
        assert match.method is MatchMethod.COMPANY_NAME

    def test_cashtag_resolves(self):
        assert resolve_entity("AAOI", text="watching $AAOI today").method is MatchMethod.CASHTAG

    def test_bare_distinctive_ticker_resolves_weakly(self):
        match = resolve_entity("AAOI", text="AAOI fell sharply")
        assert match.method is MatchMethod.TICKER_TOKEN
        assert match.confidence < 0.8


class TestTheFailureThatStartedThis:
    @pytest.mark.parametrize(
        "headline",
        [
            "These 3 AI Stocks Are Way Off Their Highs. Is the Pullback a Buying Opportunity?",
            "HPE Cannot Build AI Servers Fast Enough. Is the Stock Still Cheap?",
            "This Little-Known AI Company Is Quietly Building for the Data Center Boom",
        ],
    )
    def test_real_yfinance_headlines_for_aaoi_are_all_rejected(self, headline):
        match = resolve_entity("AAOI", text=headline, company_name=AAOI_NAME)
        assert not match.resolved
        assert match.confidence == 0.0

    def test_a_headline_that_does_name_the_company_is_kept(self):
        match = resolve_entity(
            "AAOI",
            text="Applied Optoelectronics (AAOI) Announces $600 Million At The Market Offering",
            company_name=AAOI_NAME,
        )
        assert match.resolved


class TestAmbiguousTickers:
    def test_the_books_two_letter_holding_is_ambiguous(self):
        """TE is T1 Energy. A bare 'TE' token is evidence of nothing."""
        assert is_ambiguous("TE")
        assert "TE" in AMBIGUOUS_TICKERS

    def test_ambiguous_ticker_is_not_matched_by_token(self):
        match = resolve_entity("TE", text="the te reaction rate was measured")
        assert not match.resolved

    def test_ambiguous_ticker_still_matches_on_company_name(self):
        match = resolve_entity(
            "TE", text="T1 Energy reports quarterly results", company_name="T1 Energy Inc."
        )
        assert match.method is MatchMethod.COMPANY_NAME

    def test_ambiguous_ticker_still_matches_on_cashtag(self):
        assert resolve_entity("TE", text="$TE is moving").method is MatchMethod.CASHTAG

    @pytest.mark.parametrize("symbol", ["ON", "ALL", "IT", "KEY", "GO"])
    def test_common_word_tickers_are_all_guarded(self, symbol):
        assert is_ambiguous(symbol)

    def test_four_letter_tickers_are_distinctive_enough(self):
        for symbol in ("AAOI", "CRDO", "NBIS", "COHR", "OUST"):
            assert not is_ambiguous(symbol)


class TestSubstringSafety:
    def test_a_ticker_inside_a_longer_word_does_not_match(self):
        """'AMD' must not match 'AMDOCS' or 'GAMDAY'."""
        assert not resolve_entity("AMD", text="AMDOCS reported results").resolved
        assert not resolve_entity("AMD", text="the GAMDAY promotion").resolved

    def test_a_ticker_adjacent_to_punctuation_does_match(self):
        assert resolve_entity("AMD", text="shares of AMD, up 3%").resolved
        assert resolve_entity("AMD", text="(AMD) gained").resolved

    def test_a_short_company_name_does_not_match_loosely(self):
        """A two-character 'name' must not match half the corpus."""
        match = resolve_entity("XX", text="anything at all here", company_name="Co")
        assert not match.resolved

    def test_empty_text_resolves_to_nothing(self):
        assert not resolve_entity("AAOI", text="", company_name=AAOI_NAME).resolved
