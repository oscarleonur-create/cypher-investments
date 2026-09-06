"""Offering-size extraction — which must fail closed.

A fabricated dilution percentage is far worse than a missing one. This repo
has shipped fabricated numbers before (catalysts.py dates, investment_memo.py
figures), so every test here that checks for absence matters as much as the
ones that check for a value.
"""

from __future__ import annotations

import pytest
from advisor.news.enrich import extract_offering_size

# The real opening of the AAOI 424B5, verbatim.
AAOI_TEXT = (
    "TABLE OF CONTENTS Filed pursuant to Rule 424(b)(5) Registration No. 333-283905 "
    "PROSPECTUS SUPPLEMENT $600,000,000 Applied Optoelectronics, Inc. Common Stock "
    "We have entered into an Equity Distribution Agreement with Raymond James & "
    "Associates, Inc. and Needham & Company, LLC, or the sales agreement, relating to "
    "shares of our common stock offered by this prospectus supplement. In accordance "
    "with the terms of the sales agreement, we may offer and sell through this "
    "prospectus supplement shares of our common stock having an aggregate offering "
    "price of up to $600,000,000 from time to time through Raymond James and Needham."
)


class TestTheRealFiling:
    def test_extracts_the_offering_amount(self):
        size = extract_offering_size(AAOI_TEXT)
        assert size is not None
        assert size.amount_usd == 600_000_000

    def test_keeps_the_sentence_that_proves_it(self):
        size = extract_offering_size(AAOI_TEXT)
        assert "aggregate offering price of up to $600,000,000" in size.quote

    def test_computes_dilution_against_market_cap(self):
        size = extract_offering_size(AAOI_TEXT)
        assert size.dilution_pct(8_960_160_678) == pytest.approx(0.067, abs=0.001)


class TestUnitScaling:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("having an aggregate offering price of up to $600 million.", 600e6),
            ("having an aggregate offering price of up to $1.2 billion.", 1.2e9),
            ("having an aggregate amount of up to $250,000,000.", 250e6),
            ("up to $75.5 million in aggregate offering price.", 75.5e6),
        ],
    )
    def test_units_are_scaled(self, text, expected):
        size = extract_offering_size(text)
        assert size is not None
        assert size.amount_usd == pytest.approx(expected)


class TestFailsClosed:
    def test_empty_text_returns_none(self):
        assert extract_offering_size("") is None
        assert extract_offering_size(None) is None

    def test_text_without_an_offering_returns_none(self):
        text = "The Company appointed a new director. The annual salary is $450,000."
        assert extract_offering_size(text) is None

    def test_a_dollar_figure_with_no_offering_context_is_ignored(self):
        """Prospectuses are full of unrelated dollar amounts."""
        assert extract_offering_size("We paid $12,000,000 in legal fees.") is None

    def test_absurd_amounts_are_rejected(self):
        assert extract_offering_size("aggregate offering price of up to $3.") is None
        assert extract_offering_size("aggregate offering price of up to $900 billion.") is None

    def test_a_page_number_next_to_context_is_not_a_size(self):
        assert extract_offering_size("up to $7 see aggregate offering price on page 4.") is None

    def test_malformed_numbers_do_not_raise(self):
        for text in (
            "aggregate offering price of up to $,,,.",
            "up to $. million aggregate amount",
        ):
            assert extract_offering_size(text) is None


class TestSelection:
    def test_the_largest_qualifying_amount_wins(self):
        text = (
            "We may sell shares having an aggregate offering price of up to $50,000,000. "
            "In addition we may sell shares having an aggregate offering price of up to "
            "$600,000,000 under the sales agreement."
        )
        assert extract_offering_size(text).amount_usd == 600_000_000

    def test_only_the_cover_is_read(self):
        """A figure far into the body must not be picked up."""
        buried = "aggregate offering price of up to $9,000,000."
        text = "cover text. " + ("filler. " * 4000) + buried
        assert extract_offering_size(text, max_chars=200) is None

    def test_dilution_against_zero_or_missing_market_cap_is_none(self):
        size = extract_offering_size(AAOI_TEXT)
        assert size.dilution_pct(0) is None
        assert size.dilution_pct(-1) is None


class TestConvertibleDebt:
    """Dilution is not only equity — T1 Energy sold $120m of convertible notes.

    An extractor that only understood "aggregate offering price" reported that
    Tier A event with no size at all, on a company worth $1.35bn.
    """

    TE_TEXT = (
        "Item 3.02. Unregistered Sale of Securities. On July 29, 2026, the Company "
        "entered into the Note Purchase Agreements pursuant to which it agreed to sell "
        "$120.0 million aggregate principal amount of the Convertible Notes to the "
        "Purchasers in a private placement."
    )

    def test_extracts_a_convertible_note_principal(self):
        size = extract_offering_size(self.TE_TEXT)
        assert size is not None
        assert size.amount_usd == 120_000_000

    def test_sizes_it_against_market_cap(self):
        size = extract_offering_size(self.TE_TEXT)
        assert size.dilution_pct(1_354_592_015) == pytest.approx(0.089, abs=0.001)

    def test_the_quote_names_the_instrument(self):
        assert "Convertible Notes" in extract_offering_size(self.TE_TEXT).quote
