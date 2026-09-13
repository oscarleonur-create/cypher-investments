"""Reading a 6-K, where the substance is never in the filing.

A foreign private issuer files no 8-K and no 10-Q. Its 6-K body is about 1,500
characters of SEC boilerplate saying "a report is attached"; the disclosure is
in an EX-99 exhibit and there are no item codes.

The cost of not reading it, on the live book: Nebius issued $5.75bn of
convertible senior notes across three August filings — 9.4% of its market cap,
on a position worth 14.4% of the portfolio — and every one was archived as
Tier C "other events".
"""

from __future__ import annotations

import pytest
from advisor.news.classify import FilingKind, Materiality
from advisor.news.foreign import classify_headline, exhibit_text, is_proposal

# The real EX-99.1 openings, verbatim from EDGAR.
CLOSING = (
    "Exhibit 99.1 Nebius Group announces closing of private offering of convertible "
    "senior notes, with aggregate gross proceeds of approximately $5.75 billion "
    "Amsterdam, August 24, 2026—Nebius Group N.V."
)
PRICING = (
    "Exhibit 99.1 Nebius Group announces pricing of upsized private offering of "
    "$5.0 billion of convertible senior notes Amsterdam, August 19, 2026"
)
PROPOSED = (
    "Exhibit 99.1 Nebius Group announces proposed private offering of $4.50 billion "
    "of convertible senior notes Amsterdam, August 19, 2026"
)
RESULTS = (
    "Exhibit 99.1 Nebius reports second quarter 2026 financial results Amsterdam, "
    "August 12, 2026"
)
AGM = (
    "EXHIBIT 99.1 Nebius Group N.V. announces results of its Annual General Meeting "
    "Amsterdam, the Netherlands — August 26, 2026"
)
PARTNERSHIP = (
    "Exhibit 99.1 Palantir and Nebius partner to deliver a complete sovereign AI "
    "stack to Palantir customers"
)


class TestTheRealHeadlines:
    def test_a_convertible_offering_is_dilution_and_high(self):
        result = classify_headline(CLOSING)
        assert result.kind is FilingKind.DILUTION
        assert result.materiality is Materiality.HIGH

    def test_results_are_recognised(self):
        assert classify_headline(RESULTS).kind is FilingKind.RESULTS

    def test_an_agm_is_low_materiality(self):
        assert classify_headline(AGM).materiality is Materiality.LOW

    def test_a_partnership_is_a_material_agreement(self):
        assert classify_headline(PARTNERSHIP).kind is FilingKind.MATERIAL_AGREEMENT

    def test_an_agm_announcing_results_is_not_earnings(self):
        """'announces results of its Annual General Meeting' is not earnings."""
        assert classify_headline(AGM).kind is not FilingKind.RESULTS


class TestDealStages:
    """One transaction, three filings. Only some of them are facts."""

    def test_a_proposal_is_recognised_as_an_intention(self):
        assert is_proposal(PROPOSED) is True

    def test_a_pricing_is_not_a_proposal(self):
        assert is_proposal(PRICING) is False

    def test_a_closing_is_not_a_proposal(self):
        assert is_proposal(CLOSING) is False

    @pytest.mark.parametrize(
        "text",
        ["the company intends to offer notes", "the company plans to issue shares"],
    )
    def test_other_intention_phrasings(self, text):
        assert is_proposal(text) is True


class TestFailsClosed:
    @pytest.mark.parametrize("text", ["", None, "   "])
    def test_an_empty_headline_is_other_not_a_guess(self, text):
        result = classify_headline(text)
        assert result.kind is FilingKind.OTHER
        assert result.materiality is Materiality.LOW

    def test_an_unrecognised_headline_stays_other(self):
        result = classify_headline("Nebius publishes its sustainability statement")
        assert result.kind is FilingKind.OTHER

    def test_only_the_headline_is_read(self):
        """A press release body cites everything the company has ever done."""
        text = "Nebius publishes its ESG report. " + ("filler " * 300)
        text += "In August the company closed an offering of convertible senior notes."
        assert classify_headline(text, headline_chars=60).kind is FilingKind.OTHER


class TestExhibitExtraction:
    class FakeAttachment:
        def __init__(self, doc_type, text=""):
            self.document_type, self._text = doc_type, text

        def text(self):
            if self._text is None:
                raise RuntimeError("unreadable")
            return self._text

    class FakeFiling:
        accession_no = "x"

        def __init__(self, attachments):
            self.attachments = attachments

    def test_the_ex99_exhibit_is_returned(self):
        filing = self.FakeFiling(
            [self.FakeAttachment("6-K", "cover page"), self.FakeAttachment("EX-99.1", CLOSING)]
        )
        assert "convertible" in exhibit_text(filing)

    def test_indentures_and_images_are_skipped(self):
        filing = self.FakeFiling(
            [
                self.FakeAttachment("EX-4.1", "indenture"),
                self.FakeAttachment("GRAPHIC", "image"),
                self.FakeAttachment("EX-99.1", CLOSING),
            ]
        )
        assert "convertible" in exhibit_text(filing)
        assert "indenture" not in exhibit_text(filing)

    def test_a_filing_with_no_exhibit_yields_empty(self):
        assert exhibit_text(self.FakeFiling([self.FakeAttachment("6-K", "cover")])) == ""

    def test_an_unreadable_exhibit_does_not_raise(self):
        filing = self.FakeFiling([self.FakeAttachment("EX-99.1", None)])
        assert exhibit_text(filing) == ""

    def test_a_filing_without_attachments_does_not_raise(self):
        class NoAttachments:
            accession_no = "x"

            @property
            def attachments(self):
                raise RuntimeError("none")

        assert exhibit_text(NoAttachments()) == ""
