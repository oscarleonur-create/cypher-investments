"""Leads: what a filing or article actually says, in its own words.

AAOI filed three 8-Ks in fifteen days, each archived only as "entered a
material definitive agreement" — two Houston factory leases, a property
purchase, and a ten-year factory lease in Ningbo. The item text says which;
nothing read it. Every text below is taken verbatim from EDGAR or the feed.
"""

from __future__ import annotations

from datetime import datetime, timezone

import pytest
from advisor.news.lead import (
    LEAD_CHARS,
    article_lead,
    eight_k_lead,
    exhibit_lead,
    lead_for,
    trim,
)
from advisor.news.models import EntityMatch, MatchMethod, SourceItem, SourceTier

AAOI_HOUSTON = (
    "Item 1.01 Entry into a Material Definitive Agreement. On August 31, 2026, Applied "
    'Optoelectronics, Inc. (the "Company") entered into two separate lease agreements '
    '(each, a "Lease" and collectively, the "Leases") with Hightower Phase II Owner, LLC, '
    'a Delaware limited liability company (the "Landlord"), for two industrial buildings '
    'to be constructed in Houston, Texas ("Building 4" and "Building 5"). The Landlord is '
    "an affiliate of Hightower Phase I Owner, LLC, the landlord under the Company's three "
    "previously disclosed lease agreements."
)
AAOI_2_03 = (
    "Item 2.03 Creation of a Direct Financial Obligation or an Obligation The information "
    "contained in Item 1.01 of this Current Report on Form 8-K is incorporated by reference "
    "into this Item 2.03."
)
AAOI_9_01 = (
    "Item 9.01 Financial Statements and Exhibits. (d) Exhibits Exhibit No. Description "
    "10.1+* Lease Agreement (Building 4)"
)
CEREBRAS_2_02 = (
    "Item 2.02 - Results of Operations and Financial Condition On June 23, 2026, Cerebras "
    "Systems Inc. The information in Item 2.02 of this Current Report on Form 8-K "
    "(including Exhibit 99.1) shall not be deemed “filed” for purposes of Section 18 of "
    "the Securities Exchange Act of 1934, as amended."
)
CEREBRAS_EX99 = (
    "Exhibit 99.1 Cerebras Systems Announces Strong First Quarter 2026 Results • GAAP "
    "quarterly revenue of $193.4 million ; record core revenue of $191.3 million , up 92% "
    "from a year ago • Announced a multi-year deal with OpenAI for 750MW valued at more "
    "than $20 billion • Cash of $1.2 billion"
)
AMD_EX99 = (
    "NEWS RELEASE Contact: Brandi Martina AMD Communications (512) 705-1720 "
    "brandi.martina@amd.com Liz Stine AMD Investor Relations (720) 652-3965 "
    "liz.stine@amd.com AMD Appoints Tim Ryan to Board of Directors"
)
GURUFOCUS = (
    "Logo\nLogo\n\n# Applied Optoelectronics Inc (AAOI) Shares Fall 4.0% -- GF Value Says "
    "Still Overvalued\n\nAuthor's Avatar\nArticle's Main Image\n\nOn September 01, 2026, "
    "Applied Optoelectronics Inc AAOI  shares fell 4.0%, closing at $103.39. The stock has "
    "experienced considerable price fluctuations, ranging from a 52-week low of $18.50 to a "
    "high of $233.67 over the past year.\n\n### Is AAOI Overvalued or Undervalued? [...]"
)


class FakeAttachment:
    def __init__(self, document_type: str, text: str):
        self.document_type = document_type
        self._text = text

    def text(self) -> str:
        return self._text


class FakeReport:
    def __init__(self, items: dict[str, str]):
        self._items = items

    def __getitem__(self, key: str) -> str:
        return self._items[key]


class FakeFiling:
    accession_no = "0001683168-26-006862"

    def __init__(self, items: dict[str, str], attachments=(), *, broken: bool = False):
        self._report = FakeReport(items)
        self.attachments = list(attachments)
        self._broken = broken

    def obj(self):
        if self._broken:
            raise ValueError("unparseable")
        return self._report


class TestEightK:
    def test_the_houston_lease_says_what_it_is(self):
        filing = FakeFiling({"Item 1.01": AAOI_HOUSTON, "Item 2.03": AAOI_2_03})
        lead = eight_k_lead(filing, ["1.01", "2.03", "9.01"])
        assert lead.startswith("On August 31, 2026, Applied Optoelectronics")
        assert "two industrial buildings" in lead and "Houston, Texas" in lead

    def test_the_item_heading_is_not_the_lead(self):
        lead = eight_k_lead(FakeFiling({"Item 1.01": AAOI_HOUSTON}), ["1.01"])
        assert "Item 1.01" not in lead and "Material Definitive" not in lead

    def test_defined_term_declarations_are_removed(self):
        lead = eight_k_lead(FakeFiling({"Item 1.01": AAOI_HOUSTON}), ["1.01"])
        assert '(the "Company")' not in lead and '"Leases"' not in lead

    def test_an_item_incorporated_by_reference_is_skipped(self):
        """2.03 only points at 1.01; with 1.01 unreadable there is no lead to invent."""
        filing = FakeFiling({"Item 2.03": AAOI_2_03, "Item 9.01": AAOI_9_01})
        assert eight_k_lead(filing, ["2.03", "9.01"]) is None

    def test_a_boilerplate_item_yields_to_its_press_release(self):
        filing = FakeFiling(
            {"Item 2.02": CEREBRAS_2_02}, [FakeAttachment("EX-99.1", CEREBRAS_EX99)]
        )
        lead = eight_k_lead(filing, ["2.02", "9.01"])
        assert lead.startswith("Cerebras Systems Announces Strong First Quarter 2026 Results")
        assert "shall not be deemed" not in lead

    def test_an_exhibits_only_8k_reads_its_press_release(self):
        filing = FakeFiling({"Item 9.01": AAOI_9_01}, [FakeAttachment("EX-99.1", AMD_EX99)])
        assert eight_k_lead(filing, ["9.01"]) == "AMD Appoints Tim Ryan to Board of Directors"

    def test_an_unparseable_filing_has_no_lead(self):
        assert eight_k_lead(FakeFiling({}, broken=True), ["1.01"]) is None

    def test_no_items_and_no_exhibit_is_none(self):
        assert eight_k_lead(FakeFiling({}), []) is None

    def test_a_missing_item_section_falls_through(self):
        """edgartools lists an item whose section it then cannot find."""
        filing = FakeFiling({"Item 1.01": AAOI_HOUSTON})
        assert eight_k_lead(filing, ["5.02", "1.01"]).startswith("On August 31, 2026")


AAOI_ATM = (
    "Item 1.01 Entry into a Material Definitive Agreement. On August 21, 2026, Applied "
    "Optoelectronics, Inc. entered into an Equity Distribution Agreement with Raymond James "
    "& Associates, Inc. and Needham & Company, LLC pursuant to which the Company may issue "
    "and sell shares of the Company’s common stock, par value $0.001 per share having an "
    "aggregate offering price of up to $600 million from time to time through the Sales "
    "Agents. Upon delivery of a placement notice and subject to the terms and conditions of "
    "the Agreement, sales, if any, of the Shares will be made through the Sales Agents."
)


class TestTrim:
    def test_the_atm_lead_reaches_its_dollar_amount(self):
        """At 320 characters this stopped at "up to…", one word before the number."""
        lead = eight_k_lead(FakeFiling({"Item 1.01": AAOI_ATM}), ["1.01"])
        assert "up to $600 million" in lead
        assert "par value" not in lead

    def test_empty_and_none(self):
        assert trim(None) is None
        assert trim("   ") is None

    def test_only_boilerplate_is_none(self):
        assert trim("The information shall not be deemed filed for purposes of Section 18.") is None

    def test_an_overlong_sentence_is_cut_and_marked(self):
        lead = trim("word " * 200)
        assert len(lead) <= LEAD_CHARS
        assert lead.endswith("…")

    def test_exactly_the_budget_is_not_cut(self):
        sentence = "A" * (LEAD_CHARS - 1) + "."
        assert trim(sentence) == sentence

    def test_corporate_abbreviations_do_not_end_a_sentence(self):
        text = (
            "On June 2, 2026, Alphabet Inc. entered into an agreement with Goldman Sachs & "
            "Co. LLC, J.P. Morgan Securities LLC and Morgan Stanley & Co. LLC. The " + "y " * 200
        )
        assert trim(text).endswith("Morgan Stanley & Co. LLC.")

    def test_bullets_are_separated_and_stop_between_points(self):
        lead = exhibit_lead(CEREBRAS_EX99)
        assert " · GAAP quarterly revenue" in lead
        assert "•" not in lead

    def test_a_subheading_run_into_the_sentence_is_removed(self):
        text = (
            "Notes Offering On June 22, 2026, Space Exploration Technologies Corp. "
            "commenced an offering."
        )
        assert trim(text).startswith("On June 22, 2026")

    def test_a_parenthetical_after_the_full_stop_leaves_one_period(self):
        text = 'located at No.227 Kesheng Rd., Ningbo, China. ("Leased Property"). The lease runs.'
        assert trim(text) == "located at No.227 Kesheng Rd., Ningbo, China. The lease runs."

    def test_an_ellipsis_survives(self):
        assert trim("Revenue grew... and then fell.") == "Revenue grew... and then fell."

    def test_trim_is_idempotent(self):
        once = trim(AAOI_HOUSTON)
        assert trim(once) == once


class TestArticles:
    def test_page_furniture_is_dropped(self):
        lead = article_lead(GURUFOCUS)
        assert lead.startswith("On September 01, 2026, Applied Optoelectronics")
        assert "Logo" not in lead and "Avatar" not in lead and "#" not in lead

    def test_an_abstract_passes_through(self):
        abstract = (
            "Applied Optoelectronics faces supply constraints despite surging 800G demand, "
            "while FN's broad AI data-center growth and capacity expansion strengthen its position."
        )
        assert article_lead(abstract) == abstract

    def test_benzinga_starts_below_its_headline(self):
        """Ticker tape and the hero-image caption came back as the lead."""
        page = (
            "84363.992.1277%DIA\n\n514.120.03%GLD\n\n"
            "Close-up of the SpaceX logo on a white surface, with a blurred, colorful stock "
            "chart background.\n\nSeptember 23, 2026 4:31 PM 2 min read\n\n"
            "# SpaceX Stock Slides Wednesday: What Happened?\n\n"
            "Space Exploration Technologies Corp. (NASDAQ:SPCX) shares fell Wednesday as a "
            "proposed executive insider sale and an imminent post-IPO share unlock create "
            "near-term supply overhang concerns.\n\nHere’s what investors need to know. [...]"
        )
        lead = article_lead(page)
        assert lead.startswith("Space Exploration Technologies Corp. (NASDAQ:SPCX) shares fell")
        assert "logo" not in lead

    def test_a_caption_without_a_headline_is_still_dropped(self):
        page = (
            "The SpaceX logo, white text on a dark background, is displayed on a screen.\n\n"
            "SpaceX shares are remaining stable Monday as its Nasdaq-100 weighting doubles."
        )
        assert article_lead(page).startswith("SpaceX shares are remaining stable")

    def test_a_quote_widget_is_not_a_lead(self):
        page = (
            "Space Exploration Technologies Stock Quote\n\n## NASDAQ: SPCX\n\nToday's Change"
            "\n\n(-1.36%) $-2.10\n\nMarket Cap\n\n$2.1TMarket cap calculated using publicly "
            "traded shares outstanding only. Does not include unlisted, private, or dual-class "
            "non-traded shares.\n\n52wk Range"
        )
        assert article_lead(page) is None

    def test_prose_before_a_widget_survives(self):
        page = (
            "SpaceX went public at a price that valued it near $1.8 trillion -- about 94 times "
            "sales at the first quarter's annualized pace.\n\nSpace Exploration Technologies "
            "Stock Quote\n\n## NASDAQ: SPCX"
        )
        assert article_lead(page).startswith("SpaceX went public at a price")

    def test_a_headline_with_nothing_below_falls_back_to_the_page(self):
        page = "An opening paragraph long enough to count as prose on its own.\n# Headline"
        assert article_lead(page).startswith("An opening paragraph")

    def test_nothing_but_furniture_is_none(self):
        assert article_lead("Logo\nLogo\n# Headline") is None


def _item(doc_type: str, summary: str | None, tier=SourceTier.PRIMARY) -> SourceItem:
    return SourceItem(
        tier=tier,
        provider="SEC EDGAR",
        url="https://www.sec.gov/x.htm",
        title="x",
        published_at=datetime(2026, 9, 1, tzinfo=timezone.utc),
        entity=EntityMatch(symbol="AAOI", cik=1158114, method=MatchMethod.CIK),
        doc_type=doc_type,
        summary=summary,
    )


class TestLeadFor:
    def test_a_delisting_class_is_evidence_not_a_lead(self):
        assert lead_for(_item("25-NSE", "Warrants, each whole warrant exercisable")) is None

    def test_a_6k_reads_its_exhibit(self):
        lead = lead_for(_item("6-K", "Exhibit 99.1 Nebius completes acquisition of Eigen AI."))
        assert lead == "Nebius completes acquisition of Eigen AI."

    def test_a_filing_without_summary(self):
        assert lead_for(_item("8-K", None)) is None

    @pytest.mark.parametrize("form", ["424B5", "10-Q", "144"])
    def test_forms_without_a_lead_reader_return_none(self, form):
        assert lead_for(_item(form, "anything at all.")) is None
