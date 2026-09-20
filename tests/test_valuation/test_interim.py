"""Interim figures out of 6-K prose, and what must never come out of it.

The live case: Nebius files a 20-F once a year, so `latest_fundamentals` valued
the position on figures 263 days old — 108x revenue against a real 25x. The Q2
2026 numbers were inside an EX-99.1 exhibit carrying no XBRL, as MD&A prose
with tables flattened into it:

    Total other 24.6 81.5 32.9 101.4 income, net

Every test here is about the same property: a figure is returned only when the
document states it twice, and a sentence this module misreads must produce
nothing rather than a confident wrong number.
"""

from __future__ import annotations

from advisor.valuation.interim import (
    Basis,
    Period,
    _parse_attachments,
    headline_figures,
    parse_interim,
)

# The real sentence from nbis-20260812xex99d1.htm.
NBIS_TOTAL = (
    "Revenues by operating segment: Total revenues increased by $477.2 million, "
    "or 454%, from $105.1 million in the second quarter of 2025 to $582.3 million "
    "in the second quarter of 2026."
)

NBIS_SEGMENT_TABLE = (
    "The table below presents information about the Adjusted EBITDA / (loss) of the "
    "operating segments: Three months Six months ended ended June 30, June 30, 2025 2026 "
    "2025 2026 -in millions of -in millions of U.S. dollars U.S. dollars Nebius AI cloud "
    "9.5 285.7 -17.9 459.7 Avride -17.3 -40.1 -34.2 -74.2 TripleTen -13.2 -9.4 -22.6 "
    "-19.8 Total adjusted -21.0 236.2 -74.7 365.7 EBITDA / (loss) Adjusted EBITDA / "
    "(loss) by operating segments: Total Adjusted EBITDA loss for the Group improved by "
    "$257.2 million in the second quarter of 2026 compared to the same period of 2025. "
    "Total Adjusted EBITDA / (loss) for the Group improved by $440.4 million in the six "
    "months ended June 30, 2026, compared to the same period of 2025. Adjusted EBITDA for "
    "the Nebius AI cloud business improved by $276.2 million and $477.6 million in the "
    "three months of 2026 and six months ended June 30, 2026, respectively, compared to "
    "the same periods of 2025."
)


# --- the self-checking sentence ---------------------------------------------


def test_the_stated_comparison_is_read_whole():
    fig = parse_interim(NBIS_TOTAL).find(metric="total revenues")
    assert fig is not None
    assert fig.prior == 105_100_000
    assert fig.current == 582_300_000
    assert fig.delta == 477_200_000
    assert round(fig.growth * 100, 1) == 454.0
    assert fig.basis is Basis.SELF_CHECKED
    assert fig.period is Period.QUARTER


def test_a_sentence_whose_levels_contradict_its_delta_is_refused():
    """The arithmetic is the whole safety property: if we matched the wrong
    clauses together, the numbers stop agreeing."""
    broken = NBIS_TOTAL.replace(
        "$582.3 million in the second quarter of 2026",
        "$999.9 million in the second quarter of 2026",
    )
    result = parse_interim(broken)
    assert result.find(metric="total revenues") is None
    assert any("stated change" in r for r in result.rejected)


def test_a_sentence_whose_percentage_contradicts_its_delta_is_refused():
    broken = NBIS_TOTAL.replace("or 454%", "or 12%")
    result = parse_interim(broken)
    assert result.find(metric="total revenues") is None
    assert any("computed" in r for r in result.rejected)


def test_a_decrease_carries_its_sign():
    text = (
        "Revenues from TripleTen decreased by $2.3 million, or 19%, from "
        "$12.3 million in the second quarter of 2025 to $10.0 million in the "
        "second quarter of 2026."
    )
    fig = parse_interim(text).find(metric="TripleTen")
    assert fig is not None
    assert fig.delta == -2_300_000
    assert fig.growth is not None and fig.growth < 0


def test_the_six_month_column_is_a_different_period():
    text = (
        "Total revenues increased by $825.3 million, or 529%, from $156.0 million "
        "in the six months ended June 30, 2025 to $981.3 million in the same "
        "period in 2026."
    )
    result = parse_interim(text)
    assert result.find(metric="total revenues", period=Period.QUARTER) is None
    half = result.find(metric="total revenues", period=Period.HALF)
    assert half is not None and half.current == 981_300_000


def test_rounding_at_the_tolerance_boundary_still_reconciles():
    """Figures round to a tenth, so two roundings can disagree by 0.2 and the
    sentence is still the one the issuer wrote."""
    text = (
        "Total revenues increased by $477.0 million, or 454%, from $105.1 million "
        "in the second quarter of 2025 to $582.3 million in the second quarter of 2026."
    )
    assert parse_interim(text).find(metric="total revenues") is not None


# --- the table a sentence has to confirm ------------------------------------


def test_a_segment_row_is_taken_only_with_the_sentence_that_confirms_it():
    result = parse_interim(NBIS_SEGMENT_TABLE)
    fig = result.find(metric="Nebius AI cloud")
    assert fig is not None
    assert (fig.prior, fig.current) == (9_500_000, 285_700_000)
    assert fig.basis is Basis.TABLE_CONFIRMED


def test_a_confirmed_row_is_named_by_the_sentence_not_the_row():
    """ "Nebius AI cloud" heads both a revenue table and an EBITDA table, and
    the flattener truncates the total's label to "Total adjusted". The
    sentence is the thing that says what was measured."""
    fig = parse_interim(NBIS_SEGMENT_TABLE).find(metric="Nebius AI cloud")
    assert fig is not None
    assert "Adjusted EBITDA" in fig.metric


def test_a_row_no_sentence_mentions_is_dropped():
    """Avride and TripleTen are in the same table with no confirming sentence."""
    result = parse_interim(NBIS_SEGMENT_TABLE)
    assert result.find(metric="Avride") is None
    assert any(r.startswith("Avride") for r in result.rejected)


def test_a_row_whose_change_disagrees_with_the_sentence_is_dropped():
    broken = NBIS_SEGMENT_TABLE.replace("9.5 285.7", "9.5 700.0")
    assert parse_interim(broken).find(metric="Nebius AI cloud") is None


def test_the_unit_header_is_not_absorbed_into_a_row_label():
    """The label read back as "ollars U.S. dollars Nebius AI cloud", matched no
    sentence, and dropped the one segment figure the thesis needed."""
    fig = parse_interim(NBIS_SEGMENT_TABLE).find(metric="Nebius AI cloud")
    assert fig is not None
    assert "dollars" not in fig.metric.lower()


# --- numbers that must not be invented --------------------------------------


def test_no_growth_rate_off_a_negative_base():
    """Total adjusted EBITDA went -21.0 to 236.2. Dividing by the base reads as
    +1,225% growth and describes a sign flip."""
    fig = parse_interim(NBIS_SEGMENT_TABLE).find(metric="Total Adjusted EBITDA")
    assert fig is not None
    assert fig.prior < 0 < fig.current
    assert fig.growth is None


def test_no_growth_rate_off_a_zero_base():
    text = (
        "Segment revenues increased by $5.0 million, or 100%, from $0.0 million "
        "in the second quarter of 2025 to $5.0 million in the second quarter of 2026."
    )
    for fig in parse_interim(text).figures:
        assert fig.growth is None


def test_empty_input_yields_nothing_rather_than_failing():
    result = parse_interim("")
    assert result.figures == [] and result.rejected == []


def test_prose_with_no_figures_yields_nothing():
    text = (
        "You should read the following discussion and analysis of our financial "
        "condition in conjunction with our unaudited condensed consolidated "
        "financial statements. Our actual results may differ materially."
    )
    assert parse_interim(text).figures == []


def test_a_scale_stated_once_governs_the_clause():
    text = (
        "Total revenues increased by $2.0 billion, or 100%, from $2.0 in the "
        "second quarter of 2025 to $4.0 in the second quarter of 2026."
    )
    fig = parse_interim(text).find(metric="total revenues")
    assert fig is not None and fig.current == 4_000_000_000


# --- what gets promoted to a payload ----------------------------------------


def test_headline_figures_promote_only_names_every_filer_shares():
    fields = headline_figures(parse_interim(NBIS_TOTAL + " " + NBIS_SEGMENT_TABLE))
    assert fields["revenue_usd"] == 582_300_000
    assert fields["revenue_growth_yoy"] == 4.5404
    assert fields["revenue_period"] == "QUARTER"
    assert fields["adjusted_ebitda_usd"] == 236_200_000
    # The segment line is parsed and readable, never a payload key a thesis
    # could come to depend on — its name is whatever the issuer calls itself.
    assert not any("nebius" in k.lower() for k in fields)


def test_headline_figures_of_nothing_is_an_empty_mapping():
    assert headline_figures(parse_interim("")) == {}


def test_a_missing_revenue_line_promotes_no_revenue_keys():
    fields = headline_figures(parse_interim(NBIS_SEGMENT_TABLE))
    assert "revenue_usd" not in fields
    assert fields["adjusted_ebitda_usd"] == 236_200_000


# --- choosing which filing to read ------------------------------------------


class _Attachment:
    def __init__(self, document_type: str, text: str | Exception):
        self.document_type = document_type
        self._text = text

    def text(self) -> str:
        if isinstance(self._text, Exception):
            raise self._text
        return self._text


class _Filing:
    def __init__(self, *attachments: _Attachment):
        self.attachments = list(attachments)


def test_the_exhibit_with_the_most_confirmed_figures_wins():
    """A quarter arrives as a shareholder letter, a slide deck and an
    operating review. Only one of them reconciles."""
    filing = _Filing(
        _Attachment("EX-99.2", "Our quarter was strong and our outlook is bright."),
        _Attachment("EX-99.1", NBIS_TOTAL + " " + NBIS_SEGMENT_TABLE),
    )
    result = _parse_attachments(filing)
    assert result is not None and len(result.figures) > 1


def test_an_exhibit_that_will_not_render_does_not_sink_the_filing():
    filing = _Filing(
        _Attachment("EX-99.1", RuntimeError("binary attachment")),
        _Attachment("EX-99.2", NBIS_TOTAL),
    )
    result = _parse_attachments(filing)
    assert result is not None
    assert result.find(metric="total revenues") is not None


def test_non_results_exhibits_are_not_read():
    """An indenture and a graphic are attached to the same kind of filing."""
    filing = _Filing(_Attachment("EX-4.1", NBIS_TOTAL), _Attachment("GRAPHIC", NBIS_TOTAL))
    assert _parse_attachments(filing) is None


def test_a_filing_with_no_attachments_yields_none():
    assert _parse_attachments(_Filing()) is None


def test_a_press_release_register_confirms_nothing():
    """The real sentence from the NBIS press release. It states a level and a
    rate with nothing to check them against, so it must yield no figure —
    this is why the sibling filing is read instead of loosening the gate."""
    text = (
        "Group revenue of $582.3 million, was up 454% year-over-year, and up 46% "
        "compared to Q1. Nebius AI cloud revenue grew 514% year-over-year to "
        "$575 million."
    )
    assert parse_interim(text).figures == []
