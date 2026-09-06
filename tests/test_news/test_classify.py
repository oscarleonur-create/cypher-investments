"""Filing classification, straight from the SEC's own taxonomy."""

from __future__ import annotations

import pytest
from advisor.news.classify import (
    FORM_MAP,
    ITEM_MAP,
    FilingKind,
    Materiality,
    classify_filing,
    classify_items,
)


class TestFormClassification:
    def test_a_424b5_is_dilution_and_high(self):
        """The filing that cost 13.8% in a session."""
        result = classify_filing("424B5")
        assert result.kind is FilingKind.DILUTION
        assert result.materiality is Materiality.HIGH

    def test_a_shelf_is_capacity_not_issuance(self):
        """S-3 registers the right to sell later; it is not a sale."""
        assert classify_filing("S-3").kind is FilingKind.SHELF
        assert classify_filing("S-3").materiality is Materiality.MEDIUM
        assert classify_filing("424B5").materiality is Materiality.HIGH

    def test_late_filing_is_high_materiality(self):
        """A company that cannot file on time is telling you something."""
        assert classify_filing("NT 10-Q").kind is FilingKind.LATE_FILING
        assert classify_filing("NT 10-Q").materiality is Materiality.HIGH

    def test_13d_and_13g_are_not_the_same_event(self):
        assert classify_filing("SC 13D").kind is FilingKind.ACTIVIST_STAKE
        assert classify_filing("SC 13D").materiality is Materiality.HIGH
        assert classify_filing("SC 13G").kind is FilingKind.PASSIVE_STAKE
        assert classify_filing("SC 13G").materiality is Materiality.LOW

    def test_form_lookup_is_case_and_space_insensitive(self):
        assert classify_filing(" 424b5 ").kind is FilingKind.DILUTION

    def test_an_unknown_form_is_low_not_an_error(self):
        result = classify_filing("SOMETHING-NEW")
        assert result.kind is FilingKind.OTHER
        assert result.materiality is Materiality.LOW


class TestItemClassification:
    def test_an_8k_is_classified_by_its_items_not_by_being_an_8k(self):
        """'8-K filed' is not information; '8-K reporting results' is."""
        assert classify_filing("8-K", ["2.02"]).kind is FilingKind.RESULTS

    def test_the_most_material_item_wins(self):
        """A restatement buried beside two housekeeping items still surfaces."""
        result = classify_filing("8-K", ["9.01", "4.02", "7.01"])
        assert result.kind is FilingKind.RESTATEMENT
        assert result.materiality is Materiality.HIGH

    def test_the_real_aaoi_8k_items_classify_as_material_agreement(self):
        result = classify_filing("8-K", ["1.01", "5.02", "9.01"])
        assert result.materiality is Materiality.MEDIUM

    def test_items_may_carry_the_word_item(self):
        assert classify_items(["Item 2.02"])[0].kind is FilingKind.RESULTS

    def test_unmapped_items_are_dropped_not_guessed(self):
        assert classify_items(["6.99"]) == []

    def test_an_8k_with_no_recognised_items_falls_back_to_unknown(self):
        result = classify_filing("8-K", ["6.99"])
        assert result.kind is FilingKind.OTHER
        assert result.materiality is Materiality.LOW

    def test_an_8k_with_no_items_at_all_does_not_raise(self):
        assert classify_filing("8-K", []).materiality is Materiality.LOW
        assert classify_filing("8-K", None).materiality is Materiality.LOW

    def test_a_non_8k_form_ignores_stray_item_codes(self):
        """Items belong to 8-Ks; a 424B5 is classified by its form."""
        assert classify_filing("424B5", ["9.01"]).kind is FilingKind.DILUTION


class TestTaxonomyIntegrity:
    def test_every_mapping_has_a_label(self):
        for mapping in (ITEM_MAP, FORM_MAP):
            for classification in mapping.values():
                assert classification.label.strip()

    def test_high_materiality_items_are_genuinely_severe(self):
        severe = {k for k, v in ITEM_MAP.items() if v.materiality is Materiality.HIGH}
        assert {"1.03", "2.02", "3.01", "4.01", "4.02"} <= severe

    @pytest.mark.parametrize("form", ["10-K", "10-Q"])
    def test_periodic_reports_do_not_interrupt(self, form):
        assert classify_filing(form).materiality is not Materiality.HIGH

    def test_insider_transactions_are_low_by_default(self):
        """A single Form 4 is noise; a pattern is signal, and that is Phase 4."""
        assert classify_filing("4").materiality is Materiality.LOW
