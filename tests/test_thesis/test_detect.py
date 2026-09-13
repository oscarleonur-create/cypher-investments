"""Telling a written thesis from an untouched template.

The live evidence this exists for: CRDO's thesis is 1,306 characters, byte for
byte the length of the blank template, because it is the blank template. Phase
3 reported it as a thesis and the story told the user they held a view they
had never written down.
"""

from __future__ import annotations

import pathlib

import pytest
from advisor.thesis.detect import (
    MIN_ORIGINAL_CHARS,
    completeness,
    is_substantive,
    original_prose,
    template_lines_remaining,
)
from advisor.thesis.template import TEMPLATE

REAL = (
    "AAOI makes optical transceivers and, unusually, fabricates its own lasers. "
    "The AI datacentre build-out is pulling 800G demand forward faster than "
    "transceiver capacity can be added, and in-house fabrication is the "
    "constraint competitors cannot buy their way around inside this cycle."
)


class TestTheUntouchedTemplate:
    def test_the_blank_template_is_not_substantive(self):
        assert is_substantive(TEMPLATE) is False

    def test_it_yields_no_original_prose_at_all(self):
        assert original_prose(TEMPLATE) == ""

    def test_completeness_of_the_blank_template_is_zero(self):
        assert completeness(TEMPLATE) == 0.0

    def test_wrapped_template_paragraphs_do_not_leak_through(self):
        """A first attempt matched only first lines and passed CRDO as written."""
        for fragment in (
            "competitive advantage, switching costs, network effects, pricing power.",
            "assumptions (growth, margins, multiple), and the implied upside/downside.",
            "approvals, capital returns). What and when.",
        ):
            assert fragment not in original_prose(TEMPLATE)

    def test_a_symbol_heading_does_not_count_as_content(self):
        headed = TEMPLATE.replace("# <one-line thesis>", "# AAOI — <one-line thesis>")
        assert is_substantive(headed) is False


class TestWrittenTheses:
    def test_real_prose_added_to_the_template_is_substantive(self):
        assert is_substantive(TEMPLATE + "\n\n" + REAL) is True

    def test_the_original_prose_is_what_comes_back(self):
        prose = original_prose(TEMPLATE + "\n\n" + REAL)
        assert "fabricates its own lasers" in prose
        assert "switching costs" not in prose

    def test_completeness_rises_as_template_lines_are_replaced(self):
        half = "\n".join(TEMPLATE.splitlines()[:18])
        assert completeness(half) > completeness(TEMPLATE)

    def test_a_thesis_written_from_scratch_needs_no_template(self):
        assert is_substantive(REAL) is True


class TestBoundaries:
    def test_just_under_the_floor_is_not_substantive(self):
        assert is_substantive("x" * (MIN_ORIGINAL_CHARS - 1)) is False

    def test_exactly_at_the_floor_is_substantive(self):
        assert is_substantive("x" * MIN_ORIGINAL_CHARS) is True

    @pytest.mark.parametrize("content", ["", None, "   ", "\n\n\n"])
    def test_empty_inputs_do_not_raise(self, content):
        assert is_substantive(content) is False
        assert original_prose(content) == ""
        assert template_lines_remaining(content) == 0

    def test_headings_and_checkboxes_alone_are_not_content(self):
        assert is_substantive("# Title\n## Section\n- [ ] a box\n- [x] another") is False

    def test_whitespace_differences_still_match_the_template(self):
        """Re-indented or re-wrapped template text is still template text."""
        spaced = "\n".join("   " + line for line in TEMPLATE.splitlines())
        assert is_substantive(spaced) is False


class TestTemplateMirror:
    def test_the_python_copy_matches_the_frontend_source(self):
        """The editor inserts the TS copy; detection reads the Python one."""
        ts = pathlib.Path("frontend/src/lib/thesisTemplate.ts").read_text()
        body = ts.split("return `", 1)[1].rsplit("`;", 1)[0].replace("${heading}", "# ")
        assert body.strip() == TEMPLATE.strip(), (
            "thesisTemplate.ts and advisor/thesis/template.py have drifted — "
            "detection will stop recognising the blank form"
        )
