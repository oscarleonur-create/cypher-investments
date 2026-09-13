"""Rendering. This text is what Telegram will send in phase 7, so it has to
carry its own caveats — there is no surrounding UI to add them later."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest
from advisor.daemon.market_calendar import MARKET_TZ
from advisor.daemon.models import EventTier
from advisor.story.models import (
    Anchor,
    Attribution,
    Confidence,
    Corroboration,
    PositionAtEvent,
    PriceReaction,
    Story,
    ThesisLink,
    Verdict,
)
from advisor.story.render import headline, render

FILED = datetime(2026, 8, 21, 16, 9, tzinfo=MARKET_TZ)


def story(**kw) -> Story:
    base = dict(
        symbol="AAOI",
        assembled_at=FILED + timedelta(days=16),
        anchor=Anchor(
            event_id="e1",
            kind="FILING_DILUTION",
            tier=EventTier.A,
            symbol="AAOI",
            occurred_at=FILED,
            ingested_at=FILED + timedelta(days=16),
            headline="prospectus supplement — shares offered",
            source="SEC EDGAR",
            url="https://www.sec.gov/x.htm",
            quote="aggregate offering price of up to $600,000,000",
            facts={
                "form": "424B5",
                "offering_usd": 6e8,
                "dilution_pct": 0.067,
                "market_cap": 8.96e9,
            },
        ),
        position=PositionAtEvent(
            confidence=Confidence.MEASURED,
            quantity=12,
            avg_open_price=129.31,
            weight_of_net_liq=0.1632,
            net_liq=7761.13,
            snapshot_asof=FILED - timedelta(hours=1),
            covers_event=True,
        ),
        reaction=PriceReaction(
            confidence=Confidence.MEASURED,
            session=date(2026, 8, 24),
            before=124.82,
            after=107.63,
            pct_move=-0.1377,
            dollars=-206.28,
            pct_of_book=-0.0266,
            priced_next_session=True,
        ),
        attribution=Attribution(
            confidence=Confidence.ESTIMATED,
            verdict=Verdict.PARTLY_SPECIFIC,
            actual_return=-0.1482,
            expected_return=-0.0635,
            residual_z=-1.12,
            resid_vol=0.0753,
            r2=0.2737,
        ),
        corroboration=Corroboration(window_days=4, items=[], primary_count=0),
        thesis=ThesisLink(confidence=Confidence.MEASURED, exists=False, note="no stated thesis"),
    )
    return Story(**{**base, **kw})


class TestHeadline:
    def test_a_sized_offering_leads_with_the_number_that_matters(self):
        assert headline(story()) == "AAOI — $600,000,000 offering, 6.7% dilution"

    def test_an_unsized_event_falls_back_to_the_move(self):
        s = story()
        s.anchor.facts = {}
        assert "-13.77%" in headline(s)

    def test_with_neither_it_still_produces_a_line(self):
        s = story()
        s.anchor.facts = {}
        s.reaction = PriceReaction(confidence=Confidence.UNAVAILABLE)
        assert s.symbol in headline(s)


class TestCaveatsSurvive:
    """Every warning must be in the text itself, not the surrounding page."""

    def test_a_position_that_postdates_the_event_is_flagged_in_the_output(self):
        s = story()
        s.position.covers_event = False
        s.position.note = "no snapshot from the event date; showing 2026-09-06"
        out = render(s)
        assert "postdates the event" in out
        assert "no snapshot from the event date" in out

    def test_a_sub_threshold_residual_says_so_explicitly(self):
        """Without this line the verdict reads stronger than the z supports."""
        out = render(story())
        assert "below the 2.0 alert threshold" in out

    def test_a_firing_residual_does_not_carry_the_sub_threshold_disclaimer(self):
        s = story()
        s.attribution.verdict = Verdict.UNEXPLAINED
        s.attribution.residual_z = -4.9
        out = render(s)
        assert "below the 2.0 alert threshold" not in out
        assert "macro cannot explain" in out

    def test_the_residual_is_always_shown_with_the_vol_it_is_scaled_by(self):
        out = render(story())
        assert "7.53%/day" in out

    def test_expected_and_actual_are_both_printed(self):
        out = render(story())
        assert "-6.35%" in out and "-14.82%" in out

    def test_the_next_session_note_appears_when_the_event_missed_the_close(self):
        assert "landed after the close" in render(story())

    def test_a_backfilled_event_is_marked(self):
        assert "learned later" in render(story())


class TestUnavailableSlots:
    def test_an_unfilled_position_prints_its_reason(self):
        s = story(
            position=PositionAtEvent(
                confidence=Confidence.UNAVAILABLE, note="the daemon was not recording"
            )
        )
        out = render(s)
        assert "the daemon was not recording" in out

    def test_unfilled_slots_are_listed_at_the_end(self):
        s = story(
            reaction=PriceReaction(confidence=Confidence.UNAVAILABLE, note="no price history"),
            attribution=Attribution(confidence=Confidence.UNAVAILABLE, note="no estimate"),
        )
        out = render(s)
        assert "NOT ESTABLISHED" in out
        assert "reaction" in out and "attribution" in out

    def test_a_not_held_position_reads_differently_from_an_unknown_one(self):
        held_none = render(
            story(
                position=PositionAtEvent(
                    confidence=Confidence.MEASURED, quantity=0.0, covers_event=True
                )
            )
        )
        unknown = render(
            story(
                position=PositionAtEvent(
                    confidence=Confidence.UNAVAILABLE, note="no snapshot exists"
                )
            )
        )
        assert "not held" in held_none
        assert "unknown" in unknown

    def test_a_story_with_every_slot_empty_still_renders(self):
        s = story(
            position=PositionAtEvent(confidence=Confidence.UNAVAILABLE, note="x"),
            reaction=PriceReaction(confidence=Confidence.UNAVAILABLE, note="y"),
            attribution=Attribution(confidence=Confidence.UNAVAILABLE, note="z"),
        )
        assert "AAOI" in render(s)


class TestEvidence:
    def test_the_quote_and_url_both_appear(self):
        out = render(story())
        assert "600,000,000" in out
        assert "https://www.sec.gov/x.htm" in out

    def test_a_story_without_a_quote_omits_the_evidence_block(self):
        s = story()
        s.anchor.quote = None
        assert "EVIDENCE" not in render(s)

    @pytest.mark.parametrize("verdict", list(Verdict))
    def test_every_verdict_has_renderable_text(self, verdict):
        s = story()
        s.attribution.verdict = verdict
        assert render(s)
