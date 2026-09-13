"""Slot assembly, and the two honesty rules that govern it.

Both rules exist because a prototype broke them within minutes of being
written, and CLAUDE.md now carries them as inherited constraints:

- a position from the wrong date must announce itself, never substitute
- a verdict must not claim more than its residual supports
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from advisor.daemon.book import EQUITY, BookSnapshot, Position
from advisor.daemon.market_calendar import MARKET_TZ
from advisor.daemon.models import Event, EventSource, EventTier
from advisor.daemon.store import DaemonStore
from advisor.macro.factors import Factor
from advisor.macro.sensitivity import FactorLoading, SymbolSensitivity
from advisor.news.models import EntityMatch, MatchMethod, SourceItem, SourceTier
from advisor.story.assemble import build_story, verdict_for
from advisor.story.models import Confidence, Verdict

FILED = datetime(2026, 8, 21, 16, 9, tzinfo=MARKET_TZ)


@pytest.fixture
def store(tmp_path: Path):
    s = DaemonStore(tmp_path / "research.db")
    yield s
    s.close()


def pos(symbol="AAOI", *, qty=12, entry=129.31, price=124.82, account="A") -> Position:
    return Position(
        account=account,
        symbol=symbol,
        underlying=symbol,
        instrument=EQUITY,
        quantity=qty,
        multiplier=1,
        avg_open_price=entry,
        close_price=price,
    )


def book(*positions, as_of: datetime, net_liq=7761.13) -> BookSnapshot:
    return BookSnapshot(as_of=as_of, positions=list(positions), net_liq=net_liq)


def dilution_event(**kw) -> Event:
    base = dict(
        source=EventSource.EDGAR,
        kind="FILING_DILUTION",
        tier=EventTier.A,
        symbol="AAOI",
        dedup_key="0001104659-26-099685",
        payload={
            "form": "424B5",
            "accepted_at": FILED.isoformat(),
            "offering_usd": 600_000_000.0,
            "dilution_pct": 0.067,
            "market_cap": 8_960_160_678.0,
            "quote": "aggregate offering price of up to $600,000,000",
            "url": "https://www.sec.gov/x.htm",
            "provider": "SEC EDGAR",
        },
    )
    return Event(**{**base, **kw})


def price_frame(symbol="AAOI") -> pd.DataFrame:
    idx = pd.to_datetime(["2026-08-20", "2026-08-21", "2026-08-24", "2026-08-25"])
    return pd.DataFrame({symbol: [129.10, 124.82, 107.63, 113.15]}, index=idx)


def factor_frame(rng=None) -> pd.DataFrame:
    rng = rng or np.random.default_rng(3)
    idx = pd.to_datetime(["2026-08-20", "2026-08-21", "2026-08-24", "2026-08-25"])
    return pd.DataFrame({f.value: rng.normal(0, 0.01, len(idx)) for f in Factor}, index=idx)


def sensitivity(symbol="AAOI", *, resid_vol=0.0753, r2=0.2737) -> SymbolSensitivity:
    return SymbolSensitivity(
        symbol=symbol,
        asof=date(2026, 9, 4),
        window_days=250,
        n_obs=250,
        r2=r2,
        resid_vol=resid_vol,
        loadings=[FactorLoading(factor=Factor.MKT.value, loading=3.14, tstat=2.8)],
    )


class TestVerdictLadder:
    """Language must never outrun the statistic."""

    @pytest.mark.parametrize(
        "z,expected",
        [
            (-4.9, Verdict.NOT_MARKET),
            (3.0, Verdict.NOT_MARKET),
            (2.99, Verdict.UNEXPLAINED),
            (-2.0, Verdict.UNEXPLAINED),
            (1.99, Verdict.PARTLY_SPECIFIC),
            (-1.12, Verdict.PARTLY_SPECIFIC),  # the real AAOI session
            (0.99, Verdict.CONSISTENT),
            (0.0, Verdict.CONSISTENT),
        ],
    )
    def test_thresholds(self, z, expected):
        assert verdict_for(z) == expected

    def test_the_ladder_is_symmetric_in_sign(self):
        for magnitude in (0.5, 1.5, 2.5, 3.5):
            assert verdict_for(magnitude) == verdict_for(-magnitude)

    def test_no_estimate_is_unknown_not_consistent(self):
        """'No model' must never read as 'macro explains it'."""
        assert verdict_for(None) is Verdict.UNKNOWN

    def test_the_aaoi_session_does_not_claim_to_be_idiosyncratic(self):
        """z = -1.12 on a -14.8% day. The prototype called this idiosyncratic."""
        assert verdict_for(-1.12) is not Verdict.UNEXPLAINED
        assert verdict_for(-1.12) is not Verdict.NOT_MARKET


class TestPositionAtEvent:
    def test_a_snapshot_from_before_the_event_is_used_and_marked_as_covering(self, store):
        store.save_book(book(pos(), as_of=FILED - timedelta(hours=2)))
        story = build_story(store, dilution_event(), prices=price_frame(), factors=factor_frame())
        assert story.position.covers_event is True
        assert story.position.confidence is Confidence.MEASURED
        assert story.position.quantity == 12

    def test_a_later_snapshot_is_flagged_rather_than_substituted_silently(self, store):
        """The exact trap: only today's book exists, the event is weeks old."""
        store.save_book(book(pos(), as_of=FILED + timedelta(days=16)))
        story = build_story(store, dilution_event(), prices=price_frame(), factors=factor_frame())
        assert story.position.covers_event is False
        assert story.position.confidence is Confidence.ESTIMATED
        assert "no snapshot from the event date" in story.position.note

    def test_the_note_says_when_records_begin(self, store):
        store.save_book(book(pos(), as_of=FILED + timedelta(days=16)))
        story = build_story(store, dilution_event(), prices=price_frame(), factors=factor_frame())
        assert "records begin" in story.position.note

    def test_no_snapshots_at_all_is_unavailable_not_zero(self, store):
        """'The daemon was not recording' is not 'you held nothing'."""
        story = build_story(store, dilution_event(), prices=price_frame(), factors=factor_frame())
        assert story.position.confidence is Confidence.UNAVAILABLE
        assert story.position.quantity is None
        assert "not recording" in story.position.note

    def test_a_symbol_absent_from_the_snapshot_reads_as_not_held(self, store):
        store.save_book(book(pos("CRDO"), as_of=FILED - timedelta(hours=1)))
        story = build_story(store, dilution_event(), prices=price_frame(), factors=factor_frame())
        assert story.position.quantity == 0.0
        assert story.position.held is False

    def test_holdings_across_two_accounts_are_summed(self, store):
        store.save_book(
            book(
                pos(qty=8, account="5WI30382"),
                pos(qty=4, account="5WI47366"),
                as_of=FILED - timedelta(hours=1),
            )
        )
        story = build_story(store, dilution_event(), prices=price_frame(), factors=factor_frame())
        assert story.position.quantity == 12

    def test_the_most_recent_snapshot_at_or_before_the_event_wins(self, store):
        store.save_book(book(pos(qty=5), as_of=FILED - timedelta(days=3)))
        store.save_book(book(pos(qty=12), as_of=FILED - timedelta(minutes=5)))
        store.save_book(book(pos(qty=99), as_of=FILED + timedelta(days=1)))
        story = build_story(store, dilution_event(), prices=price_frame(), factors=factor_frame())
        assert story.position.quantity == 12


class TestReaction:
    def test_the_move_is_measured_on_the_session_that_priced_it(self, store):
        store.save_book(book(pos(), as_of=FILED - timedelta(hours=1)))
        story = build_story(store, dilution_event(), prices=price_frame(), factors=factor_frame())
        assert story.reaction.session == date(2026, 8, 24)
        assert story.reaction.priced_next_session is True
        assert story.reaction.before == pytest.approx(124.82)
        assert story.reaction.after == pytest.approx(107.63)
        assert story.reaction.pct_move == pytest.approx(-0.1377, abs=0.0002)

    def test_dollars_are_computed_on_the_position_actually_held(self, store):
        store.save_book(book(pos(qty=12), as_of=FILED - timedelta(hours=1)))
        story = build_story(store, dilution_event(), prices=price_frame(), factors=factor_frame())
        assert story.reaction.dollars == pytest.approx(12 * (107.63 - 124.82))
        assert story.reaction.pct_of_book == pytest.approx(-0.0266, abs=0.0005)

    def test_no_dollars_when_the_position_is_unknown(self, store):
        story = build_story(store, dilution_event(), prices=price_frame(), factors=factor_frame())
        assert story.reaction.dollars is None

    def test_a_missing_price_series_is_unavailable_not_zero(self, store):
        story = build_story(store, dilution_event(), prices=pd.DataFrame(), factors=factor_frame())
        assert story.reaction.confidence is Confidence.UNAVAILABLE

    def test_a_session_with_no_bar_yet_says_so(self, store):
        """An event filed tonight has no pricing session until tomorrow."""
        future = dilution_event(
            dedup_key="future",
            payload={
                **dilution_event().payload,
                "accepted_at": datetime(2026, 8, 25, 17, 0, tzinfo=MARKET_TZ).isoformat(),
            },
        )
        story = build_story(store, future, prices=price_frame(), factors=factor_frame())
        assert story.reaction.confidence is Confidence.UNAVAILABLE
        assert "may not have happened yet" in story.reaction.note

    def test_the_first_bar_has_no_prior_close_to_compare(self, store):
        early = dilution_event(
            dedup_key="early",
            payload={
                **dilution_event().payload,
                "accepted_at": datetime(2026, 8, 20, 10, 0, tzinfo=MARKET_TZ).isoformat(),
            },
        )
        story = build_story(store, early, prices=price_frame(), factors=factor_frame())
        assert story.reaction.confidence is Confidence.UNAVAILABLE
        assert "no prior close" in story.reaction.note


class TestAttribution:
    def test_without_a_sensitivity_the_slot_is_unavailable(self, store):
        story = build_story(store, dilution_event(), prices=price_frame(), factors=factor_frame())
        assert story.attribution.confidence is Confidence.UNAVAILABLE
        assert story.attribution.verdict is Verdict.UNKNOWN

    def test_a_model_estimate_is_never_reported_as_measured(self, store):
        """However good the fit, a regression is an estimate."""
        store.save_sensitivity(sensitivity())
        story = build_story(store, dilution_event(), prices=price_frame(), factors=factor_frame())
        assert story.attribution.confidence is Confidence.ESTIMATED

    def test_the_residual_carries_the_vol_it_is_measured_against(self, store):
        store.save_sensitivity(sensitivity())
        story = build_story(store, dilution_event(), prices=price_frame(), factors=factor_frame())
        assert story.attribution.resid_vol == pytest.approx(0.0753)
        assert story.attribution.r2 == pytest.approx(0.2737)

    def test_would_have_fired_matches_the_alerting_threshold(self, store):
        store.save_sensitivity(sensitivity())
        story = build_story(store, dilution_event(), prices=price_frame(), factors=factor_frame())
        z = story.attribution.residual_z
        assert story.attribution.would_have_fired == (abs(z) >= 2.0)

    def test_a_low_residual_vol_name_registers_the_same_move_more_strongly(self, store):
        """CRDO's 4.78%/day vol makes a -22% day a 4.9 sigma event; AAOI's
        7.53% makes -14.8% routine. The verdict must follow the name."""
        store.save_sensitivity(sensitivity(resid_vol=0.0478))
        loud = build_story(store, dilution_event(), prices=price_frame(), factors=factor_frame())
        store.save_sensitivity(sensitivity(resid_vol=0.20))
        quiet = build_story(store, dilution_event(), prices=price_frame(), factors=factor_frame())
        assert abs(loud.attribution.residual_z) > abs(quiet.attribution.residual_z)


class TestCorroboration:
    def source(self, **kw) -> SourceItem:
        base = dict(
            tier=SourceTier.AGGREGATOR,
            provider="example.com",
            url="https://example.com/a",
            title="AAOI falls on offering",
            published_at=FILED + timedelta(hours=2),
            entity=EntityMatch(symbol="AAOI", method=MatchMethod.COMPANY_NAME),
            doc_type="NEWS",
        )
        return SourceItem(**{**base, **kw})

    def test_items_in_the_window_are_collected_and_counted_by_tier(self, store):
        store.save_source_item(self.source())
        store.save_source_item(
            self.source(url="https://sec.gov/b", tier=SourceTier.PRIMARY, accession="b")
        )
        story = build_story(store, dilution_event(), prices=price_frame(), factors=factor_frame())
        assert len(story.corroboration.items) == 2
        assert story.corroboration.primary_count == 1
        assert story.corroboration.aggregator_count == 1

    def test_the_anchor_document_is_not_listed_as_its_own_corroboration(self, store):
        store.save_source_item(
            self.source(url="https://www.sec.gov/x.htm", tier=SourceTier.PRIMARY, accession="x")
        )
        story = build_story(store, dilution_event(), prices=price_frame(), factors=factor_frame())
        assert story.corroboration.items == []

    def test_items_far_outside_the_window_are_excluded(self, store):
        store.save_source_item(
            self.source(published_at=FILED + timedelta(days=30), accession="far")
        )
        story = build_story(store, dilution_event(), prices=price_frame(), factors=factor_frame())
        assert story.corroboration.items == []

    def test_another_symbols_coverage_is_not_borrowed(self, store):
        store.save_source_item(
            self.source(accession="crdo", entity=EntityMatch(symbol="CRDO", method=MatchMethod.CIK))
        )
        story = build_story(store, dilution_event(), prices=price_frame(), factors=factor_frame())
        assert story.corroboration.items == []


class TestThesis:
    """Three states that must not be confused with one another."""

    def _create_table(self, store):
        store._conn.execute(
            "CREATE TABLE theses (symbol TEXT, title TEXT, content TEXT, "
            "conviction TEXT, status TEXT, updated_at TEXT)"
        )

    def test_a_database_with_no_theses_table_is_not_reported_as_broken(self, store):
        """A fresh install has no theses table; that is not a read failure."""
        story = build_story(store, dilution_event(), prices=price_frame(), factors=factor_frame())
        assert story.thesis.confidence is Confidence.MEASURED
        assert story.thesis.exists is False
        assert "no stated thesis" in story.thesis.note

    def test_a_symbol_with_no_thesis_says_there_is_nothing_to_test_against(self, store):
        self._create_table(store)
        story = build_story(store, dilution_event(), prices=price_frame(), factors=factor_frame())
        assert story.thesis.exists is False
        assert "no stated thesis" in story.thesis.note

    def test_an_existing_thesis_is_attached(self, store):
        self._create_table(store)
        store._conn.execute(
            "INSERT INTO theses VALUES ('AAOI', 'Optics capacity', ?, 'LOW', 'active', "
            "'2026-09-01')",
            (
                "AAOI makes optical transceivers; the AI datacentre build-out drives 800G "
                "demand faster than capacity can be added, and in-house laser fabrication "
                "is the constraint competitors cannot buy their way around this cycle.",
            ),
        )
        story = build_story(store, dilution_event(), prices=price_frame(), factors=factor_frame())
        assert story.thesis.exists is True
        assert story.thesis.title == "Optics capacity"
        assert story.thesis.conviction == "LOW"

    def test_a_broken_theses_table_degrades_instead_of_raising(self, store):
        store._conn.execute("CREATE TABLE theses (wrong_column TEXT)")
        story = build_story(store, dilution_event(), prices=price_frame(), factors=factor_frame())
        assert story.thesis.confidence is Confidence.UNAVAILABLE


class TestStoryShape:
    def test_unfilled_slots_are_enumerated_not_hidden(self, store):
        story = build_story(store, dilution_event(), prices=pd.DataFrame(), factors=pd.DataFrame())
        assert set(story.unavailable_slots) >= {"position", "reaction", "attribution"}

    def test_a_backfilled_event_knows_it_was_learned_late(self, store):
        story = build_story(store, dilution_event(), prices=price_frame(), factors=factor_frame())
        assert story.anchor.backfilled is True

    def test_the_anchor_keeps_the_verbatim_quote_and_url(self, store):
        story = build_story(store, dilution_event(), prices=price_frame(), factors=factor_frame())
        assert "600,000,000" in story.anchor.quote
        assert story.anchor.url.startswith("https://")

    def test_assembly_never_raises_on_a_bare_event(self, store):
        """An event with an empty payload must still produce a story."""
        bare = Event(
            source=EventSource.COMPUTED,
            kind="STOP_BREACHED",
            tier=EventTier.A,
            symbol="AAOI",
            dedup_key="bare",
            payload={},
        )
        story = build_story(store, bare, prices=price_frame(), factors=factor_frame())
        assert story.symbol == "AAOI"

    def test_an_unparseable_occurred_at_falls_back_to_the_event_timestamp(self, store):
        broken = dilution_event(
            dedup_key="broken",
            payload={**dilution_event().payload, "accepted_at": "last Tuesday"},
        )
        story = build_story(store, broken, prices=price_frame(), factors=factor_frame())
        assert story.anchor.occurred_at.tzinfo is not None

    def test_timestamps_stay_timezone_aware(self, store):
        story = build_story(store, dilution_event(), prices=price_frame(), factors=factor_frame())
        assert story.assembled_at.tzinfo is not None
        assert story.anchor.occurred_at.tzinfo is not None


class TestArchiveFallback:
    """Every holding with a filing history deserves a story.

    Found end to end on AMD: eight filings since July, none of them evented
    (all older than the five-day event window when first seen), and so no
    story at all. The event stream carries what was worth reporting; the
    archive carries what happened. A story needs the second.
    """

    def source(self, **kw) -> SourceItem:
        base = dict(
            tier=SourceTier.PRIMARY,
            provider="SEC EDGAR",
            url="https://www.sec.gov/amd.htm",
            title="8-K: ADVANCED MICRO DEVICES INC",
            published_at=FILED,
            entity=EntityMatch(symbol="AMD", cik=2488, method=MatchMethod.CIK),
            doc_type="8-K",
            item_codes=["5.02", "9.01"],
            accession="0000002488-26-000163",
        )
        return SourceItem(**{**base, **kw})

    def test_a_symbol_with_filings_but_no_events_still_gets_a_story(self, store):
        from advisor.story.assemble import _anchor_from_archive

        store.save_source_item(self.source())
        anchors = _anchor_from_archive(store, "AMD", 3)
        assert len(anchors) == 1
        assert anchors[0].kind == "FILING_MANAGEMENT_CHANGE"

    def test_archive_anchors_are_tier_c(self, store):
        """They were never judged worth interrupting for; that does not change."""
        from advisor.story.assemble import _anchor_from_archive

        store.save_source_item(self.source())
        assert _anchor_from_archive(store, "AMD", 3)[0].tier is EventTier.C

    def test_the_anchor_keeps_the_filing_timestamp_not_todays(self, store):
        from advisor.story.assemble import _anchor_from_archive

        store.save_source_item(self.source())
        anchor = _anchor_from_archive(store, "AMD", 3)[0]
        assert anchor.payload["accepted_at"].startswith("2026-08-21")
        assert anchor.payload["from_archive"] is True

    def test_news_items_are_not_used_as_anchors(self, store):
        """A headline is context, not an event a story can be built around."""
        from advisor.story.assemble import _anchor_from_archive

        store.save_source_item(
            self.source(
                tier=SourceTier.UNTAGGED,
                doc_type="NEWS",
                accession=None,
                url="https://news/x",
            )
        )
        assert _anchor_from_archive(store, "AMD", 3) == []

    def test_a_symbol_with_nothing_archived_yields_nothing(self, store):
        from advisor.story.assemble import _anchor_from_archive

        assert _anchor_from_archive(store, "ZZZZ", 3) == []
