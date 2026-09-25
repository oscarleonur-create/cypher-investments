"""Source items becoming events, under the tier ceiling."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from advisor.daemon.market_calendar import MARKET_TZ, now_et
from advisor.daemon.models import EventTier
from advisor.daemon.store import DaemonStore
from advisor.news.ingest import _event_for_filing, backfill_leads, context_events
from advisor.news.models import (
    MAX_EVENT_TIER,
    EntityMatch,
    MatchMethod,
    SourceItem,
    SourceTier,
    capped_tier,
)


@pytest.fixture
def store(tmp_path: Path):
    s = DaemonStore(tmp_path / "research.db")
    yield s
    s.close()


def item(**kw) -> SourceItem:
    base = dict(
        tier=SourceTier.PRIMARY,
        provider="SEC EDGAR",
        url="https://www.sec.gov/x.htm",
        title="424B5: Applied Optoelectronics prospectus supplement",
        published_at=datetime(2026, 8, 21, 20, 9, 22, tzinfo=timezone.utc),
        entity=EntityMatch(symbol="AAOI", cik=1158114, method=MatchMethod.CIK),
        doc_type="424B5",
        accession="0001104659-26-099685",
    )
    return SourceItem(**{**base, **kw})


class TestTierCeiling:
    """The rule the module exists to enforce."""

    def test_untagged_can_never_interrupt(self):
        assert capped_tier(SourceTier.UNTAGGED, EventTier.A) is EventTier.C

    def test_an_aggregator_cannot_interrupt_either(self):
        assert capped_tier(SourceTier.AGGREGATOR, EventTier.A) is EventTier.B

    def test_primary_may_interrupt(self):
        assert capped_tier(SourceTier.PRIMARY, EventTier.A) is EventTier.A

    def test_the_cap_never_promotes(self):
        """A low-materiality filing stays low even from a trusted source."""
        assert capped_tier(SourceTier.PRIMARY, EventTier.C) is EventTier.C

    def test_every_tier_has_a_ceiling(self):
        assert set(MAX_EVENT_TIER) == set(SourceTier)

    def test_yfinance_context_events_are_all_tier_c(self):
        untagged = item(tier=SourceTier.UNTAGGED, accession=None, url="https://news/1")
        events = context_events([untagged], reason="residual divergence")
        assert all(e.tier is EventTier.C for e in events)


class TestFilingEvents:
    def test_a_dilution_filing_is_tier_a(self):
        event = _event_for_filing(item(), market_caps={})
        assert event.tier is EventTier.A
        assert event.kind == "FILING_DILUTION"
        assert event.symbol == "AAOI"

    def test_the_payload_carries_provenance(self):
        event = _event_for_filing(item(), market_caps={})
        assert event.payload["accession"] == "0001104659-26-099685"
        assert event.payload["accepted_at"].startswith("2026-08-21T20:09:22")
        assert event.payload["match"] == "CIK"
        assert event.payload["url"]

    def test_dilution_is_sized_against_market_cap(self, monkeypatch):
        import advisor.news.ingest as mod
        from advisor.news.enrich import OfferingSize

        monkeypatch.setattr(
            mod,
            "offering_size_for",
            lambda _: OfferingSize(amount_usd=600_000_000, quote="aggregate offering price"),
        )
        event = _event_for_filing(item(), market_caps={"AAOI": 8_960_160_678})
        assert event.payload["offering_usd"] == 600_000_000
        assert event.payload["dilution_pct"] == pytest.approx(0.067, abs=0.001)
        assert event.payload["quote"]
        assert event.tier is EventTier.A

    def test_immaterial_dilution_is_demoted_to_the_digest(self, monkeypatch):
        """A rounding-error raise is real but not worth waking someone."""
        import advisor.news.ingest as mod
        from advisor.news.enrich import OfferingSize

        monkeypatch.setattr(
            mod,
            "offering_size_for",
            lambda _: OfferingSize(amount_usd=1_000_000, quote="q"),
        )
        event = _event_for_filing(item(), market_caps={"AAOI": 8_960_160_678})
        assert event.payload["dilution_pct"] < mod.MATERIAL_DILUTION_PCT
        assert event.tier is EventTier.B

    def test_an_unsized_offering_still_interrupts(self, monkeypatch):
        """Failing to parse a size must not silence a real offering."""
        import advisor.news.ingest as mod

        monkeypatch.setattr(mod, "offering_size_for", lambda _: None)
        event = _event_for_filing(item(), market_caps={"AAOI": 8_960_160_678})
        assert event.tier is EventTier.A
        assert "dilution_pct" not in event.payload

    def test_a_missing_market_cap_leaves_the_size_unscaled(self, monkeypatch):
        import advisor.news.ingest as mod
        from advisor.news.enrich import OfferingSize

        monkeypatch.setattr(
            mod, "offering_size_for", lambda _: OfferingSize(amount_usd=6e8, quote="q")
        )
        event = _event_for_filing(item(), market_caps={})
        assert event.payload["offering_usd"] == 6e8
        assert "dilution_pct" not in event.payload
        assert event.tier is EventTier.A

    def test_a_periodic_report_does_not_interrupt(self):
        event = _event_for_filing(item(doc_type="10-Q", accession="x-1"), market_caps={})
        assert event.tier is EventTier.B

    def test_an_insider_form_is_context_only(self):
        event = _event_for_filing(item(doc_type="4", accession="x-2"), market_caps={})
        assert event.tier is EventTier.C

    def test_dedup_key_is_the_accession_number(self):
        assert item().dedup_key() == "0001104659-26-099685"

    def test_an_item_without_an_accession_dedups_on_url(self):
        a = item(accession=None, url="https://news/a")
        b = item(accession=None, url="https://news/a")
        c = item(accession=None, url="https://news/b")
        assert a.dedup_key() == b.dedup_key()
        assert a.dedup_key() != c.dedup_key()


class TestArchive:
    def test_the_same_filing_stored_twice_inserts_once(self, store):
        assert store.save_source_item(item()) is True
        assert store.save_source_item(item()) is False

    def test_two_racing_pollers_cannot_both_insert(self, store):
        """Same accession arriving from two runs — the PK settles it."""
        first = item()
        second = item(retrieved_at=now_et() + timedelta(seconds=1))
        assert store.save_source_item(first) is True
        assert store.save_source_item(second) is False

    def test_archived_items_come_back_newest_first(self, store):
        older = datetime(2026, 8, 1, tzinfo=timezone.utc)
        newer = datetime(2026, 8, 20, tzinfo=timezone.utc)
        store.save_source_item(item(accession="a", published_at=older))
        store.save_source_item(item(accession="b", published_at=newer))
        got = store.recent_source_items("AAOI")
        assert [i.accession for i in got] == ["b", "a"]

    def test_filtering_by_symbol_excludes_others(self, store):
        store.save_source_item(item(accession="a"))
        store.save_source_item(
            item(accession="b", entity=EntityMatch(symbol="AMD", method=MatchMethod.CIK))
        )
        assert [i.entity.symbol for i in store.recent_source_items("AMD")] == ["AMD"]

    def test_the_event_and_the_archive_share_a_dedup_key(self, store):
        """An event can always be traced back to the item that produced it."""
        source = item()
        store.save_source_item(source)
        event = _event_for_filing(source, market_caps={})
        assert event.dedup_key == source.dedup_key()


class TestTimestamps:
    def test_published_at_keeps_its_timezone(self):
        assert item().published_at.tzinfo is not None

    def test_retrieved_at_defaults_to_market_time(self):
        assert item().retrieved_at.tzinfo is not None
        assert item().retrieved_at.utcoffset() == datetime.now(MARKET_TZ).utcoffset()

    def test_an_item_cannot_be_built_without_a_publish_time(self):
        with pytest.raises(Exception):
            SourceItem(
                tier=SourceTier.PRIMARY,
                provider="x",
                url="u",
                title="t",
                entity=EntityMatch(symbol="AAOI", method=MatchMethod.CIK),
            )


class TestPerSymbolWatermark:
    """One global EDGAR watermark skipped most of the book.

    Found end to end: AMD had eight filings since July and none archived,
    because the single watermark had advanced past them on another symbol's
    behalf. Six of twelve positions had nothing at all. The archive already
    records what has been seen per symbol, so it *is* the watermark.
    """

    def test_a_symbol_with_nothing_archived_has_no_watermark(self, store):
        assert store.latest_source_item_at("AMD") is None

    def test_the_watermark_is_the_newest_archived_filing(self, store):
        older = item(accession="a", published_at=datetime(2026, 7, 1, tzinfo=timezone.utc))
        newer = item(accession="b", published_at=datetime(2026, 8, 21, tzinfo=timezone.utc))
        store.save_source_item(older)
        store.save_source_item(newer)
        assert store.latest_source_item_at("AAOI") == newer.published_at

    def test_one_symbols_watermark_does_not_move_anothers(self, store):
        """The whole bug in one assertion."""
        store.save_source_item(item(accession="a"))
        assert store.latest_source_item_at("AAOI") is not None
        assert store.latest_source_item_at("AMD") is None

    def test_news_items_do_not_advance_the_filing_watermark(self, store):
        """A headline arriving must not make the daemon skip a filing."""
        store.save_source_item(
            item(
                accession=None,
                url="https://news/x",
                tier=SourceTier.UNTAGGED,
                published_at=datetime(2026, 9, 13, tzinfo=timezone.utc),
            )
        )
        assert store.latest_source_item_at("AAOI", tier="PRIMARY") is None


class TestLeads:
    """A filing event says what the filing is, not just what kind it is."""

    HOUSTON = (
        "On August 31, 2026, Applied Optoelectronics, Inc. entered into two separate lease "
        "agreements with Hightower Phase II Owner, LLC for two industrial buildings to be "
        "constructed in Houston, Texas."
    )

    def eight_k(self, **kw) -> SourceItem:
        return item(
            doc_type="8-K",
            item_codes=["1.01", "2.03", "9.01"],
            accession="0001683168-26-006862",
            title="8-K: APPLIED OPTOELECTRONICS, INC. entered a material definitive agreement",
            **kw,
        )

    def test_an_8k_event_carries_its_lead(self):
        event = _event_for_filing(self.eight_k(summary=self.HOUSTON), market_caps={})
        assert event.payload["lead"] == self.HOUSTON

    def test_no_summary_means_no_lead_key(self):
        event = _event_for_filing(self.eight_k(), market_caps={})
        assert "lead" not in event.payload

    def test_the_lead_does_not_change_classification(self):
        """The 8-K lead must never be read by the 6-K headline classifier."""
        with_lead = _event_for_filing(
            self.eight_k(summary="Nebius announces convertible notes offering."), market_caps={}
        )
        without = _event_for_filing(self.eight_k(), market_caps={})
        assert with_lead.kind == without.kind == "FILING_MATERIAL_AGREEMENT"
        assert with_lead.tier == without.tier

    def test_a_news_context_event_carries_the_publishers_abstract(self):
        news = item(
            tier=SourceTier.UNTAGGED,
            accession=None,
            doc_type="NEWS",
            url="https://news/aaoi",
            summary="Applied Optoelectronics faces supply constraints despite surging 800G demand.",
        )
        (event,) = context_events([news], reason="STOP_BREACHED")
        assert event.payload["lead"].startswith("Applied Optoelectronics faces supply")


class TestBackfillLeads:
    HOUSTON = TestLeads.HOUSTON

    def archive(self, store, **kw):
        filing = TestLeads().eight_k(**kw)
        store.save_source_item(filing)
        store.emit(_event_for_filing(filing, market_caps={}))
        return filing

    def test_an_archived_8k_and_its_event_get_the_lead(self, store):
        self.archive(store)
        result = backfill_leads(store, read_lead=lambda _i: self.HOUSTON)
        assert (result.items_read, result.items_filled, result.events_filled) == (1, 1, 1)
        (event,) = store.recent_events()
        assert event.payload["lead"] == self.HOUSTON
        # Nothing else in the event moved.
        assert event.payload["kind"] == "MATERIAL_AGREEMENT"
        assert store.get_source_item("0001683168-26-006862").summary == self.HOUSTON

    def test_a_second_run_is_a_no_op(self, store):
        self.archive(store)
        backfill_leads(store, read_lead=lambda _i: self.HOUSTON)
        calls = []
        again = backfill_leads(store, read_lead=lambda i: calls.append(i) or "different")
        assert calls == []
        assert (again.items_filled, again.events_filled) == (0, 0)
        assert store.recent_events()[0].payload["lead"] == self.HOUSTON

    def test_an_existing_summary_is_never_overwritten(self, store):
        self.archive(store, summary=self.HOUSTON)
        result = backfill_leads(store, read_lead=lambda _i: "overwritten")
        assert result.items_read == 0
        assert store.get_source_item("0001683168-26-006862").summary == self.HOUSTON

    def test_edgar_failing_is_reported_not_raised(self, store):
        self.archive(store)

        def boom(_item):
            raise TimeoutError("EDGAR timed out")

        result = backfill_leads(store, read_lead=boom)
        assert result.items_filled == 0
        assert "EDGAR timed out" in result.errors[0]
        assert "lead" not in store.recent_events()[0].payload

    def test_an_unreadable_filing_leaves_the_event_alone(self, store):
        self.archive(store)
        result = backfill_leads(store, read_lead=lambda _i: None)
        assert (result.items_filled, result.events_filled) == (0, 0)

    def test_an_event_whose_item_was_never_archived_is_skipped(self, store):
        store.emit(_event_for_filing(TestLeads().eight_k(), market_caps={}))
        assert backfill_leads(store, read_lead=lambda _i: self.HOUSTON).events_filled == 0

    def test_empty_store(self, store):
        result = backfill_leads(store, read_lead=lambda _i: self.HOUSTON)
        assert (result.items_read, result.events_filled, result.errors) == (0, 0, [])
