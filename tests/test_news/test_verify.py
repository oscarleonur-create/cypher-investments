"""A news item's date checked against the publisher, and widened when the page cannot say."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from advisor.daemon import market_calendar as mc
from advisor.daemon.models import Event, EventSource, EventTier
from advisor.daemon.store import DaemonStore
from advisor.news.ingest import context_events
from advisor.news.models import EntityMatch, MatchMethod, SourceItem, SourceTier
from advisor.news.verify import (
    MAX_RESOLVES,
    Status,
    backfill,
    check_page,
    corroborate,
    page_date,
    same_story,
    usable,
    verify,
    verify_items,
)

NOW = datetime(2026, 9, 27, 12, 0, tzinfo=mc.MARKET_TZ)
UTC = timezone.utc


def item(
    title="Nebius Stock Forecast: How Did NBIS Secure $44B in Contracts",
    provider="tradingkey.com",
    url="https://www.tradingkey.com/a",
    at=datetime(2026, 9, 26, 21, 23, tzinfo=UTC),
):
    return SourceItem(
        tier=SourceTier.AGGREGATOR,
        provider=provider,
        url=url,
        title=title,
        published_at=at,
        entity=EntityMatch(symbol="NBIS", method=MatchMethod.COMPANY_NAME),
        doc_type="NEWS",
    )


def page(published: str) -> str:
    return f'<html><script type="application/ld+json">{{"datePublished": "{published}"}}</script>'


@dataclass(frozen=True)
class Copy:
    title: str
    provider: str
    published_at: datetime


class TestPageDate:
    @pytest.mark.parametrize(
        "html,expected",
        [
            (page("2026-06-06T12:00:00Z"), datetime(2026, 6, 6, 12, tzinfo=UTC)),
            (
                '<meta property="article:published_time" content="2026-09-24T11:35:00-04:00">',
                datetime(2026, 9, 24, 15, 35, tzinfo=UTC),
            ),
            (
                '<time datetime="2026-09-21T14:49:00+00:00">',
                datetime(2026, 9, 21, 14, 49, tzinfo=UTC),
            ),
        ],
    )
    def test_the_page_states_its_date(self, html, expected):
        d, date_only = page_date(html)
        assert d == expected and not date_only

    def test_a_date_alone_is_a_new_york_day(self):
        d, date_only = page_date(page("2026-09-23"))
        assert date_only and d.astimezone(mc.MARKET_TZ).date().isoformat() == "2026-09-23"

    def test_a_time_without_a_zone_is_not_trusted(self):
        assert page_date(page("2026-09-23T10:00:00")) is None

    def test_a_page_with_no_date_says_nothing(self):
        """Credo's investor-relations home page: Google dated it, the page does not."""
        assert page_date("<html><title>Credo - Investor Relations</title></html>") is None


class TestCheckPage:
    def test_a_three_month_old_story_is_corrected_to_its_real_date(self):
        """Audit 2026-09-27: Google said 2026-09-26, TradingKey's page says 2026-06-06."""
        v = check_page(datetime(2026, 9, 26, 21, 23, tzinfo=UTC), page("2026-06-06T12:00:00Z"))
        assert v.status is Status.CORRECTED and v.published_at.date().isoformat() == "2026-06-06"

    def test_within_the_tolerance_is_confirmed(self):
        v = check_page(datetime(2026, 9, 24, 12, 35, tzinfo=UTC), page("2026-09-24T11:35:00Z"))
        assert v.status is Status.CONFIRMED

    def test_a_rewritten_page_is_corrected(self):
        """IBD rewrites one URL daily; yfinance reported the rewrite, 43 hours late."""
        v = check_page(datetime(2026, 9, 27, 17, 0, tzinfo=UTC), page("2026-09-25T21:24:00Z"))
        assert v.status is Status.CORRECTED

    def test_a_day_only_page_confirms_the_same_new_york_day(self):
        claimed = datetime(2026, 9, 24, 0, 27, tzinfo=UTC)  # 20:27 ET on the 23rd
        assert check_page(claimed, page("2026-09-23")).status is Status.CONFIRMED
        assert check_page(claimed, page("2026-09-20")).status is Status.CORRECTED


class TestSameStory:
    def test_a_syndicated_copy_is_the_same_story(self):
        assert same_story(
            "Cerebras Drops 19% in 3 Months: Buy, Sell or Hold the Stock?",
            "Cerebras Drops 19% in 3 Months: Buy, Sell or Hold the Stock? — TradingView",
        )

    def test_two_stories_on_one_company_are_not(self):
        assert not same_story(
            "Nebius Stock Soars 8% as BNP Paribas Delivers Price Target Hike",
            "Nebius Rated Platinum in SemiAnalysis ClusterMAX",
        )


class TestCorroborate:
    def test_two_outlets_dating_one_story_alike(self):
        a = item(provider="zacks.com", title="Cerebras Drops 19% in 3 Months")
        copy = Copy(
            "Cerebras Drops 19% in 3 Months", "tradingview.com", a.published_at - timedelta(hours=3)
        )
        v = corroborate(a, [copy])
        assert v.status is Status.CORROBORATED and v.published_at == copy.published_at

    def test_one_outlet_alone_is_not_corroboration(self):
        a = item()
        assert corroborate(a, [Copy(a.title, "www.tradingkey.com", a.published_at)]) is None

    def test_dates_far_apart_do_not_corroborate(self):
        a = item(provider="zacks.com")
        assert (
            corroborate(a, [Copy(a.title, "cnbc.com", a.published_at - timedelta(days=90))]) is None
        )


class TestVerify:
    def test_the_page_decides_when_it_can(self):
        v = verify(item(), fetch=lambda url: page("2026-06-06T12:00:00Z"))
        assert v.status is Status.CORRECTED

    def test_a_blocked_page_widens_to_other_outlets(self):
        """Seeking Alpha answers 403; the same story elsewhere dates it."""
        a = item(provider="seekingalpha.com", url="https://seekingalpha.com/x")
        found = [Copy(a.title, "finance.yahoo.com", a.published_at + timedelta(minutes=20))]
        v = verify(a, fetch=lambda url: None, search=lambda title: found)
        assert v.status is Status.CORROBORATED and v.published_at == a.published_at

    def test_a_google_link_that_cannot_be_resolved_widens_too(self):
        a = item(url="https://news.google.com/rss/articles/abc")
        v = verify(a, resolve=lambda link: None, fetch=lambda url: pytest.fail("no url to fetch"),
                   search=lambda title: [])  # fmt: skip
        assert v.status is Status.UNVERIFIED and v.published_at is None

    def test_a_stale_story_only_google_dates_stays_unverified(self):
        """The TradingKey case with the page unreachable: one outlet, no corroboration."""
        a = item(url="https://news.google.com/rss/articles/abc")
        only_itself = [Copy(a.title, "tradingkey.com", a.published_at)]
        v = verify(a, resolve=lambda link: None, search=lambda title: only_itself)
        assert v.status is Status.UNVERIFIED

    def test_a_failed_widening_is_unverified_not_an_error(self):
        def boom(title):
            raise TimeoutError("google")

        assert verify(item(), fetch=lambda url: None, search=boom).status is Status.UNVERIFIED


class TestBatch:
    def test_items_carry_the_checked_date_and_the_claim(self):
        (v,) = verify_items([item()], fetch=lambda url: page("2026-06-06T12:00:00Z"))
        assert v.verified == "CORRECTED" and v.published_at.date().isoformat() == "2026-06-06"
        assert v.claimed_at == datetime(2026, 9, 26, 21, 23, tzinfo=UTC)

    def test_google_resolutions_are_capped_and_stop_at_the_first_refusal(self):
        calls = []

        def resolve(link):
            calls.append(link)
            return None

        items = [
            item(url=f"https://news.google.com/rss/articles/{i}", title=f"story {i}")
            for i in range(5)
        ]
        verify_items(items, resolve=resolve)
        assert len(calls) == 1  # refused once: the rest go straight to corroboration

    def test_resolutions_never_exceed_the_cap(self):
        calls = []
        items = [
            item(url=f"https://news.google.com/rss/articles/{i}", title=f"story {i}")
            for i in range(MAX_RESOLVES + 5)
        ]
        verify_items(
            items, resolve=lambda link: calls.append(link) or "https://x/y", fetch=lambda u: None
        )
        assert len(calls) == MAX_RESOLVES

    def test_an_item_already_checked_is_not_checked_again(self, tmp_path: Path):
        store = DaemonStore(tmp_path / "r.db")
        try:
            (checked,) = verify_items([item()], fetch=lambda url: page("2026-09-26T21:00:00Z"))
            store.save_source_item(checked)
            (again,) = verify_items(
                [item()], store=store, fetch=lambda url: pytest.fail("refetched")
            )
            assert again.verified == "CONFIRMED"
        finally:
            store.close()


class TestStored:
    def test_a_news_event_is_dated_at_its_checked_publication(self):
        (v,) = verify_items([item()], fetch=lambda url: page("2026-06-06T12:00:00Z"))
        (e,) = context_events([v], reason="DISTRESS")
        assert e.ts.date().isoformat() == "2026-06-06"
        assert e.payload["verified"] == "CORRECTED" and usable(e.payload)
        assert e.payload["claimed_at"].startswith("2026-09-26")

    def test_backfill_checks_old_news_once_and_moves_it_to_its_real_date(self, tmp_path: Path):
        store = DaemonStore(tmp_path / "r.db")
        try:
            store.emit(
                Event(
                    ts=NOW - timedelta(hours=10),
                    source=EventSource.CALENDAR,
                    kind="NEWS_CONTEXT",
                    tier=EventTier.C,
                    symbol="NBIS",
                    dedup_key="old",
                    payload={
                        "title": "Nebius Stock Forecast",
                        "provider": "tradingkey.com",
                        "url": "https://www.tradingkey.com/a",
                        "published_at": (NOW - timedelta(hours=10)).isoformat(),
                    },
                )
            )
            counts = backfill(store, NOW - timedelta(days=45),
                              fetch=lambda url: page("2026-06-06T12:00:00Z"))  # fmt: skip
            assert counts == {"CORRECTED": 1}
            assert store.recent_events(symbol="NBIS", since=NOW - timedelta(days=7)) == []
            assert backfill(store, NOW - timedelta(days=200)) == {}  # once
        finally:
            store.close()

    @pytest.mark.parametrize(
        "status,ok",
        [("CONFIRMED", True), ("CORRECTED", True), ("CORROBORATED", True), ("UNVERIFIED", False),
         (None, False)],
    )  # fmt: skip
    def test_only_checked_dates_are_usable(self, status, ok):
        assert usable({"verified": status}) is ok
