"""News ingest: the age filter, and what happens when the schema moves.

The bug these exist to prevent: yfinance renamed its publish-time field, the
old name kept returning None, and `max_age_hours` silently became a no-op.
`news_headlines("AMD", max_age_hours=0)` returned five headlines. Nothing
raised, nothing logged, and the risk agent was fed headlines of any age while
asking for the last 48 hours.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from advisor.data.news import NewsItem, _published_at, news_headlines, news_items

NOW = datetime(2026, 9, 6, 15, 0, tzinfo=timezone.utc)


def nested(title: str, pub: str | None, **extra) -> dict:
    """The current yfinance shape: everything under "content"."""
    content: dict = {"title": title, **extra}
    if pub is not None:
        content["pubDate"] = pub
    return {"id": "x", "content": content}


def flat(title: str, epoch: float | None) -> dict:
    """The legacy yfinance shape."""
    item: dict = {"title": title}
    if epoch is not None:
        item["providerPublishTime"] = epoch
    return item


class TestTimestampExtraction:
    def test_reads_the_current_nested_iso_field(self):
        got = _published_at(nested("t", "2026-09-06T14:35:11Z"))
        assert got == datetime(2026, 9, 6, 14, 35, 11, tzinfo=timezone.utc)

    def test_still_reads_the_legacy_epoch_field(self):
        assert _published_at(flat("t", NOW.timestamp())) == NOW

    def test_falls_back_to_display_time(self):
        raw = {"content": {"title": "t", "displayTime": "2026-09-01T10:00:00Z"}}
        assert _published_at(raw) == datetime(2026, 9, 1, 10, 0, tzinfo=timezone.utc)

    def test_a_naive_timestamp_is_treated_as_utc_not_local(self):
        got = _published_at({"content": {"title": "t", "pubDate": "2026-09-06T14:35:11"}})
        assert got.tzinfo is not None
        assert got.utcoffset() == timedelta(0)

    def test_an_item_with_no_timestamp_yields_none(self):
        assert _published_at(nested("t", None)) is None

    def test_an_unparseable_timestamp_yields_none(self):
        assert _published_at(nested("t", "last Tuesday")) is None

    def test_a_nonsense_epoch_does_not_raise(self):
        assert _published_at(flat("t", "not-a-number")) is None
        assert _published_at(flat("t", 1e20)) is None


class TestAgeFilter:
    @pytest.fixture
    def feed(self, monkeypatch):
        """Patch the yfinance Ticker the module imports at call time."""

        def install(items):
            import yfinance as yf

            class FakeTicker:
                def __init__(self, symbol):
                    self.news = items

            monkeypatch.setattr(yf, "Ticker", FakeTicker)

        return install

    def test_zero_hours_returns_nothing(self, feed):
        """The exact regression: this used to return every headline."""
        feed([nested("fresh", (datetime.now(timezone.utc) - timedelta(minutes=5)).isoformat())])
        assert news_headlines("AMD", max_age_hours=0.0) == []

    def test_old_items_are_excluded(self, feed):
        now = datetime.now(timezone.utc)
        feed(
            [
                nested("recent", (now - timedelta(hours=2)).isoformat()),
                nested("ancient", (now - timedelta(days=90)).isoformat()),
            ]
        )
        assert news_headlines("AMD", max_age_hours=48.0) == ["recent"]

    def test_undated_items_are_dropped_not_passed_through(self, feed):
        """Fail closed: an undated headline cannot be shown as recent."""
        now = datetime.now(timezone.utc)
        feed([nested("dated", (now - timedelta(hours=1)).isoformat()), nested("undated", None)])
        assert news_headlines("AMD") == ["dated"]

    def test_exactly_at_the_age_boundary_is_included(self, feed):
        now = datetime.now(timezone.utc)
        feed([nested("edge", (now - timedelta(hours=48) + timedelta(seconds=5)).isoformat())])
        assert news_headlines("AMD", max_age_hours=48.0) == ["edge"]

    def test_results_are_newest_first(self, feed):
        now = datetime.now(timezone.utc)
        feed(
            [
                nested("older", (now - timedelta(hours=10)).isoformat()),
                nested("newest", (now - timedelta(hours=1)).isoformat()),
                nested("middle", (now - timedelta(hours=5)).isoformat()),
            ]
        )
        assert news_headlines("AMD") == ["newest", "middle", "older"]

    def test_limit_is_applied_after_sorting(self, feed):
        now = datetime.now(timezone.utc)
        feed([nested(f"h{i}", (now - timedelta(hours=i + 1)).isoformat()) for i in range(10)])
        assert news_headlines("AMD", limit=2) == ["h0", "h1"]

    def test_a_future_dated_item_is_not_excluded_as_stale(self, feed):
        """Clock skew at the provider must not silently hide a real headline."""
        now = datetime.now(timezone.utc)
        feed([nested("just filed", (now + timedelta(minutes=10)).isoformat())])
        assert news_headlines("AMD") == ["just filed"]


class TestDegradation:
    def test_a_feed_that_raises_returns_empty(self, monkeypatch):
        import yfinance as yf

        class Boom:
            def __init__(self, symbol):
                raise RuntimeError("rate limited")

        monkeypatch.setattr(yf, "Ticker", Boom)
        assert news_headlines("AMD") == []

    def test_a_none_feed_returns_empty(self, monkeypatch):
        import yfinance as yf

        class NoNews:
            def __init__(self, symbol):
                self.news = None

        monkeypatch.setattr(yf, "Ticker", NoNews)
        assert news_headlines("AMD") == []

    def test_items_without_titles_are_skipped(self, monkeypatch):
        import yfinance as yf

        now = datetime.now(timezone.utc).isoformat()

        class Partial:
            def __init__(self, symbol):
                self.news = [{"content": {"pubDate": now}}, nested("real", now)]

        monkeypatch.setattr(yf, "Ticker", Partial)
        assert news_headlines("AMD") == ["real"]


class TestProvenance:
    def test_items_carry_the_publisher_and_link(self, monkeypatch):
        import yfinance as yf

        now = datetime.now(timezone.utc).isoformat()
        raw = nested(
            "headline",
            now,
            provider={"displayName": "Reuters"},
            canonicalUrl={"url": "https://example.com/a"},
        )

        class One:
            def __init__(self, symbol):
                self.news = [raw]

        monkeypatch.setattr(yf, "Ticker", One)
        item = news_items("AMD")[0]
        assert item.provider == "Reuters"
        assert item.url == "https://example.com/a"

    def test_age_hours_is_computed_against_an_explicit_now(self):
        item = NewsItem(title="t", published_at=NOW - timedelta(hours=3))
        assert item.age_hours(NOW) == pytest.approx(3.0)
