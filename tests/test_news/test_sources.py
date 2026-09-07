"""Source adapters: timestamp parsing, tier assignment, and query shape."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest
from advisor.news.ingest import DEFAULT_KEYWORDS, REASON_KEYWORDS
from advisor.news.models import SourceTier
from advisor.news.tavily import MIN_RELEVANCE_SCORE, _parse_published


class TestTavilyTimestamps:
    def test_parses_the_rfc_1123_format_tavily_actually_returns(self):
        got = _parse_published("Fri, 21 Aug 2026 20:23:01 GMT")
        assert got == datetime(2026, 8, 21, 20, 23, 1, tzinfo=timezone.utc)

    def test_the_parsed_time_matches_the_sec_acceptance_time(self):
        """Two independent sources agreeing to the second is the whole point."""
        news = _parse_published("Fri, 21 Aug 2026 20:14:19 GMT")
        sec = datetime(2026, 8, 21, 20, 14, 19, tzinfo=timezone.utc)
        assert news == sec

    def test_parses_iso_8601(self):
        assert _parse_published("2026-08-21T20:23:01+00:00").hour == 20

    def test_a_naive_timestamp_is_treated_as_utc(self):
        got = _parse_published("2026-08-21T20:23:01")
        assert got.tzinfo is not None and got.utcoffset() == timedelta(0)

    @pytest.mark.parametrize("bad", [None, "", "last Tuesday", "21/08/2026", "garbage"])
    def test_unparseable_values_return_none_rather_than_guessing(self, bad):
        assert _parse_published(bad) is None

    def test_a_score_gate_exists_and_is_not_zero(self):
        """Measured: a verbose query returns 6 results all below 0.5."""
        assert 0.0 < MIN_RELEVANCE_SCORE < 1.0


class TestQueryShape:
    """Search engines take terms, not sentences.

    Measured against Tavily's own relevance score on the AAOI offering:
    keywords scored 0.924 with 5/5 results kept, a full sentence describing
    the move scored 0.393 with 0/5 kept.
    """

    def test_every_keyword_set_is_short(self):
        for kind, keywords in REASON_KEYWORDS.items():
            assert len(keywords.split()) <= 3, f"{kind} keywords are too verbose"

    def test_the_kinds_that_trigger_lookups_all_have_keywords(self):
        from advisor.daemon.handlers import EXPLAINABLE_KINDS

        assert EXPLAINABLE_KINDS <= set(REASON_KEYWORDS)

    def test_an_unknown_kind_falls_back_rather_than_raising(self):
        assert REASON_KEYWORDS.get("SOMETHING_NEW", DEFAULT_KEYWORDS) == DEFAULT_KEYWORDS


class TestTierAssignment:
    def test_tavily_items_are_aggregator_tier(self, monkeypatch):
        import advisor.news.tavily as mod

        class FakeResponse:
            status_code = 200

            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "results": [
                        {
                            "title": "Applied Optoelectronics Sinks 12% on $600M Equity Offering",
                            "url": "https://example.com/a",
                            "published_date": "Fri, 21 Aug 2026 20:23:01 GMT",
                            "score": 0.92,
                            "content": "Applied Optoelectronics announced an offering.",
                        }
                    ]
                }

        monkeypatch.setattr(mod.httpx, "post", lambda *a, **k: FakeResponse())
        items = mod.search_news("AAOI", "q", company_name="Applied Optoelectronics, Inc.")
        assert len(items) == 1
        assert items[0].tier is SourceTier.AGGREGATOR
        assert items[0].published_at.year == 2026

    def test_low_scoring_results_are_dropped(self, monkeypatch):
        import advisor.news.tavily as mod

        class FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "results": [
                        {
                            "title": "Applied Optoelectronics something",
                            "url": "https://example.com/a",
                            "published_date": "Fri, 21 Aug 2026 20:23:01 GMT",
                            "score": 0.10,
                            "content": "",
                        }
                    ]
                }

        monkeypatch.setattr(mod.httpx, "post", lambda *a, **k: FakeResponse())
        assert mod.search_news("AAOI", "q", company_name="Applied Optoelectronics") == []

    def test_items_that_do_not_name_the_company_are_dropped(self, monkeypatch):
        import advisor.news.tavily as mod

        class FakeResponse:
            def raise_for_status(self):
                pass

            def json(self):
                return {
                    "results": [
                        {
                            "title": "HPE Cannot Build AI Servers Fast Enough",
                            "url": "https://example.com/a",
                            "published_date": "Fri, 21 Aug 2026 20:23:01 GMT",
                            "score": 0.95,
                            "content": "HPE is doing well.",
                        }
                    ]
                }

        monkeypatch.setattr(mod.httpx, "post", lambda *a, **k: FakeResponse())
        assert mod.search_news("AAOI", "q", company_name="Applied Optoelectronics, Inc.") == []

    def test_a_network_failure_returns_empty_rather_than_raising(self, monkeypatch):
        """A news outage must never stop the position or macro pillars."""
        import advisor.news.tavily as mod

        def boom(*a, **k):
            raise RuntimeError("connection reset")

        monkeypatch.setattr(mod.httpx, "post", boom)
        assert mod.search_news("AAOI", "q") == []

    def test_no_api_key_degrades_silently(self, monkeypatch):
        import advisor.news.tavily as mod

        class NoKey:
            tavily_api_key = ""
            search_endpoint = "https://x"
            http_timeout_seconds = 5

        monkeypatch.setattr("research_agent.config.ResearchConfig", lambda: NoKey())
        assert mod.search_news("AAOI", "q") == []
