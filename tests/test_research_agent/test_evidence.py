"""Tests for research_agent.evidence."""

from __future__ import annotations

from research_agent.evidence import SourceRegistry, classify_tier


class TestClassifyTier:
    def test_tier1_sec_gov(self):
        assert classify_tier("https://www.sec.gov/cgi-bin/browse-edgar") == 1

    def test_tier1_investor_relations(self):
        assert classify_tier("https://investor.apple.com/earnings") == 1

    def test_tier1_ir_subdomain(self):
        assert classify_tier("https://ir.tesla.com/sec-filings") == 1

    def test_tier2_reuters(self):
        assert classify_tier("https://www.reuters.com/article/apple-earnings") == 2

    def test_tier2_bloomberg(self):
        assert classify_tier("https://www.bloomberg.com/news/apple") == 2

    def test_tier2_wsj(self):
        assert classify_tier("https://www.wsj.com/articles/apple-results") == 2

    def test_tier3_default(self):
        assert classify_tier("https://someblog.com/apple-thoughts") == 3

    def test_tier3_invalid_url(self):
        assert classify_tier("not-a-url") == 3


class TestSourceRegistry:
    def test_add_and_get(self):
        reg = SourceRegistry()
        sid = reg.add("https://reuters.com/article/1", title="Article 1")
        assert sid == "s1"
        assert reg.count == 1

    def test_deduplication(self):
        reg = SourceRegistry()
        s1 = reg.add("https://reuters.com/article/1", title="Article 1")
        s2 = reg.add("https://reuters.com/article/1", title="Article 1 again")
        assert s1 == s2
        assert reg.count == 1

    def test_multiple_sources(self):
        reg = SourceRegistry()
        reg.add("https://reuters.com/1", title="One")
        reg.add("https://bloomberg.com/2", title="Two")
        reg.add("https://sec.gov/3", title="Three")
        assert reg.count == 3
        sources = reg.all_sources()
        assert len(sources) == 3
        assert sources[0].tier == 2  # reuters
        assert sources[2].tier == 1  # sec.gov

    def test_get_source_by_id(self):
        reg = SourceRegistry()
        reg.add("https://reuters.com/1", title="Article")
        src = reg.get_source("s1")
        assert src is not None
        assert src.title == "Article"

    def test_get_id_existing(self):
        reg = SourceRegistry()
        reg.add("https://reuters.com/1")
        assert reg.get_id("https://reuters.com/1") == "s1"

    def test_get_id_missing(self):
        reg = SourceRegistry()
        assert reg.get_id("https://missing.com") is None

    def test_source_id_for_citation_creates(self):
        reg = SourceRegistry()
        sid = reg.source_id_for_citation("https://new.com")
        assert sid == "s1"
        assert reg.count == 1
