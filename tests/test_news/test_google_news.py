"""Google News RSS: titles matched like any source, the company's own site told apart."""

from __future__ import annotations

from datetime import datetime, timedelta
from email.utils import format_datetime

import pytest
from advisor.daemon import market_calendar as mc
from advisor.news.google_news import brand, distress_query, domain, parse, search_news
from advisor.news.models import MatchMethod, SourceTier

NOW = datetime(2026, 9, 27, 9, 0, tzinfo=mc.MARKET_TZ)


def item(title, publisher, url, hours_ago=2, link="https://news.google.com/rss/articles/x"):
    when = format_datetime(NOW - timedelta(hours=hours_ago))
    return (
        f"<item><title>{title} - {publisher}</title><link>{link}</link>"
        f'<pubDate>{when}</pubDate><source url="{url}">{publisher}</source></item>'
    )


def rss(*items):
    return f"<rss version=\"2.0\"><channel>{''.join(items)}</channel></rss>"


def search(
    xml, symbol="AAOI", company="APPLIED OPTOELECTRONICS, INC.", website="https://www.ao-inc.com"
):
    return search_news(
        symbol, "q", company_name=company, website=website, days=3, now=NOW, get=lambda url: xml
    )


class TestDomain:
    @pytest.mark.parametrize(
        "url,expected",
        [
            ("https://newsroom.ao-inc.com/news/1", "ao-inc.com"),
            ("https://www.spacex.com", "spacex.com"),
            ("https://www.bbc.co.uk/news", "bbc.co.uk"),
            ("reuters.com", "reuters.com"),
            (None, ""),
            ("", ""),
        ],
    )
    def test_registrable_part(self, url, expected):
        assert domain(url) == expected

    @pytest.mark.parametrize(
        "site,expected",
        [
            ("https://www.spacex.com", "spacex"),
            ("https://www.ao-inc.com", None),  # not one word
            ("https://www.t1energy.com", None),  # a digit
            ("https://ir.io", None),
            (None, None),
        ],
    )
    def test_brand(self, site, expected):
        assert brand(site) == expected


class TestQuery:
    def test_a_short_ticker_is_left_out_when_the_name_is_known(self):
        """TE is a common word: the name carries the query."""
        q = distress_query("TE", "T1 Energy Inc.", "https://www.t1energy.com")
        assert q.startswith('"t1 energy" (bankruptcy OR "Chapter 11" OR')

    def test_the_web_brand_is_added_when_no_headline_uses_the_registered_name(self):
        q = distress_query("SPCX", "SPACE EXPLORATION TECHNOLOGIES CORP", "https://www.spacex.com")
        assert q.startswith('("space exploration" OR spacex OR SPCX) (')

    def test_a_brand_already_in_the_name_is_not_repeated(self):
        q = distress_query("NBIS", "Nebius Group N.V.", "https://nebius.com")
        assert q.startswith('("nebius" OR NBIS) (')

    def test_without_a_name_the_ticker_is_all_there_is(self):
        assert distress_query("TE", None).startswith("TE (")

    def test_the_window_goes_in_the_query(self):
        seen = []
        search_news("AAOI", "x", days=3, now=NOW, get=lambda url: seen.append(url) or rss())
        assert "when%3A3d" in seen[0]


class TestParse:
    def test_the_publisher_suffix_is_removed(self):
        (r,) = parse(rss(item("AOI files Chapter 11", "Reuters", "https://www.reuters.com")))
        assert r["title"] == "AOI files Chapter 11" and r["publisher"] == "Reuters"
        assert r["publisher_url"] == "https://www.reuters.com" and r["published"].tzinfo


class TestSearch:
    def test_the_company_s_own_site_is_the_company_speaking(self):
        (i,) = search(
            rss(
                item(
                    "Applied Optoelectronics announces restructuring",
                    "AOI Newsroom",
                    "https://newsroom.ao-inc.com",
                )
            )
        )
        assert i.tier is SourceTier.PRIMARY and i.doc_type == "COMPANY_STATEMENT"
        assert i.provider == "ao-inc.com"

    def test_a_wire_is_not_the_company(self):
        """Law firms publish investor alerts on the same wires."""
        (i,) = search(
            rss(
                item(
                    "Applied Optoelectronics investors alert: class action filed",
                    "PR Newswire",
                    "https://www.prnewswire.com",
                )
            )
        )
        assert i.tier is SourceTier.AGGREGATOR and i.doc_type == "NEWS"
        assert i.provider == "prnewswire.com"

    def test_an_item_that_does_not_name_the_company_is_dropped(self):
        assert search(rss(item("Optical stocks slide", "Reuters", "https://www.reuters.com"))) == []

    def test_the_web_brand_matches_a_headline_the_registered_name_never_would(self):
        xml = rss(item("SpaceX misses a debt payment", "CNBC", "https://www.cnbc.com"))
        (i,) = search(xml, "SPCX", "SPACE EXPLORATION TECHNOLOGIES CORP", "https://www.spacex.com")
        assert i.entity.method is MatchMethod.COMPANY_NAME

    def test_the_brand_must_be_a_whole_word(self):
        xml = rss(item("SpaceXAI ships Grok", "CNBC", "https://www.cnbc.com"))
        assert (
            search(xml, "SPCX", "SPACE EXPLORATION TECHNOLOGIES CORP", "https://www.spacex.com")
            == []
        )

    def test_old_and_undated_items_are_dropped(self):
        old = item(
            "Applied Optoelectronics defaults",
            "Reuters",
            "https://www.reuters.com",
            hours_ago=24 * 6,
        )
        undated = (
            "<item><title>Applied Optoelectronics defaults - X</title>"
            "<link>https://news.google.com/rss/articles/u</link><pubDate>soon</pubDate></item>"
        )
        assert search(rss(old, undated)) == []

    def test_without_a_website_nothing_is_the_company_s_own(self):
        xml = rss(
            item(
                "Applied Optoelectronics restructures",
                "AOI Newsroom",
                "https://newsroom.ao-inc.com",
            )
        )
        (i,) = search(xml, website=None)
        assert i.tier is SourceTier.AGGREGATOR

    def test_a_network_failure_is_no_items(self):
        def down(url):
            raise TimeoutError("google")

        assert search_news("AAOI", "q", now=NOW, get=down) == []

    def test_a_broken_feed_is_no_items(self):
        assert search("<html>captcha</html") == []
