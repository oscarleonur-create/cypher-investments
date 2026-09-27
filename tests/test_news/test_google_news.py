"""Google News RSS: titles matched like any source, the company's own site told apart."""

from __future__ import annotations

from datetime import datetime, timedelta
from email.utils import format_datetime

import pytest
from advisor.daemon import market_calendar as mc
from advisor.news.google_news import DISTRESS_TITLE, distress_query, domain, parse, search_news
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


class TestQuery:
    def test_a_short_ticker_is_left_out_when_the_name_is_known(self):
        """TE is a common word: the listed name carries the query."""
        q = distress_query("TE", "T1 Energy Inc.", "https://www.t1energy.com")
        assert q.startswith('"T1 Energy" (bankruptcy OR "Chapter 11" OR')

    def test_the_listed_names_replace_the_registered_one(self):
        q = distress_query("SPCX", "SPACE EXPLORATION TECHNOLOGIES CORP")
        assert q.startswith('(SpaceX OR "Space Exploration Technologies" OR SPCX) (')

    def test_an_unlisted_symbol_falls_back_to_the_registered_name(self):
        q = distress_query("ZZZZ", "Zeta Widgets Inc.")
        assert q.startswith('("zeta widgets" OR ZZZZ) (')

    def test_without_any_name_the_ticker_is_all_there_is(self):
        assert distress_query("QQ", None).startswith("QQ (")

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

    def test_a_listed_name_matches_a_headline_the_registered_name_never_would(self):
        xml = rss(item("SpaceX misses a debt payment", "CNBC", "https://www.cnbc.com"))
        (i,) = search(xml, "SPCX", "SPACE EXPLORATION TECHNOLOGIES CORP", "https://www.spacex.com")
        assert i.entity.method is MatchMethod.COMPANY_NAME

    def test_space_exploration_in_general_is_not_spacex(self):
        """Audit 2026-09-27: ESA, a merit badge and an astronomy club were kept as SPCX."""
        xml = rss(item("A call to boost European space exploration", "ESA", "https://www.esa.int"))
        assert search(xml, "SPCX", "SPACE EXPLORATION TECHNOLOGIES CORP") == []

    def test_the_title_filter_keeps_only_what_names_the_situation(self):
        xml = rss(
            item(
                "Applied Optoelectronics misses a coupon payment, default looms",
                "Reuters",
                "https://www.reuters.com",
                link="https://news.google.com/rss/articles/a",
            ),
            item(
                "Applied Optoelectronics rides the AI boom",
                "Zacks",
                "https://www.zacks.com",
                link="https://news.google.com/rss/articles/b",
            ),
        )
        items = search_news(
            "AAOI",
            "q",
            company_name="APPLIED OPTOELECTRONICS, INC.",
            days=3,
            now=NOW,
            title_filter=DISTRESS_TITLE,
            get=lambda url: xml,
        )
        assert [i.title for i in items] == [
            "Applied Optoelectronics misses a coupon payment, default looms"
        ]

    def test_the_name_must_be_a_whole_word(self):
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
