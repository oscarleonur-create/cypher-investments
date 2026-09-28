"""The news agent's inputs: the article behind a thin item, the context around a name,
the gate that lets arithmetic through and keeps invention out, and the weekly synthesis."""

from __future__ import annotations

import sqlite3
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest
from advisor.daemon import market_calendar as mc
from advisor.daemon.book import EQUITY, BookSnapshot, Position
from advisor.daemon.models import Event, EventSource, EventTier
from advisor.daemon.store import DaemonStore
from advisor.news import article, context, judge
from advisor.news.judge import (
    Direction,
    Draft,
    ItemCall,
    MarketRead,
    NewsJudgmentStore,
    SummaryDraft,
    check,
    judge_symbol,
    read_market,
    synthesize,
)
from advisor.news.models import EntityMatch, MatchMethod, SourceItem, SourceTier

ET = mc.MARKET_TZ
NOW = datetime(2026, 9, 27, 9, 0, tzinfo=ET)


@pytest.fixture(autouse=True)
def offline(monkeypatch):
    from advisor.entry import sheet
    from advisor.news import verify

    monkeypatch.setattr(verify, "fetch_page", lambda url: None)
    monkeypatch.setattr(sheet, "daily_closes", lambda symbol: [])


@pytest.fixture
def stores(tmp_path: Path):
    s, conn = DaemonStore(tmp_path / "r.db"), sqlite3.connect(str(tmp_path / "r.db"))
    yield s, conn
    s.close()
    conn.close()


def item(summary="", url="https://www.fool.com/a", tier=SourceTier.UNTAGGED, days_ago=1,
         title="Cerebras Shares Are Coming Out of Lockup in Waves"):  # fmt: skip
    return SourceItem(
        tier=tier,
        provider="www.fool.com",
        url=url,
        title=title,
        summary=summary or None,
        published_at=(NOW - timedelta(days=days_ago)).astimezone(timezone.utc),
        entity=EntityMatch(symbol="CBRS", method=MatchMethod.COMPANY_NAME),
        doc_type="NEWS",
    )


PAGE = """<html><head><script>var x = 'Buy now 99%';</script></head><body>
<nav><li>Home</li><li>Markets and more links here for navigation</li></nav>
<article><h1>Cerebras Shares Are Coming Out of Lockup in Waves</h1>
<p>Last Wednesday, about 14.6 million shares of Cerebras Systems became eligible for sale.</p>
<p>Three more waves of roughly 19.4 million shares each follow on Sept. 30, Oct. 14 and Oct. 28.</p>
<p>Three more waves of roughly 19.4 million shares each follow on Sept. 30, Oct. 14 and Oct. 28.</p>
<p>The full expiry comes on Nov. 9, or two trading days after third-quarter results,
whichever is first, and the stock has fallen around each large release so far this year.</p>
<p>Short.</p></article><footer><p>Copyright notice that is long enough to be a paragraph.</p>
</footer></body></html>"""


# ── The article ───────────────────────────────────────────────────────────


class TestArticle:
    def test_keeps_paragraphs_drops_scripts_nav_footer_and_duplicates(self):
        text = article.body_text(PAGE)
        assert "14.6 million shares" in text and "Oct. 28" in text
        assert "Buy now" not in text and "Copyright" not in text and "navigation" not in text
        assert text.count("Three more waves") == 1 and "Short." not in text

    def test_malformed_or_empty_html_reads_as_nothing(self):
        assert article.body_text("") == ""
        assert article.body_text("<p>") == ""

    def test_a_stub_page_is_unreadable(self):
        assert article.fetch_article("https://x", fetch=lambda u: "<p>" + "x" * 50 + "</p>") is None

    def test_a_blocked_page_is_unreadable(self):
        assert article.fetch_article("https://x", fetch=lambda u: None) is None

    def test_long_pages_are_capped(self):
        html = "".join(f"<p>{'word ' * 30} paragraph {n}</p>" for n in range(200))
        body = article.fetch_article("https://x", fetch=lambda u: html)
        assert len(body) == article.ARTICLE_CHARS

    def test_a_thin_item_is_read_from_its_page_once(self, stores):
        _, conn = stores
        cache, calls = article.ArticleCache(conn), []
        fetch = lambda u: calls.append(u) or PAGE  # noqa: E731
        text, how = article.text_for(item(summary="teaser"), cache, fetch=fetch)
        again, _ = article.text_for(item(summary="teaser"), cache, fetch=fetch)
        assert how == "article" and "14.6 million" in text and again == text
        assert len(calls) == 1

    def test_an_unreadable_page_is_remembered_and_the_feed_kept(self, stores):
        _, conn = stores
        cache, calls = article.ArticleCache(conn), []
        fetch = lambda u: calls.append(u)  # noqa: E731
        assert article.text_for(item(summary="teaser"), cache, fetch=fetch) == ("teaser", "feed")
        assert article.text_for(item(summary=""), cache, fetch=fetch) == ("", "headline")
        assert len(calls) == 1

    def test_a_long_feed_text_is_not_replaced(self):
        long = "x " * article.THIN_CHARS
        text, how = article.text_for(item(summary=long), None, fetch=lambda u: PAGE)
        assert how == "feed" and text == long.strip()

    def test_a_filing_lead_is_the_companys_words(self):
        f = item(summary="The company entered a lease.", tier=SourceTier.PRIMARY)
        assert article.text_for(f, None, fetch=lambda u: PAGE) == (
            "The company entered a lease.", "feed",
        )  # fmt: skip


# ── The context ───────────────────────────────────────────────────────────


def closes(start=date(2026, 6, 1), n=90, jump_on=None, jump=0.0):
    out, d, px = [], start, 100.0
    while len(out) < n:
        if mc.is_trading_day(d):
            px *= 1.01 if len(out) % 2 else 0.99
            if d == jump_on:
                px *= 1 + jump
            out.append((d, px))
        d += timedelta(days=1)
    return out


class TestContext:
    def test_position_line_names_weight_cost_and_the_exit_stop(self, stores):
        s, _ = stores
        s.save_book(BookSnapshot(as_of=NOW, net_liq=10_000, positions=[Position(
            account="A", symbol="CBRS", underlying="CBRS", instrument=EQUITY, quantity=2,
            avg_open_price=217.09, close_price=206.45)]))  # fmt: skip
        line = context.position_line(s, "CBRS", 0.03)
        assert "held, 2 shares" in line and "4.1% of the book" in line
        assert "average cost $217.09" in line and "exit stop $" in line and "above it" in line

    def test_not_held_and_no_book(self, stores):
        s, _ = stores
        assert "unknown" in context.position_line(s, "CBRS", 0.03)
        s.save_book(BookSnapshot(as_of=NOW, net_liq=10_000))
        assert "not held" in context.position_line(s, "CBRS", 0.03)

    def test_market_move_in_sigma(self):
        c = closes(jump_on=date(2026, 9, 22), jump=0.08)
        line, move, z = context.market_move(
            datetime(2026, 9, 21, 18, 0, tzinfo=ET), c, context.sigma_of(c[:60])
        )
        assert "2026-09-22" in line and move > 0.06 and z > 3

    def test_market_move_before_any_session(self):
        line, move, z = context.market_move(datetime(2026, 12, 1, tzinfo=ET), closes(), 0.02)
        assert move is None and "no session" in line

    def test_sigma_needs_twenty_sessions(self):
        assert context.sigma_of(closes(n=10)) is None

    def test_history_leaves_out_the_books_own_alarms(self, stores):
        """ "stop breached" (the -8% alarm) led the agent to say CBRS was past its real stop."""
        s, _ = stores
        for kind, tier in (("STOP_BREACHED", EventTier.A), ("FILING_DILUTION", EventTier.A)):
            s.emit(
                Event(ts=NOW - timedelta(days=3), source=EventSource.COMPUTED, kind=kind,
                      tier=tier, symbol="CBRS", payload={"label": kind}, dedup_key=kind)
            )  # fmt: skip
        hist = " ".join(context.history_lines(s, "CBRS", NOW, exclude=set()))
        assert "filing dilution" in hist and "stop breached" not in hist

    def test_history_leaves_out_the_items_being_judged(self, stores):
        s, _ = stores
        s.save_source_item(item(url="https://a"))
        s.save_source_item(item(url="https://b"))
        judged = {"https://a", item().title.lower()}
        assert context.history_lines(s, "CBRS", NOW, exclude=judged) == []


# ── The gate ──────────────────────────────────────────────────────────────


def call(**kw):
    base = dict(id="N1", about_company=True, event_type="INSIDER", direction="NEGATIVE",
                materiality="HIGH", novelty="NEW", basis="REPORTED",
                quote="14.6 million shares", why="Supply overhang.")  # fmt: skip
    return ItemCall(**{**base, **kw})


TEXT = "Last Wednesday, about 14.6 million shares became eligible; 800G demand is strong."
CONTEXT = "C2: Company size: market cap $49.06bn, 237.6M shares outstanding."


class TestGate:
    def test_a_share_of_shares_outstanding_is_arithmetic(self):
        c = call(magnitude="14.6M shares, about 6.1% of the 237.6M outstanding")
        kept, _ = check(c, TEXT, {}, known_text=CONTEXT)
        assert kept is not None

    def test_a_number_from_nowhere_still_voids(self):
        c = call(magnitude="about 42% of the float")
        kept, problems = check(c, TEXT, {}, known_text=CONTEXT)
        assert kept is None and "'magnitude' uses ['42']" in problems[0]

    def test_periods_and_product_names_are_not_quantities(self):
        c = call(why="Q3 and FY2027 matter; the 800G ramp continues.")
        assert check(c, TEXT, {}, known_text=CONTEXT)[0] is not None

    def test_a_product_name_the_item_never_used_is_checked(self):
        c = call(why="The 1.6T ramp continues.")
        assert check(c, TEXT, {}, known_text=CONTEXT)[0] is None

    def test_sec_form_names_are_not_quantities(self):
        """ "Watch the next 10-Q and Form 4 filings" dropped nine watches in the first run."""
        c = call(watch="The next 10-Q, any Form 4, 8-K or 13G, and the 424B5.")
        assert check(c, TEXT, {}, known_text=CONTEXT)[0].watch

    def test_an_invented_date_loses_the_watch_not_the_call(self):
        c = call(watch="The next wave on Nov. 19")
        kept, problems = check(c, TEXT, {}, known_text=CONTEXT)
        assert kept is not None and kept.watch == "" and "'watch' dropped" in problems[0]

    def test_a_supported_watch_is_kept(self):
        c = call(watch="The 14.6 million shares already free")
        assert check(c, TEXT, {}, known_text=CONTEXT)[0].watch


class TestMarketRead:
    @pytest.mark.parametrize(
        "direction,move,z,read",
        [
            (Direction.NEGATIVE, 0.051, 0.8, MarketRead.QUIET),  # the live case
            (Direction.NEGATIVE, -0.06, -1.5, MarketRead.MOVED_WITH),
            (Direction.NEGATIVE, 0.06, 1.5, MarketRead.MOVED_AGAINST),
            (Direction.POSITIVE, 0.06, 1.0, MarketRead.MOVED_WITH),  # exactly one sigma
            (Direction.MIXED, -0.06, -1.5, MarketRead.MOVED),
            (Direction.POSITIVE, None, None, MarketRead.UNKNOWN),
            (Direction.POSITIVE, 0.02, None, MarketRead.UNKNOWN),  # no sigma estimate
        ],
    )
    def test_read(self, direction, move, z, read):
        assert read_market(direction, move, z) is read


# ── Judging with context, and the week ───────────────────────────────────


def save(s, it):
    s.save_source_item(it)
    s.emit(Event(ts=it.published_at, source=EventSource.YFINANCE, kind="NEWS_CONTEXT",
                 tier=EventTier.C, symbol="CBRS", dedup_key=it.url,
                 payload={"url": it.url, "title": it.title, "verified": "CONFIRMED"}))  # fmt: skip


class TestJudgeWithContext:
    def test_the_model_sees_the_article_the_context_and_the_market(self, stores):
        s, conn = stores
        save(s, item(summary="teaser"))
        seen = []

        def complete(system, user):
            seen.append(user)
            return Draft(judgments=[call()])

        js, _ = judge_symbol(s, conn, "CBRS", NOW, complete=complete,
                             fetch=lambda u: PAGE, closes=lambda sym: closes())  # fmt: skip
        assert "C1: Position:" in seen[0] and "full article" in seen[0]
        assert "14.6 million shares" in seen[0] and "Market:" in seen[0]
        assert js[0].read_from == "article" and js[0].market.startswith("Market:")

    def test_a_headline_only_item_is_labelled_so(self, stores):
        s, conn = stores
        save(s, item(summary=""))
        seen = []

        def complete(system, user):
            seen.append(user)
            return Draft(judgments=[call(quote="Coming Out of Lockup", why="Unclear.")])

        js, _ = judge_symbol(s, conn, "CBRS", NOW, complete=complete)
        assert "HEADLINE ONLY" in seen[0] and js[0].read_from == "headline"

    def test_the_market_read_is_computed_not_taken(self, stores):
        s, conn = stores
        save(s, item(summary="teaser"))
        c = call(market_read=MarketRead.MOVED_WITH)
        js, _ = judge_symbol(s, conn, "CBRS", NOW, complete=lambda a, b: Draft(judgments=[c]),
                             fetch=lambda u: PAGE)  # fmt: skip
        assert js[0].market_read is MarketRead.UNKNOWN  # no prices: nothing to read

    def test_a_synthesis_follows_new_judgments(self, stores):
        s, conn = stores
        save(s, item(summary="teaser"))
        summarize = lambda a, b: SummaryDraft(  # noqa: E731
            net=Direction.NEGATIVE, headline="Lock-up supply", text="14.6 million shares freed."
        )
        judge_symbol(s, conn, "CBRS", NOW, complete=lambda a, b: Draft(judgments=[call()]),
                     summarize=summarize, fetch=lambda u: PAGE)  # fmt: skip
        got = NewsJudgmentStore(conn).latest_summary("CBRS")
        assert got.net is Direction.NEGATIVE and got.items == 1 and got.day == NOW.date()


class TestSynthesize:
    def _judged(self, stores, **kw):
        s, conn = stores
        save(s, item(summary="teaser"))
        judge_symbol(s, conn, "CBRS", NOW, complete=lambda a, b: Draft(judgments=[call(**kw)]),
                     fetch=lambda u: PAGE)  # fmt: skip
        return conn

    def test_an_invented_number_voids_the_synthesis(self, stores):
        conn = self._judged(stores)
        bad = lambda a, b: SummaryDraft(net=Direction.NEGATIVE, headline="h", text="99 waves.")  # noqa: E731
        got, problems = synthesize(conn, "CBRS", NOW, [], summarize=bad)
        assert got is None and "['99']" in problems[0]
        assert NewsJudgmentStore(conn).latest_summary("CBRS") is None

    def test_only_off_topic_items_get_no_synthesis(self, stores):
        conn = self._judged(stores, about_company=False)
        calls = []
        got, _ = synthesize(conn, "CBRS", NOW, [], summarize=lambda a, b: calls.append(1))
        assert got is None and calls == []

    def test_a_model_failure_is_a_problem(self, stores):
        conn = self._judged(stores)

        def boom(a, b):
            raise TimeoutError("slow")

        got, problems = synthesize(conn, "CBRS", NOW, [], summarize=boom)
        assert got is None and "synthesis failed" in problems[0]

    def test_the_days_last_synthesis_wins(self, stores):
        conn = self._judged(stores)
        for text in ("first.", "second."):
            synthesize(conn, "CBRS", NOW, [], summarize=lambda a, b, t=text: SummaryDraft(
                net=Direction.NEUTRAL, headline="h", text=t))  # fmt: skip
        assert NewsJudgmentStore(conn).latest_summary("CBRS").text == "second."


def test_the_prompts_forbid_advice():
    assert "do NOT recommend" in judge.SYSTEM_PROMPT and "do NOT recommend" in judge.SUMMARY_PROMPT
    assert "never contradict" in judge.SYSTEM_PROMPT
