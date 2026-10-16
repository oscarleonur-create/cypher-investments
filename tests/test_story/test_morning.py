"""The morning read: each holding, what changed since the close, with sources (user, 2026-10-15)."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest
from advisor.daemon import market_calendar as mc
from advisor.daemon.book import EQUITY, EQUITY_OPTION, BookSnapshot, Position
from advisor.entry.proposal import Action, Proposal, Reason
from advisor.news.models import EntityMatch, MatchMethod, SourceItem, SourceTier
from advisor.story import morning as M

ET = mc.MARKET_TZ
THU_8 = datetime(2026, 10, 15, 8, 0, tzinfo=ET)


class J:
    """A judgment's shape, as the morning read uses it."""

    def __init__(self, sym, hours_ago, materiality="HIGH", about=True, direction="NEGATIVE",
                 what="a fact", now=THU_8):  # fmt: skip
        from types import SimpleNamespace as NS

        self.symbol, self.about_company, self.title = sym, about, f"{sym} headline"
        self.published_at = now - timedelta(hours=hours_ago)
        self.materiality, self.direction = NS(value=materiality), NS(value=direction)
        self.what, self.provider, self.url = what, "reuters.com", "https://x"


def filing(sym, hours_ago, tier=SourceTier.PRIMARY, now=THU_8):
    return SourceItem(tier=tier, provider="SEC EDGAR", url=f"https://sec/{sym}/{hours_ago}",
                      title=f"8-K: {sym}", published_at=now - timedelta(hours=hours_ago),
                      entity=EntityMatch(symbol=sym, method=MatchMethod.CIK), doc_type="8-K",
                      item_codes=["2.02"], summary="results for the quarter")  # fmt: skip


def pos(sym, qty=10.0, cost=100.0, price=110.0, close=105.0, instrument=EQUITY, account="A"):
    return Position(account=account, symbol=sym, underlying=sym, instrument=instrument,
                    quantity=qty, avg_open_price=cost, mark_price=price, close_price=close,
                    multiplier=1.0 if instrument == EQUITY else 100.0)  # fmt: skip


def prop(sym, action=Action.HOLD, reasons=("next results 2026-10-21 (in 4 sessions)",)):
    return Proposal(symbol=sym, session=date(2026, 10, 14), action=action,
                    built_at=datetime(2026, 10, 14, 9, 45, tzinfo=ET),
                    reasons=[Reason(text=r, source="s") for r in reasons])  # fmt: skip


class TestSince:
    @pytest.mark.parametrize("now,expect", [
        # Thursday 08:00: 24 hours back is earlier than Wednesday's close.
        (datetime(2026, 10, 15, 8, 0, tzinfo=ET), datetime(2026, 10, 14, 8, 0, tzinfo=ET)),
        # Monday 08:00: Friday's close, so the weekend is in it.
        (datetime(2026, 10, 19, 8, 0, tzinfo=ET), datetime(2026, 10, 16, 16, 0, tzinfo=ET)),
        # After the close: the last 24 hours.
        (datetime(2026, 10, 15, 17, 0, tzinfo=ET), datetime(2026, 10, 14, 17, 0, tzinfo=ET)),
        # The day after Thanksgiving, in its short session: Wednesday's close.
        (datetime(2026, 11, 27, 12, 0, tzinfo=ET), datetime(2026, 11, 25, 16, 0, tzinfo=ET)),
        # Thanksgiving itself: no session, so Wednesday's close or 24 hours.
        (datetime(2026, 11, 26, 9, 0, tzinfo=ET), datetime(2026, 11, 25, 9, 0, tzinfo=ET)),
    ])  # fmt: skip
    def test_the_last_close_or_24_hours_whichever_is_longer(self, now, expect):
        assert M.since_last_close(now) == expect


class TestItems:
    def test_material_news_about_the_company_since_the_close(self):
        since = M.since_last_close(THU_8)
        js = [J("X", 2, "MEDIUM", what="medium"), J("X", 5, "HIGH", what="high"),
              J("X", 3, "LOW"), J("X", 3, about=False), J("X", 40, "HIGH", what="old")]  # fmt: skip
        assert [i.text for i in M.news_items(js, since)] == ["high", "medium"]

    def test_only_the_companys_own_filings(self):
        since = M.since_last_close(THU_8)
        items = M.filing_items([filing("X", 3), filing("X", 2, tier=SourceTier.AGGREGATOR),
                                filing("X", 30)], since)  # fmt: skip
        assert len(items) == 1
        assert items[0].text == "8-K [2.02]: results for the quarter"
        assert items[0].source == "SEC EDGAR 8-K"

    def test_filings_first_then_high_then_medium(self):
        since = M.since_last_close(THU_8)
        items = M.news_items([J("X", 1, "MEDIUM", what="m"), J("X", 6, "HIGH", what="h")], since)
        items += M.filing_items([filing("X", 9)], since)
        assert [i.kind + (i.materiality or "") for i in sorted(items, key=M.order)] == [
            "filing", "newsHIGH", "newsMEDIUM"]  # fmt: skip


def test_one_story_told_twice_is_one_line_naming_both_outlets():
    def item(text, source):
        return M.Item(kind="news", text=text, source=source, at=THU_8)

    a = item(
        "Barclays analyst Dan Levy expects a Tesla-SpaceX merger before Tesla's meeting",
        "247wallst.com",
    )
    b = item(
        "Barclays (Dan Levy) expects a Tesla-SpaceX merger announcement before the meeting",
        "yahoo.com",
    )
    c = item("319 million insider shares become sellable on Oct 26", "fool.com")
    kept = M.one_per_story(
        [a, b, c, item("Same Barclays Levy Tesla-SpaceX merger note", "247wallst.com")]
    )
    assert [k.source for k in kept] == ["247wallst.com", "fool.com"]
    assert kept[0].also == ["yahoo.com"]  # the same outlet again is not "also"


def test_two_stories_about_the_same_company_stay_two():
    # META on 2026-10-15: an ads-moat analysis and a model release share Meta, OpenAI, Anthropic.
    ads = M.Item(kind="news", source="yahoo.com", at=THU_8, text=(
        "A Yahoo Finance analysis argues Meta's ads business gives it a monetization advantage "
        "OpenAI and Anthropic lack, citing Bank of America analysts"))  # fmt: skip
    muse = M.Item(kind="news", source="aimagazine.com", at=THU_8, text=(
        "Meta released Muse Wonder; the Full version scores 56 on the Artificial Analysis Index, "
        "beating OpenAI's GPT-6 Astra and trailing Anthropic's Claude"))  # fmt: skip
    assert len(M.one_per_story([ads, muse])) == 2


class TestDayMove:
    def test_from_the_stored_bars(self):
        move, on = M.day_move([("2026-10-13", 100.0), ("2026-10-14", 90.0)], {})
        assert move == pytest.approx(-0.10) and on == date(2026, 10, 14)

    def test_the_book_when_no_bars(self):
        move, on = M.day_move([], {"close": 100.0, "price": 104.0, "cost": 80.0})
        assert move == pytest.approx(0.04) and on is None

    def test_a_close_equal_to_the_fill_is_not_a_day_move(self):
        # SPCX 2026-10-15: shares added that day; the broker's close_price was the fill.
        assert M.day_move([], {"close": 137.93, "price": 146.75, "cost": 137.93}) == (None, None)

    def test_one_bar_is_not_a_move(self):
        assert M.day_move([("2026-10-14", 90.0)], {}) == (None, None)


def test_results_date():
    assert M.results_date(["x", "next results 2026-11-03 (in 13 sessions)"]) == (
        date(2026, 11, 3),
        13,
    )
    assert M.results_date(["next results 2026-10-16 (in 1 session)"]) == (date(2026, 10, 16), 1)
    assert M.results_date(["no date here"]) == (None, None)


class TestBuild:
    def build(self, **kw):
        book = BookSnapshot(positions=[pos("SPCX", qty=19, price=150.0, close=137.93, cost=137.93),
                                       pos("CCXI", qty=20, account="A"),
                                       pos("CCXI", qty=15, account="B"),
                                       pos("CCXI", qty=1, instrument=EQUITY_OPTION)],
                            net_liq=10_000.0, as_of=THU_8)  # fmt: skip
        args = dict(proposals={"SPCX": prop("SPCX")}, actionables={"_all": ["SELL CCXI"],
                    "CCXI": "SELL CCXI — all 35 shares"}, judgments={"SPCX": [J("SPCX", 3)],
                    "META": [J("META", 2, "MEDIUM", what="probe")]}, filings={},
                    summaries={}, watched=["SPCX", "META", "AMZN"], now=THU_8)  # fmt: skip
        args.update(kw)
        return M.build(book, **args)

    def test_one_card_per_holding_largest_first_accounts_summed_options_out(self):
        m = self.build()
        assert [c.symbol for c in m.cards] == ["CCXI", "SPCX"]  # 35×110 > 19×150
        ccxi = m.cards[0]
        assert ccxi.shares == 35 and ccxi.action == "SELL CCXI — all 35 shares"
        assert ccxi.weight == pytest.approx(35 * 110 / 10_000)

    def test_the_call_its_reason_and_results(self):
        spcx = self.build().cards[1]
        assert spcx.call == "HOLD" and spcx.results == date(2026, 10, 21) and spcx.results_in == 4
        assert spcx.day_move is None  # close == fill, no bars
        assert [i.text for i in spcx.new] == ["a fact"]

    def test_at_most_three_new_items_and_the_rest_counted(self):
        js = [J("SPCX", h, "HIGH", what=f"n{h}") for h in (1, 2, 3, 4, 5)]
        spcx = self.build(judgments={"SPCX": js}).cards[1]
        assert len(spcx.new) == 3 and spcx.more == 2
        assert [i.text for i in spcx.new] == ["n1", "n2", "n3"]  # newest first

    def test_watched_names_only_with_news_and_never_a_holding(self):
        m = self.build()
        assert [(w.symbol, w.item.text) for w in m.watched] == [("META", "probe")]

    def test_an_answered_action_still_in_the_book_is_said(self):
        m = self.build(answered={"CCXI": "SELL CCXI — you acted on this on 2026-10-15"})
        assert m.cards[0].answered.endswith("2026-10-15")

    def test_no_book_is_an_empty_read(self):
        m = M.build(None, {}, {}, {}, {}, {}, [], THU_8)
        assert m.cards == [] and m.net_liq is None and m.watched == []


def test_the_endpoint_reads_the_stores(tmp_path, monkeypatch):
    from advisor.api.app import create_app
    from advisor.api.routers import morning as router
    from advisor.daemon.store import DaemonStore
    from fastapi.testclient import TestClient

    db = tmp_path / "research.db"
    store = DaemonStore(db)
    store.save_book(BookSnapshot(positions=[pos("CRDO")], net_liq=5_000.0, as_of=THU_8))
    store.close()
    monkeypatch.setattr(router, "_db", lambda: db)
    monkeypatch.setattr("advisor.daemon.market_calendar.now_et", lambda: THU_8)
    with TestClient(create_app()) as c:
        body = c.get("/api/morning").json()
    assert [x["symbol"] for x in body["cards"]] == ["CRDO"]
    assert body["since"].startswith("2026-10-14T08:00")
