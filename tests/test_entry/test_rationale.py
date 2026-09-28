"""Every acting proposal carries its full rationale: the why, the events by name, the news.

User decision (2026-09-28): "lo único que quiero es que el sistema agregue el
racional, si es add es add". The action is never changed here.
"""

from __future__ import annotations

import sqlite3
from datetime import timedelta
from types import SimpleNamespace

from advisor.daemon.store import DaemonStore
from advisor.entry import run
from advisor.entry.proposal import Action, build_proposal
from advisor.entry.sheet import EventLine

from tests.test_entry.test_proposal import ENTERED, NET_LIQ, NOW, OUT, mk


def events(n):
    return [
        EventLine(ts=NOW - timedelta(hours=k), kind="FILING_OFFICER_CHANGE", tier="B",
                  text=f"8-K: director or principal officer change {k}", items=["5.02"])
        for k in range(n)
    ]  # fmt: skip


def test_the_why_comes_first_and_the_action_is_unchanged():
    sheet = mk(ENTERED, OUT)
    before = build_proposal(sheet.model_copy(update={"events_today": []}), net_liq=NET_LIQ)
    p = build_proposal(sheet, net_liq=NET_LIQ)
    assert p.action is Action.ENTER and before.action is Action.ENTER
    assert p.triggers and p.reasons[0].text == f"Why ENTER: {p.triggers[0]}"
    assert p.reasons[0].source == "entry rules"


def test_events_are_named_with_their_source_not_counted():
    p = build_proposal(mk(ENTERED, OUT).model_copy(update={"events_today": events(2)}),
                       net_liq=NET_LIQ)  # fmt: skip
    named = [r for r in p.reasons if "officer change" in r.text]
    assert len(named) == 2 and all(r.source == "SEC EDGAR 8-K item 5.02" for r in named)
    assert "ET: 8-K" in named[0].text
    assert not any("event(s) since the previous close" in r.text for r in p.reasons)


def test_many_events_are_named_up_to_a_limit_then_counted():
    from advisor.entry.proposal import EVENTS_NAMED

    p = build_proposal(mk(ENTERED, OUT).model_copy(update={"events_today": events(6)}),
                       net_liq=NET_LIQ)  # fmt: skip
    assert sum("officer change" in r.text for r in p.reasons) == EVENTS_NAMED
    assert any(r.text.startswith(f"and {6 - EVENTS_NAMED} more event(s)") for r in p.reasons)


# ── news in the rationale ─────────────────────────────────────────────────


def judgment(title, materiality="HIGH", direction="NEGATIVE", about=True, days_ago=0):
    from advisor.news.judge import Judgment

    return Judgment(
        key=title, symbol="MDB", published_at=NOW - timedelta(days=days_ago), title=title,
        provider="reuters.com", tier="AGGREGATOR", url="https://r/x", about_company=about,
        event_type="MANAGEMENT", direction=direction, materiality=materiality, novelty="NEW",
        basis="REPORTED", quote="q", why="why", what="CEO leaves for Meta",
        prompt_version="v", judged_at=NOW,
    )  # fmt: skip


def store_judgments(db, rows):
    from advisor.news.judge import NewsJudgmentStore

    conn = sqlite3.connect(db)
    js = NewsJudgmentStore(conn)
    for j in rows:
        js.add(j)
    conn.commit()
    conn.close()


def test_judged_news_is_cited_most_material_first(tmp_path):
    db = tmp_path / "research.db"
    store_judgments(db, [judgment("minor", "LOW", "POSITIVE"), judgment("CEO leaves"),
                         judgment("unrelated", about=False)])  # fmt: skip
    daemon = DaemonStore(db)
    reasons = run.news_rationale(daemon, "MDB", NOW, puller=lambda *a: pytest_fail())
    daemon.close()
    assert [r.text.split(": ", 1)[1].split(" —")[0] for r in reasons] == ["CEO leaves", "minor"]
    assert reasons[0].text.startswith(f"News {NOW.date().isoformat()}, negative high")
    assert "reuters.com https://r/x" in reasons[0].source


def pytest_fail():
    raise AssertionError("must not pull: the week's news is already judged")


def test_no_news_pulls_once_then_says_so(tmp_path):
    run._NEWS_PULLED.clear()
    daemon = DaemonStore(tmp_path / "research.db")
    pulls = []

    def puller(db, sym, now):
        pulls.append(sym)
        return {"pulled": 5, "judged": 0}

    first = run.news_rationale(daemon, "MDB", NOW, puller=puller)
    again = run.news_rationale(daemon, "MDB", NOW, puller=puller)
    daemon.close()
    assert pulls == ["MDB"]  # once per name per session
    assert "no news about MDB judged in the last 7 days (5 item(s) found" in first[0].text
    assert again[0].text.startswith("no news about MDB")


def test_a_failed_pull_is_said(tmp_path):
    run._NEWS_PULLED.clear()
    daemon = DaemonStore(tmp_path / "research.db")

    def boom(*a):
        raise TimeoutError("tavily")

    reasons = run.news_rationale(daemon, "MDB", NOW, puller=boom)
    daemon.close()
    assert reasons[0].text == "news not read: tavily"


def test_acting_proposals_carry_the_news_quiet_ones_are_not_searched(tmp_path):
    from advisor.daemon.book import BookSnapshot

    daemon = DaemonStore(tmp_path / "research.db")
    daemon.save_book(BookSnapshot(net_liq=NET_LIQ))
    from advisor.entry.proposal import Reason

    from tests.test_entry.test_proposal import IN

    sheets = {"QUIET": mk(IN, IN), "GO": mk(ENTERED, OUT)}
    asked = []

    def news(store, sym, now):
        asked.append(sym)
        return [Reason(text="News: CEO leaves", source="reuters.com (news agent)")]

    def reader(sym):
        return SimpleNamespace(status=SimpleNamespace(value="OK"),
                               stance=SimpleNamespace(value="NEUTRAL"), sentences=[])  # fmt: skip

    proposals, errors = run.propose_all(
        daemon, NOW, symbols=["QUIET", "GO"], reader=reader, news=news,
        sheet_builder=lambda st, sym, n, scanner_store=None: sheets[sym].model_copy(
            update={"symbol": sym}
        ),
    )  # fmt: skip
    daemon.close()
    go = next(p for p in proposals if p.symbol == "GO")
    assert asked == ["GO"] and errors == []
    assert go.action is Action.ENTER and go.reasons[-1].text == "News: CEO leaves"
    assert go.reasons[0].text.startswith("Why ENTER")
