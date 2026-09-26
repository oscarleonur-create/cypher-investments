"""d60/d120 on proposals without a download storm; readings stamped with their prompt."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest
from advisor.daemon import market_calendar as mc
from advisor.daemon.store import DaemonStore
from advisor.entry.proposal import build_proposal
from advisor.entry.sheet import Move, Sheet
from advisor.entry.store import EntryStore
from advisor.entry.track import Bar, due, fill_proposal_outcomes, score
from advisor.scanner.outcomes import nth_trading_day
from advisor.story import reading as reading_mod
from advisor.story.reading import PROMPT_VERSION, Reading, ReadingStatus, Stance, read_symbol

SESSION = date(2026, 3, 2)
BUILT = datetime(2026, 3, 2, 11, 0, tzinfo=mc.MARKET_TZ)


def sheet(symbol="AMZN"):
    return Sheet(
        symbol=symbol,
        built_at=BUILT,
        move=Move(price=100.0, asof=SESSION, day=0.0, sigma=0.02, z=0.0),
    )


def bars_through(n: int, px=lambda i: 100.0 + i):
    out, day = [], SESSION
    for i in range(n + 1):
        out.append(Bar(day, px(i) + 1, px(i) - 1, px(i)))
        day = nth_trading_day(day, 1)
    return out


def closed_after(n: int) -> datetime:
    day = nth_trading_day(SESSION, n)
    return datetime.combine(day, mc.session_close(day), tzinfo=mc.MARKET_TZ) + timedelta(hours=1)


class TestScore:
    def test_d60_and_d120_once_knowable(self):
        p = build_proposal(sheet(), net_liq=10_000)
        out = score(p, bars_through(120), closed_after(120))
        assert out["d60"] == pytest.approx(160 / 100 - 1)
        assert out["d120"] == pytest.approx(220 / 100 - 1)
        assert out["mae120"] == pytest.approx((101 - 1) / 100 - 1)

    def test_not_before_their_session_closes(self):
        p = build_proposal(sheet(), net_liq=10_000)
        out = score(p, bars_through(120), closed_after(59))
        assert "d20" in out and "d60" not in out and "d120" not in out


class TestDue:
    def test_nothing_missing_is_not_due(self):
        p = build_proposal(sheet(), net_liq=10_000)
        p.outcomes = {
            k: 0.0 for k in ("next_close", "d5", "d10", "d20", "mae20", "d60", "d120", "mae120")
        }
        assert not due(p, closed_after(200))

    def test_waiting_on_d60_is_not_due_on_day_30(self):
        p = build_proposal(sheet(), net_liq=10_000)
        p.outcomes = {k: 0.0 for k in ("next_close", "d5", "d10", "d20", "mae20")}
        assert not due(p, closed_after(30))
        assert due(p, closed_after(60))


class CountingBars:
    def __init__(self):
        self.calls = []

    def __call__(self, symbol, start, end):
        self.calls.append(symbol)
        return bars_through(130)


class TestFill:
    @pytest.fixture
    def store(self, tmp_path):
        s = EntryStore(tmp_path / "research.db")
        yield s
        s.close()

    def test_one_download_per_symbol_whatever_the_proposal_count(self, store):
        for action_day in range(3):
            s = sheet()
            s.built_at = BUILT + timedelta(days=action_day)
            store.add(build_proposal(s, net_liq=10_000))
        store.add(build_proposal(sheet("META"), net_liq=10_000))
        bars = CountingBars()
        fill_proposal_outcomes(store, closed_after(125), bars_fn=bars)
        assert sorted(bars.calls) == ["AMZN", "META"]

    def test_no_download_while_nothing_new_is_knowable(self, store):
        store.add(build_proposal(sheet(), net_liq=10_000))
        bars = CountingBars()
        fill_proposal_outcomes(store, closed_after(25), bars_fn=bars)
        assert bars.calls == ["AMZN"]
        fill_proposal_outcomes(store, closed_after(30), bars_fn=bars)
        assert bars.calls == ["AMZN"]  # d60 is not knowable yet: no second download
        r = fill_proposal_outcomes(store, closed_after(61), bars_fn=bars)
        assert bars.calls == ["AMZN", "AMZN"] and r.updated == 1

    def test_old_proposals_are_still_tracked_to_d120(self, store):
        store.add(build_proposal(sheet(), net_liq=10_000))
        fill_proposal_outcomes(store, closed_after(121), bars_fn=CountingBars())
        (p,) = store.list()
        assert "d120" in p.outcomes


# ── Readings ─────────────────────────────────────────────────────────────


class TestPromptVersion:
    def test_stable_within_a_process(self):
        assert reading_mod._prompt_version() == PROMPT_VERSION and len(PROMPT_VERSION) == 12

    def test_changes_with_the_system_prompt(self, monkeypatch):
        monkeypatch.setattr(reading_mod, "SYSTEM_PROMPT", reading_mod.SYSTEM_PROMPT + " ")
        assert reading_mod._prompt_version() != PROMPT_VERSION

    def test_proposal_carries_model_and_prompt_of_its_stance(self):
        r = Reading(
            symbol="AMZN",
            status=ReadingStatus.OK,
            stance=Stance.CAUTIOUS,
            model="m-1",
            prompt_version="abc123abc123",
        )
        p = build_proposal(sheet(), net_liq=10_000, reading=r)
        assert (p.stance, p.reading_model, p.reading_prompt) == ("CAUTIOUS", "m-1", "abc123abc123")

    def test_no_reading_no_stamp(self):
        p = build_proposal(sheet(), net_liq=10_000)
        assert p.reading_model is None and p.reading_prompt is None


class TestCacheByPrompt:
    @pytest.fixture
    def store(self, tmp_path):
        s = DaemonStore(tmp_path / "research.db")
        yield s
        s.close()

    def test_a_reading_from_another_prompt_is_not_served(self, store, monkeypatch):
        from tests.test_story.test_reading import GOOD, FakeModel, filing

        filing(store)
        first = read_symbol(store, "AAOI", complete=FakeModel(GOOD), model="fake")
        assert first.prompt_version == PROMPT_VERSION
        # Same facts, same prompt: served from cache (the empty model would raise).
        assert read_symbol(store, "AAOI", complete=FakeModel()).status is ReadingStatus.OK
        # The prompt changes: the cached reading is not this prompt's reading.
        monkeypatch.setattr(reading_mod, "PROMPT_VERSION", "000000000000")
        model = FakeModel(GOOD)
        again = read_symbol(store, "AAOI", complete=model, model="fake")
        assert len(model.prompts) == 1 and again.prompt_version == "000000000000"
