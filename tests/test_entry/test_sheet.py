"""The daily sheet, the research universe, and the watchlist tier cap."""

from __future__ import annotations

import asyncio
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import pytest
from advisor.daemon import market_calendar as mc
from advisor.daemon.book import BookSnapshot, Position
from advisor.daemon.models import Event, EventSource, EventTier
from advisor.daemon.store import DaemonStore
from advisor.daemon.universe import research_symbols, watched_universe
from advisor.entry.sheet import build_sheet, move_from_closes

NOW = datetime(2026, 9, 25, 11, 0, tzinfo=mc.MARKET_TZ)


@pytest.fixture
def store(tmp_path: Path):
    s = DaemonStore(tmp_path / "research.db")
    yield s
    s.close()


def closes(n=80, last=None):
    days = [date(2026, 9, 25) - timedelta(days=n - 1 - i) for i in range(n)]
    px = [100.0 * (1.01 if i % 2 else 0.99) for i in range(n)]
    if last is not None:
        px[-1] = last
    return list(zip(days, px))


class TestMove:
    def test_sigma_excludes_today(self):
        base = closes()
        huge = base[:-1] + [(base[-1][0], base[-2][1] * 1.5)]
        calm = move_from_closes(base, date(2026, 9, 25))
        shock = move_from_closes(huge, date(2026, 9, 25))
        assert shock.sigma == pytest.approx(calm.sigma)
        assert shock.z > 10

    def test_nan_and_zero_rows_dropped(self):
        rows = closes() + [(date(2026, 9, 26), float("nan")), (date(2026, 9, 27), 0.0)]
        assert move_from_closes(rows, date(2026, 9, 25)).asof == date(2026, 9, 25)

    def test_short_history(self):
        m = move_from_closes(closes(10), date(2026, 9, 25))
        assert m.sigma is None and m.z is None and m.d20 is None and m.d5 is not None
        assert move_from_closes(closes(1), date(2026, 9, 25)) is None


def sheet(store, **kw):
    defaults = dict(
        closes=lambda s: closes(),
        consensus_loader=lambda st, s: None,
        series_loader=lambda s: None,
        margin_loader=lambda s: None,
    )
    defaults.update(kw)
    return build_sheet(store, "amzn", NOW, **defaults)


class TestSheet:
    def test_every_gap_is_named(self, store):
        s = sheet(store)
        text = " ".join(s.gaps)
        assert "no entry zone" in text and "no valuation" in text and "no factor estimate" in text
        assert s.symbol == "AMZN"

    def test_no_prices_is_a_gap_not_a_crash(self, store):
        s = sheet(store, closes=lambda sym: [])
        assert s.move is None and "no price history" in s.gaps

    def test_events_since_the_previous_close(self, store):
        store.emit(
            Event(
                source=EventSource.EDGAR,
                kind="FILING_RESULTS",
                tier=EventTier.B,
                symbol="AMZN",
                ts=datetime(2026, 9, 24, 16, 30, tzinfo=mc.MARKET_TZ),
                dedup_key="a",
            )
        )
        store.emit(
            Event(
                source=EventSource.EDGAR,
                kind="FILING_RESULTS",
                tier=EventTier.B,
                symbol="AMZN",
                ts=datetime(2026, 9, 24, 15, 0, tzinfo=mc.MARKET_TZ),
                dedup_key="b",
            )
        )
        s = sheet(store)
        assert len(s.events_today) == 1 and s.events_week == 2 and s.changed

    def test_quiet_day_is_not_changed(self, store):
        assert not sheet(store).changed

    def test_two_sigma_move_is_changed(self, store):
        base = closes()
        big = base[:-1] + [(base[-1][0], base[-2][1] * 1.2)]
        assert sheet(store, closes=lambda s: big).changed

    def test_scanner_candidate_is_changed(self, store, tmp_path):
        from advisor.scanner.models import Candidate, Setup
        from advisor.scanner.store import ScannerStore

        sc = ScannerStore(tmp_path / "research.db")
        sc.add(
            Candidate(
                session=NOW.date(),
                setup=Setup.NEWS_DIP,
                symbol="AMZN",
                detected_at=NOW,
                price=100,
                prev_close=106,
                change=-0.06,
            )
        )
        s = sheet(store, scanner_store=sc)
        sc.close()
        assert s.candidates == ["2026-09-25:C:AMZN"] and s.changed

    def test_holding_weight(self, store):
        book = BookSnapshot(
            net_liq=10_000,
            positions=[
                Position(
                    account="a",
                    symbol="AMZN",
                    underlying="AMZN",
                    instrument="Equity",
                    quantity=4,
                    avg_open_price=200,
                    close_price=250,
                )
            ],
        )
        store.save_book(book)
        s = sheet(store)
        assert s.holding.weight == pytest.approx(0.10)
        assert s.holding.unrealized == pytest.approx(0.25)


class TestThesisStatus:
    """A rule broken ten days ago still counts, unless the user answered it."""

    def setup(self, store, *, days_ago=19, pct=0.067):
        from advisor.thesis.models import Claim, ClaimKind, Comparator, Trigger

        store.save_book(BookSnapshot(net_liq=10_000, positions=[pos("AAOI", 10, 100)]))
        claim = Claim(
            kind=ClaimKind.INVALIDATION,
            text="Any equity raise above 5% of market cap breaks the story",
            trigger=Trigger(
                event_kinds=["FILING_DILUTION"],
                field="dilution_pct",
                comparator=Comparator.ABOVE,
                threshold=0.05,
            ),
        )
        store.save_claim("AAOI", claim)
        store.emit(
            Event(
                source=EventSource.EDGAR,
                kind="FILING_DILUTION",
                tier=EventTier.A,
                symbol="AAOI",
                ts=NOW - timedelta(days=days_ago),
                payload={"dilution_pct": pct},
                dedup_key="atm",
            )
        )
        return claim

    def test_old_break_the_weekly_card_calls_untested_is_broken(self, store):
        from advisor.entry.sheet import thesis_status

        self.setup(store)
        status, rules = thesis_status(store, "AAOI", store.load_latest_book(), NOW)
        assert status == "broken" and "5%" in rules[0]

    def test_under_the_threshold_is_intact(self, store):
        from advisor.entry.sheet import thesis_status

        self.setup(store, pct=0.03)
        assert thesis_status(store, "AAOI", store.load_latest_book(), NOW) == ("intact", [])

    def test_beyond_ninety_days_no_longer_counts(self, store):
        from advisor.entry.sheet import thesis_status

        self.setup(store, days_ago=120)
        assert thesis_status(store, "AAOI", store.load_latest_book(), NOW)[0] == "intact"

    def test_answered_break_is_not_a_blocker(self, store):
        from advisor.action.decisions import Decision, SubjectKind, Verdict
        from advisor.entry.sheet import thesis_status

        claim = self.setup(store)
        store.record_decision(
            Decision(
                symbol="AAOI",
                subject_kind=SubjectKind.CLAIM,
                subject_id=claim.id,
                verdict=Verdict.ACKNOWLEDGED,
            )
        )
        status, _ = thesis_status(store, "AAOI", store.load_latest_book(), NOW)
        assert status == "answered"

    def test_no_thesis(self, store):
        from advisor.entry.sheet import thesis_status

        store.save_book(BookSnapshot(net_liq=10_000))
        assert thesis_status(store, "MSFT", store.load_latest_book(), NOW) == (None, [])


def pos(sym, qty, px):
    return Position(
        account="a", symbol=sym, underlying=sym, instrument="Equity", quantity=qty, close_price=px
    )


class TestUniverse:
    book = BookSnapshot(net_liq=10_000, positions=[pos("TE", 50, 4), pos("SPCX", 11, 150)])

    def test_held_first_largest_first_then_watchlist(self):
        syms, errors = research_symbols(self.book, fetch_watchlist=lambda n: (["AMZN", "TE"], None))
        assert syms[:2] == ["SPCX", "TE"] and "AMZN" in syms and syms.count("TE") == 1
        assert errors == []

    def test_watchlist_down_is_reported_and_degrades(self):
        syms, errors = research_symbols(
            self.book, fetch_watchlist=lambda n: ([], "watchlist 'Swing' unavailable: 401")
        )
        assert syms[:2] == ["SPCX", "TE"] and "401" in errors[0]

    def test_reasons_are_kept(self):
        uni, _ = watched_universe(self.book, fetch_watchlist=lambda n: (["AMZN"], None))
        assert uni.symbols["AMZN"].is_held is False and uni.symbols["TE"].is_held


class TestWatchlistFilingsDoNotInterrupt:
    """A dilution 8-K on a watched name reaches the digest (B), not an interrupt (A)."""

    def run(self, store, monkeypatch, symbols):
        import advisor.news.edgar as edgar
        import advisor.news.ingest as mod
        from advisor.news.models import EntityMatch, MatchMethod, SourceItem, SourceTier

        def filing(symbol, **kw):
            return [
                SourceItem(
                    tier=SourceTier.PRIMARY,
                    provider="SEC EDGAR",
                    url=f"https://sec/{symbol}",
                    title="424B5 prospectus",
                    published_at=datetime.now(timezone.utc),
                    entity=EntityMatch(symbol=symbol, cik=1, method=MatchMethod.CIK),
                    doc_type="424B5",
                    accession=f"acc-{symbol}",
                )
            ]

        async def caps(_):
            return {}

        monkeypatch.setattr(edgar, "recent_filings", filing)
        monkeypatch.setattr(mod, "_market_caps", caps)
        monkeypatch.setattr(mod, "offering_size_for", lambda _: None)
        book = BookSnapshot(positions=[pos("AAOI", 10, 100)], net_liq=10_000)
        return asyncio.run(mod.ingest_filings(store, book, symbols=symbols))

    def test_held_name_still_interrupts(self, store, monkeypatch):
        result = self.run(store, monkeypatch, ["AAOI"])
        assert [e.tier for e in result.events] == [EventTier.A]

    def test_watched_name_capped_at_b(self, store, monkeypatch):
        result = self.run(store, monkeypatch, ["AAOI", "AMZN"])
        tiers = {e.symbol: e.tier for e in result.events}
        assert tiers == {"AAOI": EventTier.A, "AMZN": EventTier.B}
        assert result.interrupts == 1
