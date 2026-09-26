"""rule_versions: written once per version, alongside the first record that carries it."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import date, datetime

import pytest
from advisor.daemon import market_calendar as mc
from advisor.learning import store as rule_versions
from advisor.learning.rules import PRE_REGISTRY, Kind, RuleStamp, stamp
from advisor.learning.store import RuleStore
from advisor.scanner.models import Candidate, Setup
from advisor.scanner.store import ScannerStore

NOW = datetime(2026, 9, 25, 10, 5, tzinfo=mc.MARKET_TZ)
S1 = stamp("scanner.session", {"thresholds.sigma_min": (2.0, Kind.THRESHOLD)})
S2 = stamp("scanner.session", {"thresholds.sigma_min": (2.5, Kind.THRESHOLD)})


def cand(symbol="AAPL", setup=Setup.NEWS_DIP, rules=S1, session=date(2026, 9, 25)):
    return Candidate(
        session=session,
        setup=setup,
        symbol=symbol,
        detected_at=NOW,
        price=94.0,
        prev_close=100.0,
        change=-0.06,
        rules=rules,
    )


@pytest.fixture
def conn(tmp_path):
    c = sqlite3.connect(str(tmp_path / "research.db"))
    rule_versions.ensure_schema(c)
    yield c
    c.close()


class TestRegister:
    def test_first_time_is_new_second_is_not(self, conn):
        assert rule_versions.register(conn, S1)
        assert not rule_versions.register(conn, S1)
        conn.commit()
        assert conn.execute("SELECT COUNT(*) FROM rule_versions").fetchone()[0] == 1

    def test_parameters_and_kinds_are_stored(self, conn):
        rule_versions.register(conn, S1)
        conn.commit()
        v = RuleStore(conn).get(S1.version)
        assert v.params == {"thresholds.sigma_min": 2.0}
        assert v.kinds == {"thresholds.sigma_min": Kind.THRESHOLD}
        assert v.first_code == S1.code and v.ruleset == "scanner.session"

    def test_first_seen_is_tz_aware_eastern(self, conn):
        # SQLite's datetime('now') is naive UTC: 02:13 "tomorrow" for a
        # 22:13 ET Friday run. The first live run showed exactly that.
        rule_versions.register(conn, S1)
        conn.commit()
        seen = RuleStore(conn).get(S1.version).first_seen_at
        assert seen.tzinfo is not None
        assert seen.utcoffset() == mc.now_et().utcoffset()

    def test_none_is_skipped(self, conn):
        assert not rule_versions.register(conn, None)

    def test_a_stamp_read_back_from_a_row_is_not_registered_empty(self, conn):
        back = RuleStamp.model_validate_json(S1.model_dump_json())
        assert not rule_versions.register(conn, back)
        conn.commit()
        assert conn.execute("SELECT COUNT(*) FROM rule_versions").fetchone()[0] == 0

    def test_does_not_commit_on_its_own(self, tmp_path):
        path = tmp_path / "r.db"
        a = sqlite3.connect(str(path))
        rule_versions.ensure_schema(a)
        rule_versions.register(a, S1)
        b = sqlite3.connect(str(path))
        assert b.execute("SELECT COUNT(*) FROM rule_versions").fetchone()[0] == 0
        a.commit()
        assert b.execute("SELECT COUNT(*) FROM rule_versions").fetchone()[0] == 1
        a.close()
        b.close()


class TestThroughTheScannerStore:
    def test_new_candidate_registers_its_version(self, tmp_path):
        s = ScannerStore(tmp_path / "research.db")
        assert s.add(cand())
        versions = RuleStore(s._conn).list()
        assert [v.version for v in versions] == [S1.version]
        s.close()

    def test_same_candidate_twice_one_version_row(self, tmp_path):
        s = ScannerStore(tmp_path / "research.db")
        s.add(cand())
        assert not s.add(cand())
        assert len(RuleStore(s._conn).list()) == 1
        s.close()

    def test_rules_changed_mid_session_are_two_versions(self, tmp_path):
        s = ScannerStore(tmp_path / "research.db")
        s.add(cand("AAPL", rules=S1))
        s.add(cand("NVDA", rules=S2))
        counts = RuleStore(s._conn).record_counts()["candidates"]
        assert counts == {S1.version: 1, S2.version: 1}
        s.close()

    def test_a_candidate_loaded_and_readded_does_not_register_empty(self, tmp_path):
        s = ScannerStore(tmp_path / "research.db")
        s.add(cand())
        again = s.get(cand().id)
        assert again.rules.version == S1.version and again.rules.params == {}
        s._conn.execute("DELETE FROM scan_candidates")
        s._conn.execute("DELETE FROM rule_versions")
        s._conn.commit()
        assert s.add(again)  # the row comes back...
        assert RuleStore(s._conn).list() == []  # ...without a parameterless version
        s.close()

    def test_two_writers_racing_leave_one_version(self, tmp_path):
        path = tmp_path / "research.db"
        ScannerStore(path).close()
        errors = []

        def write(sym):
            try:
                st = ScannerStore(path)
                st.add(cand(sym))
                st.close()
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=write, args=(f"S{i}",)) for i in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        conn = sqlite3.connect(str(path))
        assert conn.execute("SELECT COUNT(*) FROM rule_versions").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM scan_candidates").fetchone()[0] == 8
        conn.close()


class TestRecordCounts:
    def test_rows_before_stamps_are_pre_registry(self, tmp_path):
        s = ScannerStore(tmp_path / "research.db")
        old = cand("OLD", rules=None)
        # As written before this change: no "rules" or "origin" in the payload.
        payload = json.loads(old.model_dump_json())
        payload.pop("rules"), payload.pop("origin"), payload.pop("volume")
        s._conn.execute(
            "INSERT INTO scan_candidates (id, session, setup, symbol, detected_at, payload_json) "
            "VALUES (?, ?, ?, ?, ?, ?)",
            (old.id, "2026-09-25", "C", "OLD", NOW.isoformat(), json.dumps(payload)),
        )
        s._conn.commit()
        s.add(cand("NEW"))
        loaded = s.get(old.id)
        assert loaded.rules is None and loaded.origin == "live"
        counts = RuleStore(s._conn).record_counts()["candidates"]
        assert counts == {PRE_REGISTRY: 1, S1.version: 1}
        s.close()

    def test_ledgers_that_do_not_exist_are_left_out(self, conn):
        assert RuleStore(conn).record_counts() == {}


class TestGet:
    def test_by_prefix(self, conn):
        rule_versions.register(conn, S1)
        conn.commit()
        assert RuleStore(conn).get(S1.version[:6]).version == S1.version

    def test_ambiguous_or_unknown_prefix_is_none(self, conn):
        rule_versions.register(conn, S1)
        rule_versions.register(conn, S2)
        conn.commit()
        assert RuleStore(conn).get("") is None
        assert RuleStore(conn).get("zzzz") is None

    def test_list_by_ruleset(self, conn):
        rule_versions.register(conn, S1)
        rule_versions.register(conn, stamp("entry", {"x": (1, Kind.MODEL)}))
        conn.commit()
        assert [v.ruleset for v in RuleStore(conn).list("entry")] == ["entry"]
