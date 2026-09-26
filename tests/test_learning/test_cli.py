"""``advisor learn rules``: what versions exist, what each produced, which one runs."""

from __future__ import annotations

import io
import json
from datetime import date, datetime

import pytest
from advisor.cli import formatters, learn_cmds
from advisor.cli.app import app
from advisor.daemon import market_calendar as mc
from advisor.learning.rules import PRE_REGISTRY, Kind, stamp
from advisor.research import config
from advisor.scanner.models import Candidate, Setup
from advisor.scanner.ruleset import session_rules
from advisor.scanner.store import ScannerStore
from typer.testing import CliRunner

NOW = datetime(2026, 9, 25, 10, 5, tzinfo=mc.MARKET_TZ)
runner = CliRunner()


@pytest.fixture
def db(tmp_path, monkeypatch):
    path = tmp_path / "research.db"
    monkeypatch.setenv("ADVISOR_RESEARCH_DB_PATH", str(path))
    config.get_settings.cache_clear()
    yield path
    config.get_settings.cache_clear()


def cand(symbol, rules):
    return Candidate(
        session=date(2026, 9, 25),
        setup=Setup.NEWS_DIP,
        symbol=symbol,
        detected_at=NOW,
        price=94.0,
        prev_close=100.0,
        change=-0.06,
        rules=rules,
    )


def run(*args):
    return runner.invoke(app, ["learn", "rules", *args])


def as_json(monkeypatch, *args):
    # output_json binds sys.stdout when it is defined, so neither CliRunner
    # nor capsys sees it. The real serialiser, into a buffer.
    buf = io.StringIO()
    monkeypatch.setattr(
        learn_cmds, "output_json", lambda data, file=None: formatters.output_json(data, file=buf)
    )
    r = run(*args, "--output", "json")
    assert r.exit_code == 0, r.output
    return json.loads(buf.getvalue())


def test_empty_database(db, monkeypatch):
    data = as_json(monkeypatch)
    assert data["versions"] == [] and data["pre_registry"] == {}
    assert set(data["not_yet_recorded"]) == {"scanner.session", "scanner.premarket", "entry"}


def test_versions_counts_and_which_runs_now(db, monkeypatch):
    old = stamp("scanner.session", {"thresholds.sigma_min": (1.0, Kind.THRESHOLD)})
    s = ScannerStore(db)
    s.add(cand("AAPL", session_rules()))
    s.add(cand("NVDA", session_rules()))
    s.add(cand("INTC", old))
    s.add(cand("PRE", None))
    s.close()

    data = as_json(monkeypatch)
    by = {v["version"]: v for v in data["versions"]}
    assert by[session_rules().version]["current"] is True
    assert by[session_rules().version]["records"] == {"candidates": 2}
    assert by[old.version]["current"] is False
    assert by[old.version]["records"] == {"candidates": 1}
    assert data["pre_registry"] == {"candidates": 1}
    assert "scanner.session" not in data["not_yet_recorded"]


def test_table_output(db):
    s = ScannerStore(db)
    s.add(cand("AAPL", session_rules()))
    s.add(cand("PRE", None))
    s.close()
    r = run()
    assert r.exit_code == 0, r.output
    assert session_rules().version in r.output
    assert PRE_REGISTRY in r.output


def test_show_one_version_and_what_differs(db, monkeypatch):
    old = stamp(
        "scanner.session",
        {
            **{k: (v, Kind.THRESHOLD) for k, v in session_rules().params.items()},
            "thresholds.sigma_min": (1.0, Kind.THRESHOLD),
        },
    )
    s = ScannerStore(db)
    s.add(cand("INTC", old))
    s.close()
    data = as_json(monkeypatch, old.version[:8])
    assert data["version"] == old.version
    assert data["params"]["thresholds.sigma_min"] == 1.0
    assert data["differs_from_running"] == ["thresholds.sigma_min"]


def test_unknown_version_is_an_error(db):
    r = run("deadbeef", "--output", "json")
    assert r.exit_code == 1
