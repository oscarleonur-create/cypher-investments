"""Depth on demand: the news agent for any name, the report job, and the status read."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta

import pytest
from advisor.api import deps
from advisor.api.routers import depth as D
from advisor.daemon.market_calendar import MARKET_TZ
from advisor.news import agent_run as A
from fastapi import HTTPException

NOW = datetime(2026, 9, 28, 18, 0, tzinfo=MARKET_TZ)


class J:
    """A judgment as far as the runner looks at it."""

    def __init__(self, about=True, materiality="MEDIUM"):
        self.about_company = about
        self.materiality = type("M", (), {"value": materiality})()


def test_run_pulls_then_judges_and_counts_material(tmp_path):
    seen = []

    def puller(store, sym, company):
        seen.append(("pull", sym, company))
        return [object(), object(), object()]

    def judge(store, conn, sym, now, company=None):
        seen.append(("judge", sym))
        return [J(), J(materiality="LOW"), J(about=False)], ["N2: stray number"]

    out = A.run_news_agent(tmp_path / "r.db", " feim ", NOW, puller=puller, judge=judge,
                           company="Frequency Electronics")  # fmt: skip
    assert seen == [("pull", "FEIM", "Frequency Electronics"), ("judge", "FEIM")]
    assert out == {"symbol": "FEIM", "pulled": 3, "judged": 3, "material": 1,
                   "problems": ["N2: stray number"]}  # fmt: skip


def test_a_failed_pull_still_judges_what_is_archived(tmp_path):
    def puller(store, sym, company):
        raise TimeoutError("tavily")

    out = A.run_news_agent(tmp_path / "r.db", "X", NOW, puller=puller,
                           judge=lambda *a, **k: ([J()], []), company="X")  # fmt: skip
    assert out["judged"] == 1 and "news pull failed" in out["problems"][0]


def test_without_pull_nothing_is_searched(tmp_path):
    def puller(store, sym, company):
        raise AssertionError("must not search")

    out = A.run_news_agent(tmp_path / "r.db", "X", NOW, pull=False, puller=puller,
                           judge=lambda *a, **k: ([], []), company="X")  # fmt: skip
    assert out["pulled"] == 0 and out["problems"] == []


def test_news_view_with_no_judgments(tmp_path):
    out = A.news_view(tmp_path / "none.db", "pl", NOW)
    assert out == {"symbol": "PL", "judgments": [], "summary": None}


def test_symbols_are_cleaned_and_bounded():
    assert D._clean([" pl ", "PL", "brk.b", ""]) == ["PL", "BRK.B"]
    for bad in (["../etc"], ["A B"], [], ["X" * 11]):
        with pytest.raises(HTTPException):
            D._clean(bad)
    with pytest.raises(HTTPException):
        D._clean([f"S{i}" for i in range(D.MAX_NAMES + 1)])


def test_each_name_runs_past_a_failure_and_reports_it():
    job = deps.new_job("news", target="A,B,C")

    def work(sym):
        if sym == "B":
            raise RuntimeError("boom")
        return {"symbol": sym, "problems": []}

    D._run_each(job, ["A", "B", "C"], "news agent", work)
    j = deps.get_job(job)
    assert j["status"] == "done" and "2/3 done" in j["message"] and "failed B" in j["message"]
    assert [r["symbol"] for r in j["results"]] == ["A", "B", "C"]


def test_all_failing_is_an_error():
    job = deps.new_job("news", target="A")
    D._run_each(job, ["A"], "news agent", lambda s: 1 / 0)
    assert deps.get_job(job)["status"] == "error"


def test_status_reads_news_and_reports(tmp_path):
    db = tmp_path / "research.db"
    conn = sqlite3.connect(db)
    conn.executescript(
        """
        CREATE TABLE news_judgments (key TEXT, prompt_version TEXT, symbol TEXT,
            published_at TEXT, payload_json TEXT, judged_at TEXT);
        CREATE TABLE research_reports (symbol TEXT, as_of TEXT, report_json TEXT,
            created_at TEXT);
        """
    )
    recent = (NOW - timedelta(days=2)).isoformat()
    old = (NOW - timedelta(days=40)).isoformat()

    def row(key, when, materiality):
        payload = json.dumps({"about_company": True, "materiality": materiality})
        return (key, "v", "FEIM", when, payload, when)

    rows = [row("k1", recent, "HIGH"), row("k2", recent, "LOW"), row("k3", old, "HIGH")]
    conn.executemany("INSERT INTO news_judgments VALUES (?,?,?,?,?,?)", rows)
    conn.execute(
        "INSERT INTO research_reports VALUES ('JBL', '2026-06-09', ?, '2026-06-09 13:59:50')",
        (json.dumps({"deep_research": {"summary": "x"}}),),
    )
    conn.commit()
    conn.close()
    s = D.status_of(db, ["FEIM", "JBL", "NONE"])
    assert s["FEIM"]["news_judged"] == 2 and s["FEIM"]["news_material"] == 1
    assert s["JBL"]["report_at"] == "2026-06-09T13:59:50Z" and s["JBL"]["deep_research"]
    assert s["NONE"] == {"news_judged": 0, "news_material": 0, "news_last": None,
                         "report_at": None, "deep_research": False}  # fmt: skip


def test_status_without_the_tables(tmp_path):
    s = D.status_of(tmp_path / "empty.db", ["X"])
    assert s["X"]["report_at"] is None and s["X"]["news_judged"] == 0
