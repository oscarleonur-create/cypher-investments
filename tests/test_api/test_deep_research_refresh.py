"""Refreshing only a report's deep research (user, 2026-10-15: SPCX's was weeks old, no refresh)."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pytest
from advisor.api.routers.research import refresh_deep_research
from advisor.research.models import DeepResearch, ResearchReport
from advisor.research.store import ResearchStore

OLD = datetime(2026, 9, 28, 13, 23)


@pytest.fixture
def db(tmp_path):
    path = tmp_path / "research.db"
    store = ResearchStore(path)
    store.save_report(
        ResearchReport(
            symbol="SPCX",
            as_of=date(2026, 9, 28),
            created_at=OLD,
            business_model="launch and broadband",
            deep_research=DeepResearch(symbol="SPCX", abstract="IPO in June", fetched_at=OLD),
        )
    )
    store.close()
    return path


def meta(sym):
    return ("Space Exploration Technologies", "Industrials", "Aerospace")


def load(db):
    store = ResearchStore(db)
    try:
        return store.load_latest_report("SPCX")
    finally:
        store.close()


def test_only_the_deep_research_is_replaced(db):
    calls = []

    def build(sym, name, sector, industry, filings):
        calls.append((sym, name, sector, industry))
        return DeepResearch(symbol=sym, abstract="spectrum deal, $8bn")

    out = refresh_deep_research("spcx", build=build, meta=meta, db_path=db)
    r = load(db)
    assert r.deep_research.abstract == "spectrum deal, $8bn"
    assert datetime.fromisoformat(out["fetched_at"]) > OLD + timedelta(days=1)
    # The rest of the report keeps its own date and content.
    assert r.created_at == OLD and r.as_of == date(2026, 9, 28)
    assert r.business_model == "launch and broadband"
    assert calls == [("SPCX", "Space Exploration Technologies", "Industrials", "Aerospace")]


@pytest.mark.parametrize("fresh", [None, DeepResearch(symbol="SPCX")])
def test_an_empty_brief_keeps_the_old_one(db, fresh):
    with pytest.raises(RuntimeError, match="came back empty"):
        refresh_deep_research("SPCX", build=lambda *a: fresh, meta=meta, db_path=db)
    assert load(db).deep_research.abstract == "IPO in June"


def test_no_report_is_said(tmp_path):
    with pytest.raises(LookupError, match="build the full report first"):
        refresh_deep_research("ZZZZ", build=lambda *a: None, meta=meta, db_path=tmp_path / "r.db")


def test_a_failing_build_keeps_the_old_one(db):
    def boom(*a):
        raise TimeoutError("tavily")

    with pytest.raises(TimeoutError):
        refresh_deep_research("SPCX", build=boom, meta=meta, db_path=db)
    assert load(db).deep_research.abstract == "IPO in June"


def test_the_endpoint_returns_a_job(monkeypatch, tmp_path):
    from advisor.api.app import create_app
    from advisor.api.routers import research as router
    from fastapi.testclient import TestClient

    # The job must not reach the network or the real store from a test.
    monkeypatch.setattr(router, "refresh_deep_research", lambda sym: {"fetched_at": "x"})
    with TestClient(create_app()) as c:
        r = c.post("/api/research/spcx/deep-research/refresh")
        assert r.status_code == 200 and r.json()["job_id"]
