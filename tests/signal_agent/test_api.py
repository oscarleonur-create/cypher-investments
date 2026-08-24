"""API round-trip tests for the Signal agent router (no network, no LLM)."""

from __future__ import annotations

import time
from types import SimpleNamespace

import pytest
from advisor.agent.llm import AgentLLM
from advisor.api.app import create_app
from fastapi.testclient import TestClient


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


def _poll_until_done(client: TestClient, job_id: str) -> dict:
    deadline = time.time() + 5
    job: dict = {}
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] != "running":
            break
        time.sleep(0.05)
    return job


def test_run_with_llm_unavailable_still_completes(client: TestClient, monkeypatch):
    """No API key configured -> the agent returns gracefully, job still
    completes rather than hanging or erroring the whole request."""
    monkeypatch.setattr(AgentLLM, "configured", property(lambda self: False))

    async def _no_session():
        return None

    monkeypatch.setattr("advisor.api.deps.get_tt_session", _no_session)

    resp = client.post("/api/signal-agent/run", json={"objective": "find scalp setups"})
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]

    job = _poll_until_done(client, job_id)
    assert job["status"] == "done"

    result = client.get("/api/signal-agent/results", params={"job_id": job_id}).json()["result"]
    assert result is not None
    assert result["signals"] == []
    assert "unavailable" in result["notes"].lower()


def test_run_persists_and_returns_rationale(client: TestClient, monkeypatch):
    monkeypatch.setattr(AgentLLM, "configured", property(lambda self: True))

    async def _no_session():
        return None

    monkeypatch.setattr("advisor.api.deps.get_tt_session", _no_session)

    final = SimpleNamespace(
        content='{"rationale": "quiet market, nothing worth scanning", "focus_symbols": []}',
        tool_calls=None,
    )
    monkeypatch.setattr(
        AgentLLM,
        "chat_with_tools",
        lambda self, *a, **k: SimpleNamespace(choices=[SimpleNamespace(message=final)]),
    )

    resp = client.post(
        "/api/signal-agent/run",
        json={"objective": "find scalp setups", "universe": "custom"},
    )
    job_id = resp.json()["job_id"]
    job = _poll_until_done(client, job_id)
    assert job["status"] == "done"

    result = client.get("/api/signal-agent/results", params={"job_id": job_id}).json()["result"]
    assert result["rationale"] == "quiet market, nothing worth scanning"
    assert result["run_id"]  # persisted to agent_runs
