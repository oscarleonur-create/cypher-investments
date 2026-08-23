"""API round-trip tests for the swing router (no network, no LLM)."""

from __future__ import annotations

import time
from datetime import datetime
from types import SimpleNamespace

import pytest
from advisor.api.app import create_app
from fastapi.testclient import TestClient


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


def _fake_result(verdict: str = "ENTER"):
    """A duck-typed stand-in for confluence.orchestrator's ConfluenceResult."""
    return SimpleNamespace(
        verdict=SimpleNamespace(value=verdict),
        reasoning="synthetic reasoning",
        suggested_hold_days=5,
        technical=SimpleNamespace(signal="BUY", price=100.0, is_bullish=True, volume_ratio=1.5),
        sentiment=SimpleNamespace(
            is_bullish=True, positive_pct=70.0, key_headlines=["h1", "h2", "h3", "h4"]
        ),
        fundamental=SimpleNamespace(
            is_clear=True,
            earnings_within_7_days=False,
            earnings_date=None,
            insider_buying_detected=False,
        ),
        ml_signal=None,
        scanned_at=datetime(2026, 6, 15, 9, 30),
    )


def _poll_until_done(client: TestClient, job_id: str) -> dict:
    deadline = time.time() + 5
    job = {}
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] != "running":
            break
        time.sleep(0.05)
    return job


def test_list_strategies(client: TestClient):
    resp = client.get("/api/swing/strategies")
    assert resp.status_code == 200
    names = {s["name"] for s in resp.json()["strategies"]}
    assert {"momentum_breakout", "buy_the_dip", "pead", "mean_reversion", "sma_crossover"} <= names


def test_scan_and_fetch_signals(client: TestClient, monkeypatch):
    # run_confluence is imported at module level into swing.py, so the patch
    # target is swing's own namespace, not advisor.confluence.orchestrator.
    monkeypatch.setattr(
        "advisor.api.routers.swing.run_confluence",
        lambda sym, **kw: _fake_result("ENTER"),
    )

    resp = client.post(
        "/api/swing/scan",
        json={"universe": "custom", "symbols": ["AAA"], "strategy": "momentum_breakout"},
    )
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]

    job = _poll_until_done(client, job_id)
    assert job["status"] == "done"

    result = client.get("/api/swing/signals", params={"job_id": job_id}).json()["result"]
    assert result is not None
    assert result["symbols_scanned"] == 1
    assert result["signals"][0]["symbol"] == "AAA"
    assert result["signals"][0]["verdict"] == "ENTER"
    assert result["signals"][0]["strategy"] == "momentum_breakout"


def test_scan_rejects_unknown_strategy(client: TestClient):
    resp = client.post(
        "/api/swing/scan",
        json={"universe": "custom", "symbols": ["AAA"], "strategy": "not_a_strategy"},
    )
    assert resp.status_code == 400


def test_scan_rejects_empty_symbol_list(client: TestClient):
    resp = client.post("/api/swing/scan", json={"universe": "custom", "symbols": []})
    assert resp.status_code == 400


def test_scan_caps_at_max_symbols(client: TestClient, monkeypatch):
    monkeypatch.setattr(
        "advisor.api.routers.swing.run_confluence",
        lambda sym, **kw: _fake_result("PASS"),
    )
    symbols = [f"SYM{i}" for i in range(10)]
    resp = client.post(
        "/api/swing/scan",
        json={"universe": "custom", "symbols": symbols, "max_symbols": 3},
    )
    assert resp.status_code == 200
    assert resp.json()["symbols"] == 3
