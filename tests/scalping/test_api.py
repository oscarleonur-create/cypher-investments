"""API round-trip tests for the scalping router (no network)."""

from __future__ import annotations

import time

import pandas as pd
import pytest
from advisor.api.app import create_app
from fastapi.testclient import TestClient


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


def _df(closes: list[float]) -> pd.DataFrame:
    idx = pd.date_range("2026-06-15 09:30", periods=len(closes), freq="5min")
    close = pd.Series(closes, index=idx, dtype=float)
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.1,
            "low": close - 0.1,
            "close": close,
            "volume": 1000.0,
        },
        index=idx,
    )


def test_list_strategies(client: TestClient):
    resp = client.get("/api/scalping/strategies")
    assert resp.status_code == 200
    names = {s["name"] for s in resp.json()["strategies"]}
    assert {"vwap_reversion", "rsi2_mean_reversion", "opening_range_breakout"} <= names


def test_run_and_fetch_signals(client: TestClient, monkeypatch):
    # Avoid the live feed and TastyTrade session entirely.
    frames = {"AAA": _df([100.0] * 29 + [95.0])}
    monkeypatch.setattr(
        "advisor.scalping.scanner.fetch_intraday_candles",
        lambda symbols, **kw: ({s: frames[s] for s in symbols if s in frames}, "yfinance"),
    )

    async def _no_session():
        return None

    monkeypatch.setattr("advisor.api.deps.get_tt_session", _no_session)
    # Keep the catalyst tiers offline.
    monkeypatch.setattr("advisor.scalping.catalysts.earnings_context", lambda s: (False, None))
    monkeypatch.setattr("advisor.scalping.catalysts.news_headlines", lambda s, **k: [])

    resp = client.post(
        "/api/scalping/run",
        # min_rvol=0 disables the gate so the flat-volume synthetic signal survives.
        json={"universe": "custom", "symbols": ["AAA"], "interval": "5m", "min_rvol": 0},
    )
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]

    # Background job runs via asyncio.to_thread; poll until done.
    deadline = time.time() + 5
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] != "running":
            break
        time.sleep(0.05)
    assert job["status"] == "done"

    result = client.get("/api/scalping/signals", params={"job_id": job_id}).json()["result"]
    assert result is not None
    assert result["source"] == "yfinance"
    assert any(s["symbol"] == "AAA" for s in result["signals"])


def test_run_rejects_unknown_strategy(client: TestClient):
    resp = client.post(
        "/api/scalping/run",
        json={"universe": "custom", "symbols": ["AAA"], "strategies": ["nope"]},
    )
    assert resp.status_code == 400
