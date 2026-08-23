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


def test_run_sizes_signals_using_real_account_state(client: TestClient, monkeypatch):
    """The risk gate must actually consume live balances/positions, not just
    the session-is-None fallback (0.0, {}) that test_run_and_fetch_signals hits."""
    frames = {
        "AAA": _df([100.0] * 29 + [95.0]),
        "BBB": _df([100.0] * 29 + [95.0]),
    }
    monkeypatch.setattr(
        "advisor.scalping.scanner.fetch_intraday_candles",
        lambda symbols, **kw: ({s: frames[s] for s in symbols if s in frames}, "yfinance"),
    )

    async def _session():
        return "FAKE_SESSION"

    monkeypatch.setattr("advisor.api.deps.get_tt_session", _session)
    monkeypatch.setattr("advisor.scalping.catalysts.earnings_context", lambda s: (False, None))
    monkeypatch.setattr("advisor.scalping.catalysts.news_headlines", lambda s, **k: [])

    async def _fake_get_balances(session):
        return {"account": "TEST", "net_liq": 100_000.0, "cash": 50_000.0, "buying_power": 50_000.0}

    async def _fake_get_positions(session):
        # $10,000 notional already held in AAA == the 10% max-symbol-exposure
        # cap at $100k net liq, so AAA should get zero room; BBB (no existing
        # position) should size and approve normally off the same signal shape.
        return [
            {
                "symbol": "AAA",
                "underlying_symbol": "AAA",
                "quantity": 100,
                "quantity_direction": "Long",
                "instrument_type": "Equity",
                "multiplier": 1,
                "average_open_price": 90.0,
                "close_price": 95.0,
                "mark_price": 100.0,
                "mark": 100.0,
            }
        ]

    monkeypatch.setattr("advisor.market.tastytrade_client.get_balances", _fake_get_balances)
    monkeypatch.setattr("advisor.market.tastytrade_client.get_positions", _fake_get_positions)

    resp = client.post(
        "/api/scalping/run",
        json={
            "universe": "custom",
            "symbols": ["AAA", "BBB"],
            "interval": "5m",
            "min_rvol": 0,
            # Restrict to one strategy so each symbol produces exactly one
            # signal -- multiple strategies firing on the same synthetic
            # pattern would otherwise give each symbol several signals with
            # different risk_reward, muddying the exposure-cap comparison.
            "strategies": ["vwap_reversion"],
        },
    )
    assert resp.status_code == 200
    job_id = resp.json()["job_id"]

    deadline = time.time() + 5
    job = {}
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] != "running":
            break
        time.sleep(0.05)
    assert job["status"] == "done"

    result = client.get("/api/scalping/signals", params={"job_id": job_id}).json()["result"]
    by_symbol = {s["symbol"]: s for s in result["signals"]}
    assert len(result["signals"]) == 2  # exactly one per symbol

    assert by_symbol["AAA"]["risk_approved"] is False
    assert by_symbol["AAA"]["risk_quantity"] == 0

    assert by_symbol["BBB"]["risk_approved"] is True
    assert by_symbol["BBB"]["risk_quantity"] > 0


def test_run_falls_back_to_blocked_when_account_state_fails(client: TestClient, monkeypatch):
    """A live-session hiccup fetching balances/positions must not crash the
    scan — every signal should just come back ungated (net_liq=0 -> blocked)."""
    frames = {"AAA": _df([100.0] * 29 + [95.0])}
    monkeypatch.setattr(
        "advisor.scalping.scanner.fetch_intraday_candles",
        lambda symbols, **kw: ({s: frames[s] for s in symbols if s in frames}, "yfinance"),
    )

    async def _session():
        return "FAKE_SESSION"

    monkeypatch.setattr("advisor.api.deps.get_tt_session", _session)
    monkeypatch.setattr("advisor.scalping.catalysts.earnings_context", lambda s: (False, None))
    monkeypatch.setattr("advisor.scalping.catalysts.news_headlines", lambda s, **k: [])

    async def _boom(session):
        raise RuntimeError("tastytrade API hiccup")

    monkeypatch.setattr("advisor.market.tastytrade_client.get_balances", _boom)
    monkeypatch.setattr("advisor.market.tastytrade_client.get_positions", _boom)

    resp = client.post(
        "/api/scalping/run",
        json={"universe": "custom", "symbols": ["AAA"], "interval": "5m", "min_rvol": 0},
    )
    job_id = resp.json()["job_id"]

    deadline = time.time() + 5
    job = {}
    while time.time() < deadline:
        job = client.get(f"/api/jobs/{job_id}").json()
        if job["status"] != "running":
            break
        time.sleep(0.05)
    assert job["status"] == "done"

    result = client.get("/api/scalping/signals", params={"job_id": job_id}).json()["result"]
    assert result["signals"], "expected the synthetic AAA signal to still be produced"
    for s in result["signals"]:
        assert s["risk_approved"] is False
        assert s["risk_note"] == "no account data"
