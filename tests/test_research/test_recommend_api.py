"""API round-trip test for POST /api/research/{symbol}/recommend."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from advisor.api.app import create_app
from advisor.research.models import ActionRecommendation, PositionContext
from fastapi.testclient import TestClient


@pytest.fixture
def client() -> TestClient:
    return TestClient(create_app())


def test_recommend_returns_grounded_result(client: TestClient, monkeypatch):
    async def _fake_position(symbol: str) -> PositionContext:
        return PositionContext(has_position=False)

    def _fake_build(symbol, report, position):
        return ActionRecommendation(
            symbol=symbol,
            action="BUY",
            conviction="MEDIUM",
            reasoning="synthetic",
            key_factors=["f1"],
            risks=["r1"],
            position=position,
        )

    monkeypatch.setattr("advisor.research.recommendation.fetch_position_context", _fake_position)
    monkeypatch.setattr("advisor.research.recommendation.build_recommendation", _fake_build)

    resp = client.post("/api/research/AAPL/recommend")
    assert resp.status_code == 200
    body = resp.json()
    assert body["symbol"] == "AAPL"
    assert body["action"] == "BUY"
    assert body["position"]["has_position"] is False


def test_recommend_never_exposes_a_place_order_action(client: TestClient, monkeypatch):
    """Decision support only -- assert the response shape has no field that
    could be mistaken for (or misused as) an execution trigger."""

    async def _fake_position(symbol: str) -> PositionContext:
        return PositionContext(has_position=True, equity_quantity=10.0)

    monkeypatch.setattr("advisor.research.recommendation.fetch_position_context", _fake_position)

    # No mocked build_recommendation here (unlike the test above) -- keep the
    # real deterministic fallback path, but force no live LLM call.
    with patch("research_agent.config.ResearchConfig") as mock_cfg:
        mock_cfg.return_value = MagicMock(openrouter_api_key="")
        resp = client.post("/api/research/AAPL/recommend")
    assert resp.status_code == 200
    body = resp.json()
    assert set(body.keys()) == {
        "symbol",
        "action",
        "conviction",
        "reasoning",
        "key_factors",
        "risks",
        "position",
        "generated_at",
    }
