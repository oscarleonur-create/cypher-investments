"""Tests for the position recommendation builder.

Same anti-hallucination discipline as investment_memo.py/thesis.py (this
session's earlier fixes): the regression guard here is a prompt-content
assertion that the "use only given facts" instruction stays present, plus
tests confirming position numbers are always Python-computed, never trusted
from the LLM's own output.
"""

from __future__ import annotations

import asyncio
from datetime import date
from unittest.mock import MagicMock, patch

from advisor.research.models import (
    CatalystItem,
    CatalystRiskResult,
    CatalystType,
    DcfAssumptions,
    DcfResult,
    DcfScenario,
    PositionContext,
    ResearchReport,
)
from advisor.research.recommendation import (
    _build_context,
    _llm_recommend,
    build_recommendation,
    fetch_position_context,
)


def _report(**overrides) -> ResearchReport:
    base = dict(symbol="ACME", as_of=date.today(), business_model="ACME makes sensors.")
    base.update(overrides)
    return ResearchReport(**base)


def _no_position() -> PositionContext:
    return PositionContext(has_position=False)


def _equity_position() -> PositionContext:
    return PositionContext(
        has_position=True,
        equity_quantity=100.0,
        average_open_price=50.0,
        mark=55.0,
        unrealized_pnl_pct=0.10,
        accounts=["5WI30382"],
    )


# ── _build_context ──────────────────────────────────────────────────────────


def test_build_context_no_position():
    context = _build_context("ACME", _report(), _no_position())
    assert "Current position: none." in context


def test_build_context_includes_real_position_numbers():
    context = _build_context("ACME", _report(), _equity_position())
    assert "100 shares @ avg $50.00, mark $55.00" in context
    assert "unrealized +10.0%" in context


def test_build_context_no_report_says_so_explicitly():
    context = _build_context("ACME", None, _no_position())
    assert "No cached research report is available" in context


def test_build_context_includes_only_near_term_catalysts():
    report = _report(
        catalyst_risk=CatalystRiskResult(
            symbol="ACME",
            catalysts=[
                CatalystItem(
                    description="Earnings",
                    catalyst_type=CatalystType.EARNINGS,
                    expected_date="2099-01-01",
                    is_near_term=False,
                ),
                CatalystItem(
                    description="Product launch",
                    catalyst_type=CatalystType.PRODUCT_LAUNCH,
                    expected_date="2026-09-01",
                    is_near_term=True,
                ),
            ],
        )
    )
    context = _build_context("ACME", report, _no_position())
    assert "Product launch" in context
    assert "Earnings" not in context  # not near-term -> excluded


# ── _llm_recommend prompt guardrail (anti-hallucination, applied from day 1) ─


def test_llm_recommend_system_prompt_forbids_inventing_facts():
    captured = {}

    def _fake_complete(*, system_prompt, user_prompt, response_model):
        captured["system_prompt"] = system_prompt
        return response_model(
            action="HOLD", conviction="LOW", reasoning="r", key_factors=[], risks=[]
        )

    with (
        patch("research_agent.config.ResearchConfig") as mock_cfg,
        patch("research_agent.llm.OpenRouterLLM") as mock_llm_cls,
    ):
        mock_cfg.return_value = MagicMock(openrouter_api_key="key")
        mock_llm_cls.return_value.complete.side_effect = _fake_complete
        _llm_recommend("ACME", _report(), _no_position())

    prompt = captured["system_prompt"]
    assert "Use ONLY the facts" in prompt
    assert "Do NOT invent" in prompt
    assert date.today().isoformat() in prompt  # the catalysts.py date-anchor lesson


def test_llm_recommend_restricts_valid_actions_when_no_position():
    captured = {}

    def _fake_complete(*, system_prompt, user_prompt, response_model):
        captured["system_prompt"] = system_prompt
        return response_model(
            action="BUY", conviction="LOW", reasoning="r", key_factors=[], risks=[]
        )

    with (
        patch("research_agent.config.ResearchConfig") as mock_cfg,
        patch("research_agent.llm.OpenRouterLLM") as mock_llm_cls,
    ):
        mock_cfg.return_value = MagicMock(openrouter_api_key="key")
        mock_llm_cls.return_value.complete.side_effect = _fake_complete
        _llm_recommend("ACME", _report(), _no_position())

    assert "never SELL/INCREASE/DECREASE" in captured["system_prompt"]


def test_llm_recommend_restricts_valid_actions_when_position_held():
    captured = {}

    def _fake_complete(*, system_prompt, user_prompt, response_model):
        captured["system_prompt"] = system_prompt
        return response_model(
            action="HOLD", conviction="MEDIUM", reasoning="r", key_factors=[], risks=[]
        )

    with (
        patch("research_agent.config.ResearchConfig") as mock_cfg,
        patch("research_agent.llm.OpenRouterLLM") as mock_llm_cls,
    ):
        mock_cfg.return_value = MagicMock(openrouter_api_key="key")
        mock_llm_cls.return_value.complete.side_effect = _fake_complete
        _llm_recommend("ACME", _report(), _equity_position())

    assert "never BUY" in captured["system_prompt"]


def test_llm_recommend_returns_empty_dict_without_api_key():
    with patch("research_agent.config.ResearchConfig") as mock_cfg:
        mock_cfg.return_value = MagicMock(openrouter_api_key="")
        assert _llm_recommend("ACME", _report(), _no_position()) == {}


def test_llm_recommend_swallows_unexpected_exceptions():
    with patch("research_agent.config.ResearchConfig", side_effect=RuntimeError("boom")):
        assert _llm_recommend("ACME", _report(), _no_position()) == {}


# ── build_recommendation (position is always Python-computed, never LLM) ────


def test_build_recommendation_uses_real_position_data_not_llm_output():
    position = _equity_position()

    def _fake_complete(*, system_prompt, user_prompt, response_model):
        # An LLM response has no way to express position data at all --
        # this asserts the response schema, not just that we ignore it.
        return response_model(
            action="INCREASE", conviction="HIGH", reasoning="r", key_factors=["f"], risks=["x"]
        )

    with (
        patch("research_agent.config.ResearchConfig") as mock_cfg,
        patch("research_agent.llm.OpenRouterLLM") as mock_llm_cls,
    ):
        mock_cfg.return_value = MagicMock(openrouter_api_key="key")
        mock_llm_cls.return_value.complete.side_effect = _fake_complete
        result = build_recommendation("ACME", _report(), position)

    assert result.action == "INCREASE"
    assert result.position == position  # exact real object, not LLM-derived
    assert result.position.equity_quantity == 100.0
    assert result.position.average_open_price == 50.0


def test_build_recommendation_falls_back_gracefully_without_llm():
    with patch("research_agent.config.ResearchConfig") as mock_cfg:
        mock_cfg.return_value = MagicMock(openrouter_api_key="")
        result = build_recommendation("ACME", None, _no_position())

    assert result.action == "HOLD"  # safe default, no action fabricated
    assert result.conviction == "LOW"
    assert "build research" in result.reasoning.lower()


def test_build_recommendation_report_missing_and_position_held():
    """No report is the more actionable fact regardless of position state --
    the fallback tells the user what to do next (build research) rather than
    just restating that they hold something."""
    with patch("research_agent.config.ResearchConfig") as mock_cfg:
        mock_cfg.return_value = MagicMock(openrouter_api_key="")
        result = build_recommendation("ACME", None, _equity_position())

    assert result.action == "HOLD"
    assert "build research" in result.reasoning.lower()
    assert result.position.has_position is True  # real position data still returned


def _dcf_assumptions() -> DcfAssumptions:
    return DcfAssumptions(
        scenario="base",
        revenue_growth_yr1_3=0.1,
        revenue_growth_yr4_10=0.05,
        target_fcf_margin=0.2,
        capex_intensity=0.05,
    )


def test_build_recommendation_with_dcf_context_full_report():
    position = _no_position()
    report = _report(
        dcf=DcfResult(
            symbol="ACME",
            current_price=100.0,
            shares_outstanding=1.0,
            net_debt=0.0,
            wacc=0.09,
            risk_free_rate=0.04,
            base=DcfScenario(
                assumptions=_dcf_assumptions(),
                enterprise_value=1.0,
                equity_value=1.0,
                implied_price=120.0,
                upside_pct=0.20,
            ),
        )
    )

    def _fake_complete(*, system_prompt, user_prompt, response_model):
        assert "DCF: current $100.00" in user_prompt
        return response_model(
            action="BUY", conviction="MEDIUM", reasoning="r", key_factors=["f"], risks=[]
        )

    with (
        patch("research_agent.config.ResearchConfig") as mock_cfg,
        patch("research_agent.llm.OpenRouterLLM") as mock_llm_cls,
    ):
        mock_cfg.return_value = MagicMock(openrouter_api_key="key")
        mock_llm_cls.return_value.complete.side_effect = _fake_complete
        result = build_recommendation("ACME", report, position)

    assert result.action == "BUY"


# ── fetch_position_context (account-looping, equity merge, option legs) ─────


def test_fetch_position_context_merges_equity_across_accounts(monkeypatch):
    async def _session():
        return "FAKE_SESSION"

    async def _positions(session, acct):
        if acct == "5WI30382":
            return [
                {
                    "symbol": "AAPL",
                    "underlying_symbol": "AAPL",
                    "quantity": 50,
                    "quantity_direction": "Long",
                    "instrument_type": "Equity",
                    "multiplier": 1,
                    "average_open_price": 100.0,
                    "close_price": 110.0,
                    "mark_price": 110.0,
                    "mark": 110.0,
                },
            ]
        return [
            {
                "symbol": "AAPL",
                "underlying_symbol": "AAPL",
                "quantity": 50,
                "quantity_direction": "Long",
                "instrument_type": "Equity",
                "multiplier": 1,
                "average_open_price": 120.0,
                "close_price": 110.0,
                "mark_price": 110.0,
                "mark": 110.0,
            },
        ]

    monkeypatch.setattr("advisor.api.deps.try_get_tt_session", _session)
    monkeypatch.setattr("advisor.market.tastytrade_client.get_positions", _positions)

    position = asyncio.run(fetch_position_context("AAPL"))

    assert position.has_position is True
    assert position.equity_quantity == 100.0
    # weighted average: (100*50 + 120*50) / 100 = 110.0
    assert position.average_open_price == 110.0
    assert position.mark == 110.0
    assert set(position.accounts) == {"5WI30382", "5WI47366"}


def test_fetch_position_context_lists_option_legs_separately(monkeypatch):
    async def _session():
        return "FAKE_SESSION"

    async def _positions(session, acct):
        if acct != "5WI30382":
            return []
        return [
            {
                "symbol": "AAPL  260320P00150000",
                "underlying_symbol": "AAPL",
                "quantity": 1,
                "quantity_direction": "Short",
                "instrument_type": "Equity Option",
                "multiplier": 100,
                "average_open_price": 2.5,
                "close_price": 2.0,
                "mark_price": 2.0,
                "mark": 2.0,
            },
        ]

    monkeypatch.setattr("advisor.api.deps.try_get_tt_session", _session)
    monkeypatch.setattr("advisor.market.tastytrade_client.get_positions", _positions)

    position = asyncio.run(fetch_position_context("AAPL"))

    assert position.has_position is True
    assert position.equity_quantity == 0.0  # no equity leg, only the option
    assert len(position.option_legs) == 1
    assert "SELL 1x" in position.option_legs[0]


def test_fetch_position_context_filters_to_requested_symbol(monkeypatch):
    async def _session():
        return "FAKE_SESSION"

    async def _positions(session, acct):
        return [
            {
                "symbol": "MSFT",
                "underlying_symbol": "MSFT",
                "quantity": 10,
                "quantity_direction": "Long",
                "instrument_type": "Equity",
                "multiplier": 1,
                "average_open_price": 300.0,
                "close_price": 310.0,
                "mark_price": 310.0,
                "mark": 310.0,
            },
        ]

    monkeypatch.setattr("advisor.api.deps.try_get_tt_session", _session)
    monkeypatch.setattr("advisor.market.tastytrade_client.get_positions", _positions)

    position = asyncio.run(fetch_position_context("AAPL"))
    assert position.has_position is False  # only MSFT positions exist, AAPL requested


def test_fetch_position_context_no_session_returns_error_not_crash(monkeypatch):
    async def _no_session():
        return None

    monkeypatch.setattr("advisor.api.deps.try_get_tt_session", _no_session)

    position = asyncio.run(fetch_position_context("AAPL"))
    assert position.has_position is False
    assert position.error
