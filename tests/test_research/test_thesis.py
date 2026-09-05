"""Tests for the bull/base/bear thesis builder.

`_llm_enrich`'s system prompt is the fix for a real, live-reproduced bug: with
a sparse context (no DCF/catalyst/variant-perception data -- a real code path,
not a contrived edge case, since dcf=None whenever DCF computation is
unavailable), the model fabricated specific figures with zero basis in the
input: "150-200bps" margin expansion, ">90% FCF conversion", "4-6% annually"
growth, "200bps+" margin contraction -- despite the model's own summary
admitting no valuation context was provided. Same bug class as
investment_memo.py's fabrication bug, prompt-only fix (no structural
post-check applies to free-text prose the way date math applies to
catalysts.py's ISO dates).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from advisor.research.models import (
    CatalystItem,
    CatalystRiskResult,
    CatalystType,
    DcfAssumptions,
    DcfResult,
    DcfScenario,
    MispricingType,
    RiskItem,
    RiskSeverity,
    VariantPerceptionResult,
)
from advisor.research.thesis import build_thesis


def _scenario(growth: float, margin: float, price: float, upside: float) -> DcfScenario:
    return DcfScenario(
        assumptions=DcfAssumptions(
            scenario="x",
            revenue_growth_yr1_3=growth,
            revenue_growth_yr4_10=growth / 2,
            target_fcf_margin=margin,
            capex_intensity=0.05,
        ),
        enterprise_value=1.0,
        equity_value=1.0,
        implied_price=price,
        upside_pct=upside,
    )


def _dcf() -> DcfResult:
    return DcfResult(
        symbol="ACME",
        current_price=100.0,
        shares_outstanding=1.0,
        net_debt=0.0,
        wacc=0.10,
        risk_free_rate=0.04,
        beta=1.2,
        base=_scenario(0.10, 0.20, 120.0, 0.20),
        bull=_scenario(0.20, 0.25, 160.0, 0.60),
        bear=_scenario(0.02, 0.15, 70.0, -0.30),
    )


# ── build_thesis (deterministic target-price assembly) ──────────────────────


def test_build_thesis_targets_come_from_dcf_not_the_llm():
    """Target prices/upside are computed in build_thesis from the DCF
    scenarios directly -- the LLM never supplies them, so they can't drift
    from the deterministic valuation even if the LLM call fails entirely."""
    with patch("advisor.research.thesis._llm_enrich", return_value={}):
        result = build_thesis("acme", "ACME Corp", dcf=_dcf())

    assert result.symbol == "ACME"
    assert result.current_price == 100.0
    assert result.bull.target_price == 160.0
    assert result.bull.upside_pct == pytest.approx(0.60)
    assert result.base.target_price == 120.0
    assert result.bear.target_price == 70.0
    # LLM enrichment empty -> falls back to _DEFAULT_PROBS, empty narrative.
    assert result.bull.probability == 0.25
    assert result.base.probability == 0.50
    assert result.bear.probability == 0.25
    assert result.conviction == "MEDIUM"


def test_build_thesis_uses_llm_narrative_when_available():
    llm_out = {
        "bull_prob": 0.3,
        "bull_desc": "bull story",
        "bull_assumptions": ["a1"],
        "bull_wrong": ["w1"],
        "base_prob": 0.5,
        "base_desc": "base story",
        "base_assumptions": ["a2"],
        "base_wrong": ["w2"],
        "bear_prob": 0.2,
        "bear_desc": "bear story",
        "bear_assumptions": ["a3"],
        "bear_wrong": ["w3"],
        "summary": "the summary",
        "conviction": "HIGH",
    }
    with patch("advisor.research.thesis._llm_enrich", return_value=llm_out):
        result = build_thesis("ACME", "ACME Corp", dcf=_dcf())

    assert result.bull.description == "bull story"
    assert result.bull.probability == 0.3
    assert result.summary == "the summary"
    assert result.conviction == "HIGH"


# ── _llm_enrich prompt guardrail (the fix itself) ────────────────────────────


def test_llm_enrich_system_prompt_forbids_inventing_numbers():
    """Regression guard: the anti-fabrication instruction must stay present,
    not be quietly dropped or watered down in a future edit."""
    captured = {}

    def _fake_complete(*, system_prompt, user_prompt, response_model):
        captured["system_prompt"] = system_prompt
        return response_model(
            bull_prob=0.25,
            bull_desc="",
            bull_assumptions=[],
            bull_wrong=[],
            base_prob=0.5,
            base_desc="",
            base_assumptions=[],
            base_wrong=[],
            bear_prob=0.25,
            bear_desc="",
            bear_assumptions=[],
            bear_wrong=[],
            summary="",
            conviction="LOW",
        )

    with (
        patch("research_agent.config.ResearchConfig") as mock_cfg,
        patch("research_agent.llm.OpenRouterLLM") as mock_llm_cls,
    ):
        mock_cfg.return_value = MagicMock(openrouter_api_key="key")
        mock_llm_cls.return_value.complete.side_effect = _fake_complete
        from advisor.research.thesis import _llm_enrich

        _llm_enrich("ACME", "ACME Corp", None, None, None)

    prompt = captured["system_prompt"]
    assert "Do NOT" in prompt and "invent" in prompt
    assert "sparse" in prompt.lower()
    assert "LOW whenever the context is sparse" in prompt


def test_llm_enrich_returns_empty_dict_without_api_key():
    from advisor.research.thesis import _llm_enrich

    with patch("research_agent.config.ResearchConfig") as mock_cfg:
        mock_cfg.return_value = MagicMock(openrouter_api_key="")
        assert _llm_enrich("ACME", "ACME Corp", None, None, None) == {}


def test_llm_enrich_swallows_exceptions():
    from advisor.research.thesis import _llm_enrich

    with patch("research_agent.config.ResearchConfig", side_effect=RuntimeError("boom")):
        assert _llm_enrich("ACME", "ACME Corp", None, None, None) == {}


def test_llm_enrich_context_includes_only_available_sections():
    """The context block the LLM sees should reflect exactly what's passed --
    proves the sparse-context test above is actually representative of a real
    partial-data code path, not an artificial empty case."""
    from advisor.research.thesis import _llm_enrich

    captured = {}

    def _fake_complete(*, system_prompt, user_prompt, response_model):
        captured["user_prompt"] = user_prompt
        return response_model(
            bull_prob=0.25,
            bull_desc="",
            bull_assumptions=[],
            bull_wrong=[],
            base_prob=0.5,
            base_desc="",
            base_assumptions=[],
            base_wrong=[],
            bear_prob=0.25,
            bear_desc="",
            bear_assumptions=[],
            bear_wrong=[],
            summary="",
            conviction="LOW",
        )

    catalyst_risk = CatalystRiskResult(
        symbol="ACME",
        risks=[RiskItem(description="concentration risk", severity=RiskSeverity.HIGH)],
        catalysts=[
            CatalystItem(description="launch event", catalyst_type=CatalystType.PRODUCT_LAUNCH)
        ],
    )
    vp = VariantPerceptionResult(
        symbol="ACME", our_key_insight="the edge", mispricing_type=MispricingType.ESTIMATE_REVISION
    )

    with (
        patch("research_agent.config.ResearchConfig") as mock_cfg,
        patch("research_agent.llm.OpenRouterLLM") as mock_llm_cls,
    ):
        mock_cfg.return_value = MagicMock(openrouter_api_key="key")
        mock_llm_cls.return_value.complete.side_effect = _fake_complete
        _llm_enrich("ACME", "ACME Corp", None, catalyst_risk, vp)

    prompt = captured["user_prompt"]
    assert "concentration risk" in prompt
    assert "launch event" in prompt
    assert "the edge" in prompt
    assert "DCF" not in prompt  # no dcf passed -> not mentioned
