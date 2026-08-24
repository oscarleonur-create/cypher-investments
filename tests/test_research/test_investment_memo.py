"""Tests for the investment memo builder.

`_llm_enrich`'s system prompt is the fix for a real, live-reproduced bug: with
a fully controlled context containing no price/multiple/downside figures at
all, the model fabricated specific numbers ("35-40% downside", "5-6x
earnings", a "12x" re-rate threshold, invented QoQ/concentration thresholds)
that had no basis in the input — dangerous for a memo that drives real
position-sizing and exit decisions. The fix is prompt-only (no structural
post-check like catalysts.py's date math applies to free-text prose), so the
regression guard here is a prompt-content assertion: the "do not invent
numbers" instruction must stay present and specific.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

from advisor.research.investment_memo import (
    _build_context,
    _fallback_exit_triggers,
    _fallback_summary,
    _llm_enrich,
    build_investment_memo,
)
from advisor.research.models import (
    CatalystRiskResult,
    DcfAssumptions,
    DcfResult,
    DcfScenario,
    IndustryAnalysis,
    MispricingType,
    MoatType,
    ResearchReport,
    RiskItem,
    RiskSeverity,
    ThesisResult,
    ThesisScenario,
    VariantPerceptionResult,
)


def _dcf_assumptions() -> DcfAssumptions:
    return DcfAssumptions(
        scenario="base",
        revenue_growth_yr1_3=0.1,
        revenue_growth_yr4_10=0.05,
        target_fcf_margin=0.2,
        capex_intensity=0.05,
    )


def _report(**overrides) -> ResearchReport:
    base = dict(symbol="ACME", as_of=date.today(), business_model="ACME makes sensors.")
    base.update(overrides)
    return ResearchReport(**base)


# ── _build_context ────────────────────────────────────────────────────────────


def test_build_context_includes_only_available_sections():
    report = _report(
        variant_perception=VariantPerceptionResult(
            symbol="ACME", our_key_insight="edge", our_edge="why we differ"
        ),
    )
    context = _build_context(report)
    assert "Our edge: edge" in context
    assert "Why we differ: why we differ" in context
    assert "DCF" not in context  # no dcf section on the report -> not mentioned


def test_build_context_truncates_long_free_text_fields():
    report = _report(
        variant_perception=VariantPerceptionResult(symbol="ACME", our_edge="x" * 500),
    )
    context = _build_context(report)
    edge_line = next(line for line in context.splitlines() if line.startswith("Why we differ"))
    assert len(edge_line) < 320  # "Why we differ: " prefix + 300-char cap


# ── _llm_enrich prompt guardrail (the fix itself) ────────────────────────────


def test_llm_enrich_system_prompt_forbids_inventing_numbers():
    """Regression guard for the fabrication bug: the anti-invention
    instruction must stay present verbatim-in-spirit, not be quietly dropped
    or watered down in a future edit."""
    report = _report()
    captured = {}

    def _fake_complete(*, system_prompt, user_prompt, response_model):
        captured["system_prompt"] = system_prompt
        return response_model(
            executive_summary="s",
            max_loss_scenario="m",
            bear_case_deep_dive="b",
            exit_triggers=["t"],
        )

    with (
        patch("research_agent.config.ResearchConfig") as mock_cfg,
        patch("research_agent.llm.OpenRouterLLM") as mock_llm_cls,
    ):
        mock_cfg.return_value = MagicMock(openrouter_api_key="key")
        mock_llm_cls.return_value.complete.side_effect = _fake_complete
        _llm_enrich(report)

    prompt = captured["system_prompt"]
    assert "Do NOT" in prompt and "invent" in prompt
    assert "not explicitly stated" in prompt
    assert "do not introduce new risk factors" in prompt.lower()


def test_llm_enrich_returns_empty_dict_without_api_key():
    report = _report()
    with patch("research_agent.config.ResearchConfig") as mock_cfg:
        mock_cfg.return_value = MagicMock(openrouter_api_key="")
        assert _llm_enrich(report) == {}


def test_llm_enrich_swallows_exceptions():
    report = _report()
    with patch("research_agent.config.ResearchConfig", side_effect=RuntimeError("boom")):
        assert _llm_enrich(report) == {}


# ── build_investment_memo (deterministic assembly, LLM mocked) ──────────────


def test_build_investment_memo_assembles_deterministic_sections():
    report = _report(
        variant_perception=VariantPerceptionResult(
            symbol="ACME", our_key_insight="key insight", mispricing_type=MispricingType.NONE
        ),
        industry=IndustryAnalysis(
            symbol="ACME", moat_type=MoatType.SWITCHING_COSTS, moat_description="locked in"
        ),
        dcf=DcfResult(
            symbol="ACME",
            current_price=100.0,
            shares_outstanding=1.0,
            net_debt=0.0,
            wacc=0.10,
            risk_free_rate=0.04,
            base=DcfScenario(
                assumptions=_dcf_assumptions(),
                enterprise_value=1.0,
                equity_value=1.0,
                implied_price=120.0,
                upside_pct=0.2,
            ),
        ),
        catalyst_risk=CatalystRiskResult(
            symbol="ACME",
            catalysts=[],
            risks=[RiskItem(description="concentration risk", severity=RiskSeverity.HIGH)],
        ),
        thesis=ThesisResult(symbol="ACME", conviction="HIGH"),
    )

    with patch("advisor.research.investment_memo._llm_enrich", return_value={}):
        memo = build_investment_memo(report)

    assert memo.key_insight == "key insight"
    assert memo.moat_type == "switching_costs"
    assert memo.current_price == 100.0
    assert memo.base_target == 120.0
    assert memo.conviction == "HIGH"
    assert memo.position_size_pct_low == 5.0 and memo.position_size_pct_high == 8.0
    # LLM enrichment empty -> falls back to deterministic summary/triggers.
    assert memo.executive_summary != ""
    assert memo.exit_triggers


def test_build_investment_memo_uses_llm_output_when_available():
    report = _report(thesis=ThesisResult(symbol="ACME", conviction="LOW"))
    llm_out = {
        "executive_summary": "llm summary",
        "max_loss_scenario": "llm max loss",
        "bear_case_deep_dive": "llm bear case",
        "exit_triggers": ["llm trigger 1", "llm trigger 2"],
    }
    with patch("advisor.research.investment_memo._llm_enrich", return_value=llm_out):
        memo = build_investment_memo(report)

    assert memo.executive_summary == "llm summary"
    assert memo.max_loss_scenario == "llm max loss"
    assert memo.bear_case_deep_dive == "llm bear case"
    assert memo.exit_triggers == ["llm trigger 1", "llm trigger 2"]


# ── Fallbacks (used when the LLM is unavailable) ─────────────────────────────


def test_fallback_summary_uses_business_model_and_edge():
    report = _report(
        business_model="ACME makes sensors.",
        variant_perception=VariantPerceptionResult(symbol="ACME", our_key_insight="the edge"),
    )
    summary = _fallback_summary(report)
    assert "ACME makes sensors." in summary
    assert "the edge" in summary


def test_fallback_exit_triggers_caps_at_five():
    report = _report(
        dcf=DcfResult(
            symbol="ACME",
            current_price=100.0,
            shares_outstanding=1.0,
            net_debt=0.0,
            wacc=0.10,
            risk_free_rate=0.04,
            bear=DcfScenario(
                assumptions=_dcf_assumptions(),
                enterprise_value=1.0,
                equity_value=1.0,
                implied_price=60.0,
                upside_pct=-0.4,
            ),
        ),
        thesis=ThesisResult(
            symbol="ACME",
            bear=ThesisScenario(
                scenario="bear",
                probability=0.2,
                what_proves_wrong=["reason A", "reason B", "reason C"],
            ),
        ),
        catalyst_risk=CatalystRiskResult(
            symbol="ACME", risks=[RiskItem(description="risk X", severity=RiskSeverity.HIGH)]
        ),
    )
    triggers = _fallback_exit_triggers(report)
    assert len(triggers) <= 5
    assert any("60.00" in t for t in triggers)
