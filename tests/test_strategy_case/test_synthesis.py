"""Tests for case synthesis — Stage 6."""

from advisor.strategy_case.models import (
    CaseVerdict,
    OptionsAnalysisResult,
    OptionsStrategyType,
    ResearchSummary,
    RiskProfile,
    ScenarioResult,
    ScenarioType,
    StrategyMatch,
    StrategyRanking,
    StrikeRecommendation,
)
from advisor.strategy_case.synthesis import _compute_conviction, _fallback_synthesis


def _make_scenario(confidence: float = 0.8, iv_pctile: float = 65) -> ScenarioResult:
    return ScenarioResult(
        scenario_type=ScenarioType.IV_SPIKE,
        confidence=confidence,
        summary="Test scenario",
        iv_percentile=iv_pctile,
    )


def _make_ranking(fit_score: float = 80) -> StrategyRanking:
    m = StrategyMatch(
        strategy=OptionsStrategyType.PUT_CREDIT_SPREAD,
        fit_score=fit_score,
        reasoning="Good fit",
    )
    return StrategyRanking(matches=[m], selected=m)


def _make_options(sell_score: float = 70, pop: float = 0.75) -> OptionsAnalysisResult:
    return OptionsAnalysisResult(
        recommendations=[
            StrikeRecommendation(
                strategy="put_credit_spread",
                strike=170,
                long_strike=165,
                dte=30,
                credit=0.85,
                pop=pop,
                sell_score=sell_score,
                max_loss=415,
                margin_req=415,
            )
        ]
    )


def _make_risk(pop: float = 0.78, ev: float = 15.0) -> RiskProfile:
    return RiskProfile(
        source="bsm",
        pop=pop,
        ev=ev,
        sizing_feasible=True,
        suggested_contracts=1,
    )


class TestConvictionScoring:
    def test_high_conviction(self):
        score = _compute_conviction(
            _make_scenario(confidence=0.9),
            _make_ranking(fit_score=90),
            _make_options(sell_score=80, pop=0.80),
            _make_risk(pop=0.85, ev=25),
        )
        assert score >= 60

    def test_low_conviction(self):
        score = _compute_conviction(
            _make_scenario(confidence=0.3),
            _make_ranking(fit_score=40),
            _make_options(sell_score=30, pop=0.55),
            _make_risk(pop=0.55, ev=-10),
        )
        assert score < 50

    def test_no_research_redistributes(self):
        # With research
        with_research = _compute_conviction(
            _make_scenario(),
            _make_ranking(),
            _make_options(),
            _make_risk(),
            research=ResearchSummary(verdict="BUY_THE_DIP", grounding_score=0.9),
        )
        # Without research
        without_research = _compute_conviction(
            _make_scenario(),
            _make_ranking(),
            _make_options(),
            _make_risk(),
            research=None,
        )
        # Both should produce a score (redistribution works)
        assert with_research > 0
        assert without_research > 0

    def test_no_risk_uses_bsm_pop(self):
        score = _compute_conviction(
            _make_scenario(),
            _make_ranking(),
            _make_options(pop=0.80),
            risk=None,
        )
        assert score > 0

    def test_infeasible_sizing_penalized(self):
        feasible = _compute_conviction(
            _make_scenario(),
            _make_ranking(),
            _make_options(),
            RiskProfile(source="bsm", pop=0.75, ev=10, sizing_feasible=True),
        )
        infeasible = _compute_conviction(
            _make_scenario(),
            _make_ranking(),
            _make_options(),
            RiskProfile(source="bsm", pop=0.75, ev=10, sizing_feasible=False),
        )
        assert feasible > infeasible

    def test_score_clamped_to_100(self):
        score = _compute_conviction(
            _make_scenario(confidence=1.0),
            _make_ranking(fit_score=100),
            _make_options(sell_score=100, pop=0.95),
            _make_risk(pop=0.95, ev=100),
            research=ResearchSummary(verdict="BUY_THE_DIP", grounding_score=1.0),
        )
        assert score <= 100


class TestFallbackSynthesis:
    def test_produces_valid_synthesis(self):
        result = _fallback_synthesis(
            _make_scenario(),
            _make_ranking(),
            _make_options(),
            _make_risk(),
        )
        assert result.thesis_summary != ""
        assert result.verdict in CaseVerdict
        assert 0 <= result.conviction_score <= 100

    def test_no_strikes(self):
        result = _fallback_synthesis(
            _make_scenario(),
            _make_ranking(),
            OptionsAnalysisResult(),
            None,
        )
        assert "No qualifying strikes" in result.thesis_summary
