"""Tests for the pipeline builder — orchestrator integration."""

from datetime import date
from unittest.mock import patch

from advisor.strategy_case.builder import StrategyCaseBuilder
from advisor.strategy_case.models import (
    CaseSynthesis,
    CaseVerdict,
    OptionsAnalysisResult,
    OptionsStrategyType,
    RiskProfile,
    ScenarioResult,
    ScenarioType,
    StrategyMatch,
    StrategyRanking,
    StrikeRecommendation,
)


def _make_scenario():
    return ScenarioResult(
        scenario_type=ScenarioType.IV_SPIKE,
        confidence=0.75,
        summary="AAPL IV at 80th percentile",
        iv_percentile=80,
        price=175.0,
    )


def _make_ranking():
    m = StrategyMatch(
        strategy=OptionsStrategyType.PUT_CREDIT_SPREAD,
        fit_score=85,
        reasoning="Sell rich premium",
    )
    return StrategyRanking(matches=[m], selected=m)


def _make_options():
    return OptionsAnalysisResult(
        recommendations=[
            StrikeRecommendation(
                strategy="put_credit_spread",
                strike=170,
                long_strike=165,
                expiry=date(2026, 4, 17),
                dte=30,
                credit=0.85,
                delta=0.25,
                pop=0.75,
                sell_score=68,
                max_loss=415,
                margin_req=415,
            )
        ],
        iv_percentile=80,
    )


def _make_risk():
    return RiskProfile(
        source="bsm",
        pop=0.75,
        ev=12.50,
        suggested_contracts=1,
        sizing_feasible=True,
        max_loss_total=415,
        risk_pct=8.3,
    )


def _make_synthesis():
    return CaseSynthesis(
        thesis_summary="Test thesis for AAPL",
        verdict=CaseVerdict.MODERATE,
        conviction_score=62.0,
    )


class TestStrategyCaseBuilder:
    @patch("advisor.strategy_case.synthesis.synthesize_case")
    @patch("advisor.strategy_case.risk_assessment.assess_risk_bsm")
    @patch("advisor.strategy_case.options_analysis.analyze_options")
    @patch("advisor.strategy_case.strategy_mapper.rank_strategies")
    @patch("advisor.strategy_case.scenarios.detect_scenario")
    def test_full_pipeline(self, mock_detect, mock_rank, mock_options, mock_risk, mock_synth):
        mock_detect.return_value = _make_scenario()
        mock_rank.return_value = _make_ranking()
        mock_options.return_value = _make_options()
        mock_risk.return_value = _make_risk()
        mock_synth.return_value = _make_synthesis()

        builder = StrategyCaseBuilder()
        case = builder.build("AAPL")

        assert case.symbol == "AAPL"
        assert case.scenario.scenario_type == ScenarioType.IV_SPIKE
        assert case.ranking.selected.strategy == OptionsStrategyType.PUT_CREDIT_SPREAD
        assert len(case.options.recommendations) == 1
        assert case.risk.pop == 0.75
        assert case.synthesis.verdict == CaseVerdict.MODERATE
        assert case.elapsed_seconds >= 0

    @patch("advisor.strategy_case.scenarios.detect_scenario")
    def test_scenario_failure_aborts(self, mock_detect):
        mock_detect.side_effect = RuntimeError("API down")

        builder = StrategyCaseBuilder()
        case = builder.build("AAPL")

        assert "Scenario detection failed" in case.errors[0]
        assert case.ranking is None

    @patch("advisor.strategy_case.synthesis.synthesize_case")
    @patch("advisor.strategy_case.risk_assessment.assess_risk_bsm")
    @patch("advisor.strategy_case.options_analysis.analyze_options")
    @patch("advisor.strategy_case.strategy_mapper.rank_strategies")
    @patch("advisor.strategy_case.scenarios.detect_scenario")
    def test_research_off_by_default(
        self, mock_detect, mock_rank, mock_options, mock_risk, mock_synth
    ):
        mock_detect.return_value = _make_scenario()
        mock_rank.return_value = _make_ranking()
        mock_options.return_value = _make_options()
        mock_risk.return_value = _make_risk()
        mock_synth.return_value = _make_synthesis()

        builder = StrategyCaseBuilder()
        case = builder.build("AAPL")

        assert case.research is None

    @patch("advisor.strategy_case.synthesis.synthesize_case")
    @patch("advisor.strategy_case.risk_assessment.assess_risk_bsm")
    @patch("advisor.strategy_case.options_analysis.analyze_options")
    @patch("advisor.strategy_case.strategy_mapper.rank_strategies")
    @patch("advisor.strategy_case.scenarios.detect_scenario")
    def test_progress_callback(self, mock_detect, mock_rank, mock_options, mock_risk, mock_synth):
        mock_detect.return_value = _make_scenario()
        mock_rank.return_value = _make_ranking()
        mock_options.return_value = _make_options()
        mock_risk.return_value = _make_risk()
        mock_synth.return_value = _make_synthesis()

        messages: list[str] = []
        builder = StrategyCaseBuilder(progress_callback=messages.append)
        builder.build("AAPL")

        assert any("Stage 1" in m for m in messages)
        assert any("Stage 2" in m for m in messages)
        assert any("Stage 4" in m for m in messages)
        assert any("Stage 6" in m for m in messages)

    @patch("advisor.strategy_case.synthesis.synthesize_case")
    @patch("advisor.strategy_case.risk_assessment.assess_risk_bsm")
    @patch("advisor.strategy_case.options_analysis.analyze_options")
    @patch("advisor.strategy_case.strategy_mapper.rank_strategies")
    @patch("advisor.strategy_case.scenarios.detect_scenario")
    def test_json_roundtrip(self, mock_detect, mock_rank, mock_options, mock_risk, mock_synth):
        mock_detect.return_value = _make_scenario()
        mock_rank.return_value = _make_ranking()
        mock_options.return_value = _make_options()
        mock_risk.return_value = _make_risk()
        mock_synth.return_value = _make_synthesis()

        builder = StrategyCaseBuilder()
        case = builder.build("AAPL")

        data = case.model_dump()
        assert data["symbol"] == "AAPL"
        assert data["synthesis"]["verdict"] == "MODERATE"
