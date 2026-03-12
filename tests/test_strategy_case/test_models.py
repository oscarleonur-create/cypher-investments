"""Tests for strategy case data models."""

from advisor.strategy_case.models import (
    CaseSynthesis,
    CaseVerdict,
    OptionsStrategyType,
    RiskProfile,
    ScenarioResult,
    ScenarioType,
    StrategyCase,
    StrategyCaseConfig,
    StrategyMatch,
    StrategyRanking,
    StrikeRecommendation,
)


class TestEnums:
    def test_scenario_types(self):
        assert ScenarioType.EARNINGS_DIP == "EARNINGS_DIP"
        assert ScenarioType.IV_SPIKE == "IV_SPIKE"
        assert len(ScenarioType) == 6

    def test_strategy_types(self):
        assert OptionsStrategyType.PUT_CREDIT_SPREAD == "put_credit_spread"
        assert len(OptionsStrategyType) == 7

    def test_verdict_values(self):
        assert CaseVerdict.STRONG == "STRONG"
        assert CaseVerdict.REJECT == "REJECT"


class TestScenarioResult:
    def test_defaults(self):
        r = ScenarioResult(scenario_type=ScenarioType.IV_SPIKE)
        assert r.confidence == 0.5
        assert r.price == 0.0

    def test_full(self):
        r = ScenarioResult(
            scenario_type=ScenarioType.EARNINGS_DIP,
            confidence=0.85,
            summary="AAPL dropped 8%",
            price=175.0,
            price_change_pct=-8.0,
            iv_percentile=72.0,
            pead_score="BUY",
        )
        assert r.scenario_type == ScenarioType.EARNINGS_DIP
        assert r.confidence == 0.85


class TestStrategyRanking:
    def test_ranking(self):
        m1 = StrategyMatch(
            strategy=OptionsStrategyType.PUT_CREDIT_SPREAD,
            fit_score=85,
            reasoning="Best fit",
        )
        m2 = StrategyMatch(
            strategy=OptionsStrategyType.NAKED_PUT,
            fit_score=70,
            reasoning="Runner up",
        )
        ranking = StrategyRanking(matches=[m1, m2], selected=m1)
        assert ranking.selected.strategy == OptionsStrategyType.PUT_CREDIT_SPREAD
        assert len(ranking.matches) == 2


class TestStrikeRecommendation:
    def test_defaults(self):
        r = StrikeRecommendation()
        assert r.strike == 0.0
        assert r.flags == []

    def test_spread(self):
        r = StrikeRecommendation(
            strategy="put_credit_spread",
            strike=170.0,
            long_strike=165.0,
            dte=30,
            credit=0.85,
            pop=0.72,
            sell_score=68.0,
        )
        assert r.long_strike == 165.0


class TestRiskProfile:
    def test_bsm_default(self):
        r = RiskProfile(source="bsm", pop=0.75, ev=12.50)
        assert r.source == "bsm"
        assert r.cvar_95 == 0.0

    def test_mc_full(self):
        r = RiskProfile(
            source="mc",
            pop=0.78,
            ev=15.30,
            cvar_95=-230.0,
            stop_prob=0.08,
        )
        assert r.source == "mc"
        assert r.stop_prob == 0.08


class TestStrategyCase:
    def test_empty_case(self):
        case = StrategyCase(symbol="AAPL")
        assert case.symbol == "AAPL"
        assert case.scenario is None
        assert case.errors == []

    def test_full_case(self):
        case = StrategyCase(
            symbol="AAPL",
            scenario=ScenarioResult(scenario_type=ScenarioType.IV_SPIKE),
            synthesis=CaseSynthesis(
                thesis_summary="Test thesis",
                verdict=CaseVerdict.MODERATE,
                conviction_score=62.0,
            ),
        )
        assert case.synthesis.verdict == CaseVerdict.MODERATE

    def test_json_roundtrip(self):
        case = StrategyCase(symbol="TSLA")
        data = case.model_dump()
        restored = StrategyCase.model_validate(data)
        assert restored.symbol == "TSLA"


class TestStrategyCaseConfig:
    def test_defaults(self):
        c = StrategyCaseConfig()
        assert c.account_size == 5_000.0
        assert c.enable_research is False
        assert c.enable_mc is False
        assert c.strategy_override is None
