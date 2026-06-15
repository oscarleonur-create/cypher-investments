"""Phase 3 tests — industry, transcripts, market_data, thesis, render, report."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from unittest.mock import patch

from advisor.research.models import (
    DcfAssumptions,
    DcfResult,
    DcfScenario,
    IndustryAnalysis,
    MoatType,
    PortersForces,
    PortersForceStrength,
    ResearchReport,
    ThesisResult,
    ThesisScenario,
    TranscriptSummary,
    TranscriptTone,
)

# ── Industry ──────────────────────────────────────────────────────────────────


def test_industry_stub_no_llm():
    """build_industry returns a valid stub when LLM keys are absent."""
    from advisor.research.industry import build_industry

    with patch("research_agent.config.ResearchConfig") as mock_cfg:
        mock_cfg.return_value.openrouter_api_key = ""
        mock_cfg.return_value.tavily_api_key = ""
        result = build_industry("AAPL", "Apple Inc", "Technology", "Consumer Electronics")

    assert result.symbol == "AAPL"
    assert result.moat_type == MoatType.NONE
    assert result.sector == "Technology"


def test_industry_model_round_trip():
    """IndustryAnalysis serialises and deserialises cleanly."""
    ind = IndustryAnalysis(
        symbol="MSFT",
        sector="Technology",
        industry="Software",
        porters_forces=PortersForces(
            competitive_rivalry=PortersForceStrength.HIGH,
            supplier_power=PortersForceStrength.LOW,
            buyer_power=PortersForceStrength.MEDIUM,
            threat_of_new_entrants=PortersForceStrength.MEDIUM,
            threat_of_substitutes=PortersForceStrength.LOW,
            rationale="Network effect protects installed base.",
        ),
        moat_type=MoatType.SWITCHING_COSTS,
        moat_description="High switching costs from Azure integration.",
        key_competitors=["GOOGL", "AMZN"],
    )
    data = ind.model_dump(mode="json")
    restored = IndustryAnalysis.model_validate(data)
    assert restored.moat_type == MoatType.SWITCHING_COSTS
    assert restored.porters_forces.competitive_rivalry == PortersForceStrength.HIGH


# ── Transcripts ───────────────────────────────────────────────────────────────


def test_transcript_model():
    """TranscriptSummary stores tone correctly."""
    ts = TranscriptSummary(
        quarter="Q1 2025",
        tone=TranscriptTone.BULLISH,
        key_topics=["AI growth", "cloud expansion", "margins"],
        management_guidance="Revenue up 15-18% YoY.",
        analyst_concerns="Capital expenditure trajectory.",
        highlight_quote="AI is the defining opportunity of our generation.",
    )
    assert ts.tone == TranscriptTone.BULLISH
    assert len(ts.key_topics) == 3


def test_transcript_analysis_stub_no_llm():
    """build_transcripts returns empty summaries when LLM keys are missing."""
    from advisor.research.transcripts import build_transcripts

    with patch("research_agent.config.ResearchConfig") as mock_cfg:
        mock_cfg.return_value.openrouter_api_key = ""
        mock_cfg.return_value.tavily_api_key = ""
        result = build_transcripts("NVDA", "NVIDIA")

    assert result.symbol == "NVDA"
    assert result.summaries == []


# ── Market data ───────────────────────────────────────────────────────────────


def test_market_data_yfinance():
    """get_market_data populates fields from mocked yfinance info."""
    from advisor.research.market_data import get_market_data

    mock_info = {
        "floatShares": 15_000_000_000,
        "sharesOutstanding": 15_200_000_000,
        "shortPercentOfFloat": 0.012,  # 1.2%
        "shortRatio": 2.5,
        "averageVolume": 60_000_000,
        "averageVolume10days": 58_000_000,
    }

    with patch("yfinance.Ticker") as mock_ticker:
        mock_ticker.return_value.info = mock_info
        snap = get_market_data("AAPL")

    assert snap.symbol == "AAPL"
    assert snap.float_shares == 15_000_000_000
    assert abs(snap.short_interest_pct - 1.2) < 0.01
    assert snap.days_to_cover == 2.5


def test_market_data_yfinance_failure():
    """get_market_data returns empty snapshot on yfinance failure."""
    from advisor.research.market_data import get_market_data

    with patch("yfinance.Ticker", side_effect=RuntimeError("network error")):
        snap = get_market_data("FAIL")

    assert snap.symbol == "FAIL"
    assert snap.short_interest_pct is None


# ── Thesis ────────────────────────────────────────────────────────────────────


def _make_dcf(current_price: float = 100.0) -> DcfResult:
    def _scenario(name: str, implied: float) -> DcfScenario:
        return DcfScenario(
            assumptions=DcfAssumptions(
                scenario=name,
                revenue_growth_yr1_3=0.10,
                revenue_growth_yr4_10=0.06,
                target_fcf_margin=0.15,
                capex_intensity=0.05,
                wacc=0.09,
            ),
            projected_fcf=[10.0] * 10,
            enterprise_value=1000.0,
            equity_value=900.0,
            implied_price=implied,
            upside_pct=(implied / current_price) - 1,
        )

    return DcfResult(
        symbol="TEST",
        current_price=current_price,
        shares_outstanding=100e6,
        net_debt=0.0,
        wacc=0.09,
        risk_free_rate=0.043,
        base=_scenario("base", 110.0),
        bull=_scenario("bull", 150.0),
        bear=_scenario("bear", 70.0),
    )


def test_thesis_price_anchoring():
    """build_thesis anchors target prices to DCF implied prices."""
    from advisor.research.thesis import build_thesis

    with patch("research_agent.config.ResearchConfig") as mock_cfg:
        mock_cfg.return_value.openrouter_api_key = ""
        dcf = _make_dcf(100.0)
        result = build_thesis("TEST", dcf=dcf)

    assert result.symbol == "TEST"
    assert result.current_price == 100.0
    assert result.base is not None and abs(result.base.target_price - 110.0) < 0.01
    assert result.bull is not None and abs(result.bull.target_price - 150.0) < 0.01
    assert result.bear is not None and abs(result.bear.target_price - 70.0) < 0.01
    # Upside sanity
    assert abs(result.base.upside_pct - 0.10) < 0.01
    assert result.bull.upside_pct > 0
    assert result.bear.upside_pct < 0


def test_thesis_probabilities_default():
    """Default scenario probabilities are in valid range and sum to ~1."""
    from advisor.research.thesis import build_thesis

    with patch("research_agent.config.ResearchConfig") as mock_cfg:
        mock_cfg.return_value.openrouter_api_key = ""
        dcf = _make_dcf()
        result = build_thesis("TEST", dcf=dcf)

    probs = [s.probability for s in [result.bull, result.base, result.bear] if s]
    assert abs(sum(probs) - 1.0) < 0.01
    assert all(0 <= p <= 1 for p in probs)


# ── Render ────────────────────────────────────────────────────────────────────


def _minimal_report() -> ResearchReport:
    return ResearchReport(
        symbol="TST",
        as_of=date.today(),
        business_model="Test Corp",
        thesis=ThesisResult(
            symbol="TST",
            current_price=100.0,
            bull=ThesisScenario(
                scenario="bull",
                probability=0.25,
                target_price=150.0,
                upside_pct=0.50,
                description="Strong AI tailwinds.",
                key_assumptions=["AI revenue doubles"],
                what_proves_wrong=["Competition intensifies"],
            ),
            base=ThesisScenario(
                scenario="base",
                probability=0.50,
                target_price=110.0,
                upside_pct=0.10,
                description="Steady execution.",
                key_assumptions=["Revenue grows 10%"],
                what_proves_wrong=["Margin compression"],
            ),
            bear=ThesisScenario(
                scenario="bear",
                probability=0.25,
                target_price=70.0,
                upside_pct=-0.30,
                description="Macro headwinds.",
            ),
            summary="Balanced risk/reward.",
            conviction="MEDIUM",
        ),
    )


def test_render_produces_markdown_sections():
    """render_report returns a string with all 7 section headers."""
    from advisor.research.render import render_report

    report = _minimal_report()
    md = render_report(report)

    assert "# Research Report: TST" in md
    assert "§1 Business" in md
    assert "§2 Ecosystem" in md
    assert "§3 Industry" in md
    assert "§4 Financial Analysis" in md
    assert "§5 Valuation" in md
    assert "§6 Catalysts" in md
    assert "§7 Investment Thesis" in md


def test_render_thesis_prices():
    """Rendered thesis contains target prices from ThesisScenario."""
    from advisor.research.render import render_report

    report = _minimal_report()
    md = render_report(report)

    assert "$150.00" in md
    assert "$110.00" in md
    assert "$70.00" in md
    assert "+50.0%" in md


# ── Report orchestrator (integration-light) ───────────────────────────────────


def test_build_report_cache_round_trip(tmp_path: Path):
    """build_report saves to SQLite and load_latest_report returns the same report."""
    from advisor.research.report import build_report
    from advisor.research.store import ResearchStore

    db = tmp_path / "research.db"

    # Blank the research-agent API keys so every Tavily/LLM-backed layer hits its
    # offline guard and returns empty — same path keyless CI takes. Without this,
    # a developer machine with real keys in .env makes live LLM calls here and hangs.
    from research_agent.config import ResearchConfig

    offline_cfg = ResearchConfig(openrouter_api_key="", tavily_api_key="")

    # Patch every expensive layer so the test runs offline and fast
    with (
        patch("advisor.research.report._build_statements", return_value=None),
        patch("advisor.research.report._company_meta", return_value=("FakeCo", "Tech", "SW")),
        patch("advisor.research.config.get_settings") as mock_settings,
        patch("research_agent.config.ResearchConfig", return_value=offline_cfg),
    ):
        mock_settings.return_value.db_path = db
        report = build_report("FAKE", n_years=1, force_refresh=True)

    assert report.symbol == "FAKE"
    loaded = ResearchStore(db).load_latest_report("FAKE")
    assert loaded is not None
    assert loaded.symbol == "FAKE"


def test_research_notes_from_cache(tmp_path: Path):
    """check_fundamental with use_research=True surfaces cached report notes."""
    from advisor.confluence.fundamental import check_fundamental
    from advisor.research.store import ResearchStore

    db = tmp_path / "research.db"
    store = ResearchStore(db)

    # Build a minimal report with ratios
    from advisor.research.models import (
        RatioBundle,
        RatioPeriod,
    )

    ratios = RatioBundle(
        symbol="TSLA",
        periods=[
            RatioPeriod(
                period_end=date.today(),
                fiscal_year=2024,
                roic=0.18,
                debt_to_ebitda=1.5,
                fcf_margin=0.08,
            )
        ],
    )
    rpt = ResearchReport(symbol="TSLA", as_of=date.today(), ratios=ratios)
    store.save_report(rpt)

    with (
        patch("advisor.research.config.get_settings") as mock_cfg,
        patch("yfinance.Ticker") as mock_ticker,
    ):
        mock_cfg.return_value.db_path = db
        mock_ticker.return_value.info = {}
        mock_ticker.return_value.calendar = None
        mock_ticker.return_value.insider_transactions = None
        result = check_fundamental("TSLA", use_research=True)

    assert any("ROIC" in n for n in result.research_notes)
    assert any("D/EBITDA" in n for n in result.research_notes)
