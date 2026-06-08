"""Pydantic round-trip tests for the research models."""

from __future__ import annotations

from datetime import date

from advisor.research.models import (
    FilingRef,
    FormType,
    RatioPeriod,
    RedFlag,
    RedFlagList,
    RedFlagSeverity,
    ResearchReport,
)


def test_filing_ref_serialises():
    ref = FilingRef(
        accession_number="0000320193-24-000123",
        form=FormType.K10,
        filing_date=date(2024, 11, 1),
        period_of_report=date(2024, 9, 28),
        url="https://www.sec.gov/Archives/edgar/data/320193/...",
    )
    raw = ref.model_dump_json()
    loaded = FilingRef.model_validate_json(raw)
    assert loaded == ref


def test_research_report_round_trip(clean_bundle):
    from advisor.research.ratios import compute_ratios

    ratios = compute_ratios(clean_bundle)
    report = ResearchReport(
        symbol="TEST",
        as_of=date(2024, 12, 31),
        statements=clean_bundle,
        ratios=ratios,
        red_flags=RedFlagList(
            symbol="TEST",
            flags=[
                RedFlag(
                    code="X",
                    title="Test flag",
                    severity=RedFlagSeverity.LOW,
                    detail="example",
                )
            ],
        ),
        notes=["fetched from synthetic fixture"],
    )

    raw = report.model_dump_json()
    loaded = ResearchReport.model_validate_json(raw)
    assert loaded.symbol == "TEST"
    assert loaded.statements.income[0].revenue == 1000.0
    assert isinstance(loaded.ratios.latest(), RatioPeriod)
    assert loaded.red_flags.flags[0].severity == RedFlagSeverity.LOW
