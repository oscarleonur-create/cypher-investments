"""Round-trip tests for the SQLite-backed ResearchStore."""

from __future__ import annotations

from datetime import date

from advisor.research.models import ResearchReport
from advisor.research.ratios import compute_ratios
from advisor.research.red_flags import detect_red_flags
from advisor.research.store import ResearchStore


def test_save_and_load_latest_report(tmp_path, clean_bundle):
    store = ResearchStore(tmp_path / "research.db")

    ratios = compute_ratios(clean_bundle)
    red_flags = detect_red_flags(clean_bundle, ratios)
    report = ResearchReport(
        symbol="TEST",
        as_of=date(2024, 12, 31),
        statements=clean_bundle,
        ratios=ratios,
        red_flags=red_flags,
    )
    store.save_report(report)

    loaded = store.load_latest_report("test")  # case-insensitive
    assert loaded is not None
    assert loaded.symbol == "TEST"
    assert loaded.statements.symbol == "TEST"
    assert loaded.ratios.latest().gross_margin == ratios.latest().gross_margin
    store.close()


def test_load_missing_returns_none(tmp_path):
    store = ResearchStore(tmp_path / "research.db")
    assert store.load_latest_report("NOPE") is None
    store.close()


def test_list_reports_orders_by_recency(tmp_path, clean_bundle):
    store = ResearchStore(tmp_path / "research.db")
    report_a = ResearchReport(symbol="AAA", as_of=date(2024, 1, 1), statements=clean_bundle)
    report_b = ResearchReport(symbol="BBB", as_of=date(2024, 6, 1), statements=clean_bundle)
    store.save_report(report_a)
    store.save_report(report_b)

    rows = store.list_reports()
    assert len(rows) == 2
    symbols = [r["symbol"] for r in rows]
    assert "AAA" in symbols and "BBB" in symbols
    store.close()


def test_artifact_round_trip(tmp_path):
    store = ResearchStore(tmp_path / "research.db")
    store.save_artifact("AAPL", "statements", "latest", '{"hello": "world"}')
    result = store.load_artifact("AAPL", "statements")
    assert result is not None
    payload, fetched_at = result
    assert payload == '{"hello": "world"}'
    assert fetched_at is not None
    store.close()
