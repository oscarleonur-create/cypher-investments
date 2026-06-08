"""Report orchestrator — runs all research layers in dependency order.

Layers run sequentially (later layers depend on earlier ones):
  1. statements  → ratios → red_flags
  2. peers → multiples → dcf (+ reverse-dcf)
  3. ecosystem (holders, insiders, customers, suppliers) → catalysts → risks
  4. industry → transcripts → market_data → thesis
  5. Assemble ResearchReport and cache to SQLite

Each layer is best-effort; failures are logged and the report is still
returned with whatever data was gathered.
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Callable

import yfinance as yf

from advisor.research.models import (
    CatalystRiskResult,
    EcosystemResult,
    ResearchReport,
    VariantPerceptionResult,
)

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, str], None]
"""Signature: progress(event, label). `event` is "started" | "done" | "failed"."""


def build_report(
    symbol: str,
    n_years: int = 5,
    explicit_peers: list[str] | None = None,
    force_refresh: bool = False,
    progress: ProgressCallback | None = None,
) -> ResearchReport:
    """Orchestrate all research layers and return a fully-assembled ResearchReport.

    If `progress` is provided, it is called as `progress(event, label)` around
    every layer; `event` is "started" | "done" | "failed".
    """
    sym = symbol.upper()
    from advisor.research.config import get_settings
    from advisor.research.store import ResearchStore

    settings = get_settings()
    store = ResearchStore(settings.db_path)

    if not force_refresh:
        cached = store.load_latest_report(sym)
        if cached is not None:
            logger.info("Returning cached report for %s", sym)
            if progress is not None:
                progress("done", f"cache-hit({sym})")
            return cached

    # Pull company metadata for downstream naming
    company_name, sector, industry = _company_meta(sym)

    report = ResearchReport(symbol=sym, as_of=date.today())

    # ── Layer 1: Statements / Ratios / Red Flags ──────────────────────────────
    bundle = _run(f"statements({sym})", lambda: _build_statements(sym, n_years), progress)
    report.statements = bundle

    if bundle:
        report.ratios = _run(
            f"ratios({sym})",
            lambda: __import__(
                "advisor.research.ratios", fromlist=["compute_ratios"]
            ).compute_ratios(bundle),
            progress,
        )
        report.red_flags = _run(
            f"red_flags({sym})",
            lambda: __import__(
                "advisor.research.red_flags", fromlist=["detect_red_flags"]
            ).detect_red_flags(bundle, report.ratios),
            progress,
        )

    # ── Layer 2: Peers / Multiples / DCF ─────────────────────────────────────
    peers = (
        _run(
            f"peers({sym})",
            lambda: __import__(
                "advisor.research.peers", fromlist=["build_peer_set"]
            ).build_peer_set(sym, explicit_peers=explicit_peers),
            progress,
        )
        or []
    )

    report.multiples = _run(
        f"multiples({sym})",
        lambda: __import__(
            "advisor.research.valuation.multiples", fromlist=["build_multiples"]
        ).build_multiples(sym, peers, statements=bundle),
        progress,
    )

    dcf = _run(
        f"dcf({sym})",
        lambda: __import__("advisor.research.valuation.dcf", fromlist=["build_dcf"]).build_dcf(
            sym, statements=bundle
        ),
        progress,
    )
    if dcf is not None:
        dcf.implied_growth_rate = _run(
            f"reverse_dcf({sym})",
            lambda: __import__(
                "advisor.research.valuation.reverse_dcf", fromlist=["solve_implied_growth"]
            ).solve_implied_growth(dcf),
            progress,
        )
    report.dcf = dcf

    # ── Layer 3: Ecosystem / Catalysts / Risks / Management Quality ──────────
    holders = _run(
        f"holders({sym})",
        lambda: __import__(
            "advisor.research.ecosystem.holders", fromlist=["get_holders"]
        ).get_holders(sym),
        progress,
    )
    insiders = _run(
        f"insiders({sym})",
        lambda: __import__(
            "advisor.research.ecosystem.insiders", fromlist=["get_insiders"]
        ).get_insiders(sym),
        progress,
    )
    customers = (
        _run(
            f"customers({sym})",
            lambda: __import__(
                "advisor.research.ecosystem.customers", fromlist=["get_customers"]
            ).get_customers(sym, company_name),
            progress,
        )
        or []
    )
    suppliers = (
        _run(
            f"suppliers({sym})",
            lambda: __import__(
                "advisor.research.ecosystem.suppliers", fromlist=["get_suppliers"]
            ).get_suppliers(sym, company_name),
            progress,
        )
        or []
    )

    report.ecosystem = EcosystemResult(
        symbol=sym,
        holders=holders,
        insiders=insiders,
        top_customers=customers,
        top_suppliers=suppliers,
    )

    report.management_quality = _run(
        f"management_quality({sym})",
        lambda: __import__(
            "advisor.research.management_quality", fromlist=["build_management_quality"]
        ).build_management_quality(sym, ratios=report.ratios, holders=holders, insiders=insiders),
        progress,
    )

    catalyst_result: CatalystRiskResult | None = _run(
        f"catalysts({sym})",
        lambda: __import__(
            "advisor.research.catalysts", fromlist=["build_catalysts"]
        ).build_catalysts(sym, company_name),
        progress,
    )
    report.catalyst_risk = _run(
        f"risks({sym})",
        lambda: __import__("advisor.research.risks", fromlist=["build_risks"]).build_risks(
            sym, company_name, report.red_flags, catalyst_result
        ),
        progress,
    )

    # ── Layer 3.5b: X / Twitter social sentiment ─────────────────────────────
    report.x_sentiment = _run(
        f"x_sentiment({sym})",
        lambda: __import__(
            "advisor.research.x_sentiment", fromlist=["build_x_sentiment"]
        ).build_x_sentiment(sym, company_name),
        progress,
    )

    # ── Layer 4: Industry / Transcripts / Market Data / Thesis ───────────────
    report.industry = _run(
        f"industry({sym})",
        lambda: __import__("advisor.research.industry", fromlist=["build_industry"]).build_industry(
            sym, company_name, sector, industry
        ),
        progress,
    )

    report.transcripts = _run(
        f"transcripts({sym})",
        lambda: __import__(
            "advisor.research.transcripts", fromlist=["build_transcripts"]
        ).build_transcripts(sym, company_name),
        progress,
    )

    report.market_data = _run(
        f"market_data({sym})",
        lambda: __import__(
            "advisor.research.market_data", fromlist=["get_market_data"]
        ).get_market_data(sym),
        progress,
    )

    # ── Layer 3.5: Consensus + Variant Perception ─────────────────────────────
    report.consensus = _run(
        f"consensus({sym})",
        lambda: __import__(
            "advisor.research.consensus", fromlist=["build_consensus"]
        ).build_consensus(sym),
        progress,
    )

    vp: VariantPerceptionResult | None = _run(
        f"variant_perception({sym})",
        lambda: __import__(
            "advisor.research.variant_perception",
            fromlist=["build_variant_perception"],
        ).build_variant_perception(
            sym,
            company_name,
            dcf=report.dcf,
            consensus=report.consensus,
            catalyst_risk=report.catalyst_risk,
            red_flags=report.red_flags,
            industry=report.industry,
        ),
        progress,
    )
    report.variant_perception = vp

    report.thesis = _run(
        f"thesis({sym})",
        lambda: __import__("advisor.research.thesis", fromlist=["build_thesis"]).build_thesis(
            sym,
            company_name,
            dcf=report.dcf,
            catalyst_risk=report.catalyst_risk,
            variant_perception=vp,
        ),
        progress,
    )

    report.business_model = company_name

    # ── Layer 5: Network map ──────────────────────────────────────────────────
    report.network = _run(
        f"network({sym})",
        lambda: __import__("advisor.research.network", fromlist=["build_network"]).build_network(
            sym, report
        ),
        progress,
    )

    # ── Layer 6: KPI Monitor (after thesis so KPIs can reference thesis context)
    report.kpi_monitor = _run(
        f"kpi_monitor({sym})",
        lambda: __import__(
            "advisor.research.kpi_tracker", fromlist=["build_thesis_monitor"]
        ).build_thesis_monitor(report),
        progress,
    )
    if report.kpi_monitor is not None:
        try:
            _save_kpi_watchlist(sym, report.kpi_monitor.kpis, store)
            store.save_kpi_check(sym, report.kpi_monitor)
        except Exception as exc:  # noqa: BLE001
            logger.warning("KPI persistence failed for %s: %s", sym, exc)

    # ── Layer 7: Options flow (live chain data — fast, no AI)
    report.options_flow = _run(
        f"options_flow({sym})",
        lambda: __import__(
            "advisor.research.options_flow", fromlist=["build_options_flow"]
        ).build_options_flow(sym),
        progress,
    )

    # ── Layer 7.5: Filing index (so the report can link/read full filings) ───
    report.filings = _run(f"filings({sym})", lambda: _list_filings(sym, n_years), progress) or []

    # ── Layer 8: Investment Memo (synthesis — runs last so all layers are ready)
    report.investment_memo = _run(
        f"investment_memo({sym})",
        lambda: __import__(
            "advisor.research.investment_memo", fromlist=["build_investment_memo"]
        ).build_investment_memo(report),
        progress,
    )

    store.save_report(report)
    return report


# ── Helpers ───────────────────────────────────────────────────────────────────


def _run(label: str, fn, progress: ProgressCallback | None = None):  # type: ignore[no-untyped-def]
    """Run a research layer; log and return None on any failure.

    Emits `progress(event, label)` around the call when `progress` is provided.
    """
    if progress is not None:
        progress("started", label)
    try:
        result = fn()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Layer %s failed: %s", label, exc)
        if progress is not None:
            progress("failed", label)
        return None
    if progress is not None:
        progress("done", label)
    return result


def _build_statements(symbol: str, n_years: int):  # type: ignore[no-untyped-def]
    from advisor.research.edgar import EdgarClient
    from advisor.research.statements import extract_statements

    edgar_client = None
    try:
        edgar_client = EdgarClient()
    except Exception:  # noqa: BLE001
        pass
    return extract_statements(symbol, edgar_client=edgar_client, n_years=n_years)


def _list_filings(symbol: str, n_years: int):  # type: ignore[no-untyped-def]
    """List recent SEC filings (10-K / 10-Q / 8-K / proxy) for citing + reading."""
    from advisor.research.edgar import EdgarClient
    from advisor.research.models import FormType

    client = EdgarClient()
    forms = [FormType.K10, FormType.Q10, FormType.K8, FormType.DEF14A]
    refs = []
    for form in forms:
        try:
            refs.extend(client.list_filings(symbol, form, lookback_years=n_years))
        except Exception as exc:  # noqa: BLE001
            logger.warning("Listing %s filings for %s failed: %s", form.value, symbol, exc)
    refs.sort(key=lambda f: f.filing_date, reverse=True)
    return refs


def _save_kpi_watchlist(symbol: str, kpis: list, store) -> None:  # type: ignore[no-untyped-def]
    store.save_kpi_watchlist(symbol, kpis)


def _company_meta(symbol: str) -> tuple[str, str, str]:
    """Return (shortName, sector, industry) from yfinance; empty strings on failure."""
    try:
        info = yf.Ticker(symbol).info or {}
        return (
            info.get("shortName", symbol),
            info.get("sector", ""),
            info.get("industry", ""),
        )
    except Exception:  # noqa: BLE001
        return symbol, "", ""
