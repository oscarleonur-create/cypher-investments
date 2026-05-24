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

import yfinance as yf

from advisor.research.models import (
    CatalystRiskResult,
    EcosystemResult,
    ResearchReport,
)

logger = logging.getLogger(__name__)


def build_report(
    symbol: str,
    n_years: int = 5,
    explicit_peers: list[str] | None = None,
    force_refresh: bool = False,
) -> ResearchReport:
    """Orchestrate all research layers and return a fully-assembled ResearchReport."""
    sym = symbol.upper()
    from advisor.research.config import get_settings
    from advisor.research.store import ResearchStore

    settings = get_settings()
    store = ResearchStore(settings.db_path)

    if not force_refresh:
        cached = store.load_latest_report(sym)
        if cached is not None:
            logger.info("Returning cached report for %s", sym)
            return cached

    # Pull company metadata for downstream naming
    company_name, sector, industry = _company_meta(sym)

    report = ResearchReport(symbol=sym, as_of=date.today())

    # ── Layer 1: Statements / Ratios / Red Flags ──────────────────────────────
    bundle = _run(f"statements({sym})", lambda: _build_statements(sym, n_years))
    report.statements = bundle

    if bundle:
        report.ratios = _run(
            f"ratios({sym})",
            lambda: __import__(
                "advisor.research.ratios", fromlist=["compute_ratios"]
            ).compute_ratios(bundle),
        )
        report.red_flags = _run(
            f"red_flags({sym})",
            lambda: __import__(
                "advisor.research.red_flags", fromlist=["detect_red_flags"]
            ).detect_red_flags(bundle, report.ratios),
        )

    # ── Layer 2: Peers / Multiples / DCF ─────────────────────────────────────
    peers = (
        _run(
            f"peers({sym})",
            lambda: __import__(
                "advisor.research.peers", fromlist=["build_peer_set"]
            ).build_peer_set(sym, explicit_peers=explicit_peers),
        )
        or []
    )

    report.multiples = _run(
        f"multiples({sym})",
        lambda: __import__(
            "advisor.research.valuation.multiples", fromlist=["build_multiples"]
        ).build_multiples(sym, peers, statements=bundle),
    )

    dcf = _run(
        f"dcf({sym})",
        lambda: __import__("advisor.research.valuation.dcf", fromlist=["build_dcf"]).build_dcf(
            sym, statements=bundle
        ),
    )
    if dcf is not None:
        dcf.implied_growth_rate = _run(
            f"reverse_dcf({sym})",
            lambda: __import__(
                "advisor.research.valuation.reverse_dcf", fromlist=["solve_implied_growth"]
            ).solve_implied_growth(dcf),
        )
    report.dcf = dcf

    # ── Layer 3: Ecosystem / Catalysts / Risks ────────────────────────────────
    holders = _run(
        f"holders({sym})",
        lambda: __import__(
            "advisor.research.ecosystem.holders", fromlist=["get_holders"]
        ).get_holders(sym),
    )
    insiders = _run(
        f"insiders({sym})",
        lambda: __import__(
            "advisor.research.ecosystem.insiders", fromlist=["get_insiders"]
        ).get_insiders(sym),
    )
    customers = (
        _run(
            f"customers({sym})",
            lambda: __import__(
                "advisor.research.ecosystem.customers", fromlist=["get_customers"]
            ).get_customers(sym, company_name),
        )
        or []
    )
    suppliers = (
        _run(
            f"suppliers({sym})",
            lambda: __import__(
                "advisor.research.ecosystem.suppliers", fromlist=["get_suppliers"]
            ).get_suppliers(sym, company_name),
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

    catalyst_result: CatalystRiskResult | None = _run(
        f"catalysts({sym})",
        lambda: __import__(
            "advisor.research.catalysts", fromlist=["build_catalysts"]
        ).build_catalysts(sym, company_name),
    )
    report.catalyst_risk = _run(
        f"risks({sym})",
        lambda: __import__("advisor.research.risks", fromlist=["build_risks"]).build_risks(
            sym, company_name, report.red_flags, catalyst_result
        ),
    )

    # ── Layer 4: Industry / Transcripts / Market Data / Thesis ───────────────
    report.industry = _run(
        f"industry({sym})",
        lambda: __import__("advisor.research.industry", fromlist=["build_industry"]).build_industry(
            sym, company_name, sector, industry
        ),
    )

    report.transcripts = _run(
        f"transcripts({sym})",
        lambda: __import__(
            "advisor.research.transcripts", fromlist=["build_transcripts"]
        ).build_transcripts(sym, company_name),
    )

    report.market_data = _run(
        f"market_data({sym})",
        lambda: __import__(
            "advisor.research.market_data", fromlist=["get_market_data"]
        ).get_market_data(sym),
    )

    report.thesis = _run(
        f"thesis({sym})",
        lambda: __import__("advisor.research.thesis", fromlist=["build_thesis"]).build_thesis(
            sym, company_name, dcf=report.dcf, catalyst_risk=report.catalyst_risk
        ),
    )

    report.business_model = company_name

    store.save_report(report)
    return report


# ── Helpers ───────────────────────────────────────────────────────────────────


def _run(label: str, fn):  # type: ignore[no-untyped-def]
    """Run a research layer; log and return None on any failure."""
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        logger.warning("Layer %s failed: %s", label, exc)
        return None


def _build_statements(symbol: str, n_years: int):  # type: ignore[no-untyped-def]
    from advisor.research.edgar import EdgarClient
    from advisor.research.statements import extract_statements

    edgar_client = None
    try:
        edgar_client = EdgarClient()
    except Exception:  # noqa: BLE001
        pass
    return extract_statements(symbol, edgar_client=edgar_client, n_years=n_years)


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
