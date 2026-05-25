"""Data-access wrappers between the dashboard and the research engine.

A thin layer over `advisor.research.report.build_report` so pages don't import
the orchestrator directly. Keeps the UI code free of pipeline internals and
gives one place to add UI-level caching later.
"""

from __future__ import annotations

import logging
import sqlite3
from typing import Callable

import pandas as pd
import streamlit as st

from advisor.research.config import get_settings
from advisor.research.models import ResearchReport
from advisor.research.report import build_report
from advisor.research.store import ResearchStore

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[str, str], None]


def load_report(
    symbol: str,
    n_years: int = 5,
    explicit_peers: list[str] | None = None,
    force_refresh: bool = False,
    progress: ProgressCallback | None = None,
) -> ResearchReport:
    """Build (or load cached) ResearchReport for `symbol`.

    `progress(event, label)` is forwarded into the orchestrator so the page
    can stream layer-by-layer status to `st.status(...)`.
    """
    return build_report(
        symbol=symbol,
        n_years=n_years,
        explicit_peers=explicit_peers,
        force_refresh=force_refresh,
        progress=progress,
    )


def list_cached_reports(limit: int = 50) -> list[dict]:
    """Return the most recently saved reports from the research store."""
    store = ResearchStore(get_settings().db_path)
    try:
        return store.list_reports(limit=limit)
    finally:
        store.close()


def load_latest_cached(symbol: str) -> ResearchReport | None:
    """Return the latest cached ResearchReport for `symbol`, or None."""
    store = ResearchStore(get_settings().db_path)
    try:
        return store.load_latest_report(symbol.upper())
    finally:
        store.close()


def is_report_empty(report: ResearchReport) -> bool:
    """Return True if no data source returned anything for this ticker.

    Typically means the symbol is invalid: every layer succeeded (no
    exceptions) but every payload is None or empty. Used by the dashboard to
    flag bad tickers instead of rendering a page full of em-dashes.
    """
    has_statements = bool(report.statements and report.statements.income)
    has_ratios = bool(report.ratios and report.ratios.periods)
    has_multiples = bool(
        report.multiples
        and report.multiples.subject
        and (report.multiples.subject.market_cap or report.multiples.subject.price)
    )
    has_market = bool(
        report.market_data
        and (report.market_data.shares_outstanding or report.market_data.float_shares)
    )
    has_dcf = bool(report.dcf and report.dcf.current_price)
    return not (has_statements or has_ratios or has_multiples or has_market or has_dcf)


@st.cache_data(ttl=3600, show_spinner="Fetching quarterly statements…")
def load_quarterly_statements(symbol: str) -> dict[str, pd.DataFrame]:
    """Pull quarterly statements via yfinance, returned as a dict of DataFrames.

    Keys: 'income', 'balance', 'cashflow'. Columns are period-end timestamps
    (most-recent-first). Returns empty DataFrames on any failure — the UI
    can detect emptiness with `df.empty`.
    """
    import yfinance as yf

    sym = symbol.upper()
    try:
        ticker = yf.Ticker(sym)
        return {
            "income": _safe_frame(ticker, "quarterly_income_stmt"),
            "balance": _safe_frame(ticker, "quarterly_balance_sheet"),
            "cashflow": _safe_frame(ticker, "quarterly_cashflow"),
        }
    except Exception as exc:  # noqa: BLE001
        logger.warning("Quarterly fetch failed for %s: %s", sym, exc)
        empty = pd.DataFrame()
        return {"income": empty, "balance": empty, "cashflow": empty}


def _safe_frame(ticker, attr: str) -> pd.DataFrame:  # type: ignore[no-untyped-def]
    try:
        df = getattr(ticker, attr, None)
        if df is None:
            return pd.DataFrame()
        return df
    except Exception as exc:  # noqa: BLE001
        logger.warning("yfinance %s failed: %s", attr, exc)
        return pd.DataFrame()


def refresh_layer(
    report: ResearchReport,
    layer: str,
    *,
    n_years: int = 5,
    explicit_peers: list[str] | None = None,
) -> ResearchReport:
    """Re-run one specific layer and return the updated report.

    Supported `layer` values: 'statements', 'multiples', 'ecosystem',
    'catalysts'. The orchestrator already ran every layer once; this calls
    the underlying builders directly to avoid rebuilding everything just to
    update one tab.
    """
    sym = report.symbol.upper()
    layer = layer.lower()

    if layer == "statements":
        from advisor.research.edgar import EdgarClient
        from advisor.research.ratios import compute_ratios
        from advisor.research.red_flags import detect_red_flags
        from advisor.research.statements import extract_statements

        try:
            edgar = EdgarClient()
        except Exception:  # noqa: BLE001
            edgar = None
        bundle = extract_statements(sym, edgar_client=edgar, n_years=n_years)
        report.statements = bundle
        if bundle and bundle.income:
            report.ratios = compute_ratios(bundle)
            report.red_flags = detect_red_flags(bundle, report.ratios)

    elif layer == "multiples":
        from advisor.research.peers import build_peer_set
        from advisor.research.valuation.multiples import build_multiples

        peers = build_peer_set(sym, explicit_peers=explicit_peers)
        report.multiples = build_multiples(sym, peers, statements=report.statements)

    elif layer == "ecosystem":
        from advisor.research.ecosystem.customers import get_customers
        from advisor.research.ecosystem.holders import get_holders
        from advisor.research.ecosystem.insiders import get_insiders
        from advisor.research.ecosystem.suppliers import get_suppliers
        from advisor.research.models import EcosystemResult

        company = report.business_model or sym
        report.ecosystem = EcosystemResult(
            symbol=sym,
            holders=get_holders(sym),
            insiders=get_insiders(sym),
            top_customers=get_customers(sym, company) or [],
            top_suppliers=get_suppliers(sym, company) or [],
        )

    elif layer == "catalysts":
        from advisor.research.catalysts import build_catalysts
        from advisor.research.risks import build_risks

        company = report.business_model or sym
        catalysts = build_catalysts(sym, company)
        report.catalyst_risk = build_risks(sym, company, report.red_flags, catalysts)

    elif layer == "transcripts":
        from advisor.research.transcripts import build_transcripts

        company = report.business_model or sym
        report.transcripts = build_transcripts(sym, company)

    elif layer == "edge":
        from advisor.research.consensus import build_consensus
        from advisor.research.variant_perception import build_variant_perception

        company = report.business_model or sym
        report.consensus = build_consensus(sym)
        report.variant_perception = build_variant_perception(
            sym,
            company,
            dcf=report.dcf,
            consensus=report.consensus,
            catalyst_risk=report.catalyst_risk,
            red_flags=report.red_flags,
            industry=report.industry,
        )

    else:
        raise ValueError(f"Unknown layer: {layer}")

    # Persist the updated report so future cache hits see the refresh.
    store = ResearchStore(get_settings().db_path)
    try:
        store.save_report(report)
    finally:
        store.close()

    return report


def purge_cached_report(symbol: str) -> None:
    """Delete every cached row for `symbol` from research_reports.

    Called by the dashboard when a run produced an empty report — we don't
    want bogus tickers polluting the recent-reports list.
    """
    db_path = get_settings().db_path
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute("DELETE FROM research_reports WHERE symbol = ?", (symbol.upper(),))
        conn.commit()
    finally:
        conn.close()
