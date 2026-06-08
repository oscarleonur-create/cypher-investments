"""KPI Tracker — thesis-monitoring dashboard.

Phase 6: identifies 3-5 measurable KPIs from the thesis, maps each to a
yfinance or ratio field so they can be re-checked automatically, and produces
a ThesisMonitorResult with status (ON_TRACK / CAUTION / INVALIDATED).

Flow:
  1. `build_kpi_definitions(report)` — LLM picks KPIs from thesis context;
      fallback to a standard set when LLM is unavailable.
  2. `refresh_kpi_values(symbol, kpis)` — fetches current yfinance values and
      evaluates bull/bear thresholds.
  3. `build_thesis_monitor(report)` — runs 1+2 and returns ThesisMonitorResult.
  4. `re_check_kpis(symbol, store)` — standalone monitor: loads saved KPIs,
      refreshes values, saves a new history row, returns updated result.
"""

from __future__ import annotations

import logging
from datetime import datetime

import yfinance as yf

from advisor.research.models import (
    KpiDefinition,
    KpiStatus,
    ResearchReport,
    ThesisMonitorResult,
)

logger = logging.getLogger(__name__)

# ── Source registry ───────────────────────────────────────────────────────────
# Maps measurement_source strings → (yfinance info key, unit)
_YFINANCE_MAP: dict[str, tuple[str, str]] = {
    "yfinance:revenueGrowth": ("revenueGrowth", "pct"),
    "yfinance:earningsGrowth": ("earningsGrowth", "pct"),
    "yfinance:grossMargins": ("grossMargins", "pct"),
    "yfinance:operatingMargins": ("operatingMargins", "pct"),
    "yfinance:profitMargins": ("profitMargins", "pct"),
    "yfinance:returnOnEquity": ("returnOnEquity", "pct"),
    "yfinance:returnOnAssets": ("returnOnAssets", "pct"),
    "yfinance:debtToEquity": ("debtToEquity", "x"),
    "yfinance:currentRatio": ("currentRatio", "x"),
    "yfinance:quickRatio": ("quickRatio", "x"),
    "yfinance:freeCashflow": ("freeCashflow", "usd"),
    "yfinance:revenuePerShare": ("revenuePerShare", "usd"),
    "yfinance:bookValue": ("bookValue", "usd"),
    "yfinance:shortRatio": ("shortRatio", "x"),
}

# Default KPI set when LLM is unavailable — broad enough to cover most theses
_DEFAULT_KPIS: list[dict] = [
    {
        "metric_name": "Revenue Growth YoY",
        "description": "Year-over-year revenue growth rate",
        "measurement_source": "yfinance:revenueGrowth",
        "bull_threshold": 0.10,
        "bear_threshold": 0.00,
        "unit": "pct",
    },
    {
        "metric_name": "Gross Margin",
        "description": "Gross profit as % of revenue",
        "measurement_source": "yfinance:grossMargins",
        "bull_threshold": None,
        "bear_threshold": None,
        "unit": "pct",
    },
    {
        "metric_name": "Operating Margin",
        "description": "Operating income as % of revenue",
        "measurement_source": "yfinance:operatingMargins",
        "bull_threshold": None,
        "bear_threshold": 0.00,
        "unit": "pct",
    },
    {
        "metric_name": "Return on Equity",
        "description": "Net income / average shareholders' equity",
        "measurement_source": "yfinance:returnOnEquity",
        "bull_threshold": 0.15,
        "bear_threshold": 0.00,
        "unit": "pct",
    },
    {
        "metric_name": "Debt / Equity",
        "description": "Total debt relative to equity (lower = safer)",
        "measurement_source": "yfinance:debtToEquity",
        "bull_threshold": None,
        "bear_threshold": None,
        "unit": "x",
    },
]


# ── Public API ────────────────────────────────────────────────────────────────


def build_thesis_monitor(report: ResearchReport) -> ThesisMonitorResult:
    """Build KPIs from the thesis and immediately check current values."""
    sym = report.symbol
    kpis = build_kpi_definitions(report)
    kpis = refresh_kpi_values(sym, kpis)
    return _assemble_result(sym, kpis)


def re_check_kpis(symbol: str) -> ThesisMonitorResult:
    """Load saved KPI definitions for `symbol`, refresh values, return result.

    Used by `advisor research monitor` to update an existing watchlist without
    rebuilding the full report.
    """
    from advisor.research.config import get_settings
    from advisor.research.store import ResearchStore

    store = ResearchStore(get_settings().db_path)
    try:
        kpis = store.load_kpi_watchlist(symbol.upper())
        if not kpis:
            # Fall back to building from the cached report
            report = store.load_latest_report(symbol.upper())
            if report is None:
                return ThesisMonitorResult(symbol=symbol.upper(), thesis_status="UNKNOWN")
            kpis = build_kpi_definitions(report)

        kpis = refresh_kpi_values(symbol, kpis)
        result = _assemble_result(symbol, kpis)
        store.save_kpi_check(symbol.upper(), result)
        store.save_kpi_watchlist(symbol.upper(), kpis)
        return result
    finally:
        store.close()


def build_kpi_definitions(report: ResearchReport) -> list[KpiDefinition]:
    """Use the LLM to pick 3-5 thesis-specific KPIs; fall back to defaults."""
    llm_kpis = _llm_kpis(report)
    if llm_kpis:
        return llm_kpis
    return [KpiDefinition(**k) for k in _DEFAULT_KPIS]


def refresh_kpi_values(symbol: str, kpis: list[KpiDefinition]) -> list[KpiDefinition]:
    """Fetch current values for each KPI and evaluate status."""
    info: dict = {}
    try:
        info = yf.Ticker(symbol.upper()).info or {}
    except Exception as exc:  # noqa: BLE001
        logger.warning("yfinance info fetch failed for %s: %s", symbol, exc)

    now = datetime.now()
    for kpi in kpis:
        kpi.previous_value = kpi.current_value
        kpi.current_value = _fetch_value(kpi.measurement_source, info)
        kpi.checked_at = now
        kpi.status = _evaluate_status(kpi)

    return kpis


# ── LLM enrichment ────────────────────────────────────────────────────────────


def _llm_kpis(report: ResearchReport) -> list[KpiDefinition]:
    try:
        from pydantic import BaseModel as PydanticBase
        from research_agent.config import ResearchConfig
        from research_agent.llm import OpenRouterLLM

        config = ResearchConfig()
        if not config.openrouter_api_key:
            return []

        context = _build_context(report)
        source_list = "\n".join(f"  {k}" for k in _YFINANCE_MAP)

        class KpiOut(PydanticBase):
            metric_name: str
            description: str
            measurement_source: str
            bull_threshold: float | None
            bear_threshold: float | None
            unit: str

        class KpisOut(PydanticBase):
            kpis: list[KpiOut]

        llm = OpenRouterLLM(config)
        result = llm.complete(
            system_prompt=(
                "You are a hedge-fund portfolio manager defining KPIs to monitor "
                "an investment thesis. "
                "Pick 3-5 KPIs that are most relevant to validating or invalidating "
                "THIS specific thesis. "
                "Each KPI must map to one of the following measurement_source keys:\n"
                f"{source_list}\n"
                "CRITICAL — threshold format: use the SAME numeric scale that yfinance returns.\n"
                "  • revenueGrowth, grossMargins, operatingMargins, profitMargins, "
                "returnOnEquity, returnOnAssets, earningsGrowth: DECIMAL (0.15 = 15%, NOT 15).\n"
                "  • debtToEquity: raw ratio (1.5 = 150% D/E as yfinance reports it).\n"
                "  • currentRatio, quickRatio, shortRatio: raw ratio (2.0 = 2.0x).\n"
                "  • freeCashflow: USD absolute (e.g. 1000000000 for $1B).\n"
                "bull_threshold: value ABOVE which the thesis is confirmed (null if N/A). "
                "bear_threshold: value BELOW which the thesis is challenged (null if N/A). "
                "unit: 'pct' for margin/growth fields, 'x' for ratio fields, "
                "'usd' for dollar fields. "
                "Be precise. Use real numbers based on the context provided."
            ),
            user_prompt=context,
            response_model=KpisOut,
        )
        return [
            KpiDefinition(
                metric_name=k.metric_name,
                description=k.description,
                measurement_source=k.measurement_source,
                bull_threshold=k.bull_threshold,
                bear_threshold=k.bear_threshold,
                unit=k.unit,
            )
            for k in result.kpis
            if k.measurement_source in _YFINANCE_MAP
        ]

    except Exception as exc:  # noqa: BLE001
        logger.warning("KPI LLM enrichment failed for %s: %s", report.symbol, exc)
        return []


def _build_context(report: ResearchReport) -> str:
    parts = [f"Company: {report.business_model or report.symbol} ({report.symbol})"]

    if report.thesis:
        if report.thesis.summary:
            parts.append(f"Thesis: {report.thesis.summary[:300]}")
        if report.thesis.base and report.thesis.base.key_assumptions:
            parts.append("Base assumptions: " + "; ".join(report.thesis.base.key_assumptions[:3]))
        if report.thesis.bear and report.thesis.bear.what_proves_wrong:
            parts.append(
                "What proves wrong: " + "; ".join(report.thesis.bear.what_proves_wrong[:3])
            )

    vp = report.variant_perception
    if vp and vp.our_key_insight:
        parts.append(f"Our edge: {vp.our_key_insight}")

    if report.industry:
        parts.append(f"Moat: {report.industry.moat_type.value}")

    dcf = report.dcf
    if dcf and dcf.base:
        a = dcf.base.assumptions
        parts.append(
            f"Base DCF: revenue growth {a.revenue_growth_yr1_3:.1%}, "
            f"FCF margin {a.target_fcf_margin:.1%}"
        )

    return "\n".join(parts)


# ── Value fetching ────────────────────────────────────────────────────────────


def _fetch_value(source: str, info: dict) -> float | None:
    if source in _YFINANCE_MAP:
        key, _ = _YFINANCE_MAP[source]
        val = info.get(key)
        if val is not None:
            try:
                return float(val)
            except (TypeError, ValueError):
                pass
    return None


# ── Status evaluation ─────────────────────────────────────────────────────────


def _evaluate_status(kpi: KpiDefinition) -> KpiStatus:
    v = kpi.current_value
    if v is None:
        return KpiStatus.UNKNOWN

    if kpi.bear_threshold is not None and v < kpi.bear_threshold:
        return KpiStatus.BREACHED

    if kpi.bull_threshold is not None and v >= kpi.bull_threshold:
        return KpiStatus.ON_TRACK

    # In between thresholds or only one threshold set
    if kpi.bull_threshold is None and kpi.bear_threshold is None:
        return KpiStatus.UNKNOWN

    return KpiStatus.CAUTION


# ── Result assembly ────────────────────────────────────────────────────────────


def _assemble_result(symbol: str, kpis: list[KpiDefinition]) -> ThesisMonitorResult:
    alerts: list[str] = []
    breach_count = 0
    caution_count = 0

    for kpi in kpis:
        if kpi.status == KpiStatus.BREACHED:
            breach_count += 1
            val_str = _fmt_kpi_value(kpi)
            thr_str = _fmt_threshold(kpi.bear_threshold, kpi.unit)
            alerts.append(f"⚠ {kpi.metric_name} breached bear threshold: {val_str} < {thr_str}")
        elif kpi.status == KpiStatus.CAUTION:
            caution_count += 1
            alerts.append(f"• {kpi.metric_name} in caution zone — monitor closely")

    if breach_count >= 2:
        thesis_status = "INVALIDATED"
    elif breach_count == 1 or caution_count >= 2:
        thesis_status = "CAUTION"
    else:
        thesis_status = "ON_TRACK"

    return ThesisMonitorResult(
        symbol=symbol.upper(),
        kpis=kpis,
        alerts=alerts,
        thesis_status=thesis_status,
    )


def _fmt_kpi_value(kpi: KpiDefinition) -> str:
    v = kpi.current_value
    if v is None:
        return "—"
    if kpi.unit == "pct":
        return f"{v:.1%}"
    if kpi.unit in ("x", "ratio"):
        return f"{v:.2f}x"
    if kpi.unit == "usd":
        return f"${v:,.0f}"
    return f"{v:.3f}"


def _fmt_threshold(v: float | None, unit: str) -> str:
    if v is None:
        return "—"
    if unit == "pct":
        return f"{v:.1%}"
    if unit in ("x", "ratio"):
        return f"{v:.2f}x"
    return f"{v:.3f}"
