"""KPI Monitor tab — thesis health dashboard.

Shows 3-5 thesis-specific KPIs with current values, bull/bear thresholds,
historical trend, and overall thesis status (ON_TRACK / CAUTION / INVALIDATED).
"""

from __future__ import annotations

import streamlit as st

from advisor.dashboard import data as _data
from advisor.dashboard import state as _state
from advisor.dashboard.theme import PALETTE
from advisor.research.models import KpiDefinition, KpiStatus, ResearchReport, ThesisMonitorResult

_STATUS_COLOR = {
    "ON_TRACK": PALETTE["positive"],
    "CAUTION": PALETTE["warn"],
    "INVALIDATED": PALETTE["negative"],
    "UNKNOWN": PALETTE["neutral"],
}
_KPI_COLOR = {
    KpiStatus.ON_TRACK: PALETTE["positive"],
    KpiStatus.CAUTION: PALETTE["warn"],
    KpiStatus.BREACHED: PALETTE["negative"],
    KpiStatus.UNKNOWN: PALETTE["neutral"],
}


def render_kpi(report: ResearchReport) -> None:
    monitor = report.kpi_monitor

    if monitor is None:
        _render_no_data(report)
        return

    _render_thesis_status(monitor)
    st.divider()
    _render_kpi_grid(monitor)

    if monitor.alerts:
        st.divider()
        _render_alerts(monitor)

    st.divider()
    _render_history(report.symbol)
    st.divider()
    _render_refresh_footer(report)


# ── Thesis status banner ──────────────────────────────────────────────────────


def _render_thesis_status(monitor: ThesisMonitorResult) -> None:
    status = monitor.thesis_status
    color = _STATUS_COLOR.get(status, PALETTE["neutral"])
    icon = {"ON_TRACK": "✓", "CAUTION": "⚠", "INVALIDATED": "✗"}.get(status, "?")
    label = status.replace("_", " ").title()

    col1, col2 = st.columns([2, 3])

    col1.markdown(
        f"<div style='padding:16px;background:{color}1a;border:2px solid {color};"
        f"border-radius:8px;text-align:center'>"
        f"<p style='margin:0;font-size:2em'>{icon}</p>"
        f"<p style='margin:4px 0 0;font-size:1.1em;font-weight:700;color:{color}'>{label}</p>"
        f"<p style='margin:2px 0 0;font-size:0.8em;color:#9ca3af'>Thesis Status</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

    with col2:
        breached = sum(1 for k in monitor.kpis if k.status == KpiStatus.BREACHED)
        caution = sum(1 for k in monitor.kpis if k.status == KpiStatus.CAUTION)
        on_track = sum(1 for k in monitor.kpis if k.status == KpiStatus.ON_TRACK)

        m1, m2, m3 = st.columns(3)
        m1.metric("On Track", on_track)
        m2.metric("Caution", caution)
        m3.metric(
            "Breached", breached, delta=f"-{breached}" if breached else None, delta_color="inverse"
        )

        if monitor.checked_at:
            st.caption(f"Last checked: {monitor.checked_at:%Y-%m-%d %H:%M}")


# ── KPI grid ──────────────────────────────────────────────────────────────────


def _render_kpi_grid(monitor: ThesisMonitorResult) -> None:
    st.subheader("KPI Dashboard")

    for kpi in monitor.kpis:
        _render_kpi_card(kpi)


def _render_kpi_card(kpi: KpiDefinition) -> None:
    color = _KPI_COLOR.get(kpi.status, PALETTE["neutral"])
    status_label = kpi.status.value.replace("_", " ").title()
    val_str = _fmt_value(kpi.current_value, kpi.unit)
    prev_str = _fmt_value(kpi.previous_value, kpi.unit) if kpi.previous_value is not None else None

    with st.container(border=True):
        col1, col2, col3, col4 = st.columns([3, 2, 2, 1])

        with col1:
            st.markdown(f"**{kpi.metric_name}**")
            if kpi.description:
                st.caption(kpi.description)

        with col2:
            delta = None
            if prev_str and kpi.current_value is not None and kpi.previous_value is not None:
                delta = _fmt_value(kpi.current_value - kpi.previous_value, kpi.unit)
            st.metric("Current", val_str, delta=delta)

        with col3:
            thr_lines = []
            if kpi.bull_threshold is not None:
                thr_lines.append(f"Bull: ≥ {_fmt_value(kpi.bull_threshold, kpi.unit)}")
            if kpi.bear_threshold is not None:
                thr_lines.append(f"Bear: < {_fmt_value(kpi.bear_threshold, kpi.unit)}")
            if thr_lines:
                st.caption("\n".join(thr_lines))
            else:
                st.caption("No thresholds set")

        with col4:
            st.markdown(
                f"<div style='text-align:center;padding-top:8px'>"
                f"<span style='color:{color};font-weight:700;font-size:0.9em'>"
                f"{status_label}</span></div>",
                unsafe_allow_html=True,
            )


# ── Alerts ────────────────────────────────────────────────────────────────────


def _render_alerts(monitor: ThesisMonitorResult) -> None:
    st.subheader("Active Alerts")
    for alert in monitor.alerts:
        is_breach = alert.startswith("⚠")
        color = PALETTE["negative"] if is_breach else PALETTE["warn"]
        st.markdown(
            f"<div style='border-left:3px solid {color};padding:8px 12px;"
            f"margin-bottom:6px;background:{color}18;border-radius:3px'>"
            f"{alert}</div>",
            unsafe_allow_html=True,
        )


# ── History ───────────────────────────────────────────────────────────────────


def _render_history(symbol: str) -> None:
    st.subheader("Check History")
    try:
        from advisor.research.config import get_settings
        from advisor.research.store import ResearchStore

        store = ResearchStore(get_settings().db_path)
        try:
            history = store.load_kpi_history(symbol.upper(), limit=10)
        finally:
            store.close()
    except Exception:  # noqa: BLE001
        st.caption("Could not load history.")
        return

    if not history:
        st.caption("No prior KPI checks recorded.")
        return

    rows = []
    for result in history:
        breached = sum(1 for k in result.kpis if k.status == KpiStatus.BREACHED)
        caution = sum(1 for k in result.kpis if k.status == KpiStatus.CAUTION)
        rows.append(
            {
                "Checked At": result.checked_at.strftime("%Y-%m-%d %H:%M"),
                "Status": result.thesis_status,
                "Breached": breached,
                "Caution": caution,
                "Alerts": len(result.alerts),
            }
        )

    import pandas as pd

    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ── Empty state / refresh ─────────────────────────────────────────────────────


def _render_no_data(report: ResearchReport) -> None:
    st.info(
        "KPI monitor not yet computed. Click **↻ Refresh KPIs** below to build the "
        "thesis-tracking dashboard from the cached report."
    )
    _render_refresh_footer(report)


def _render_refresh_footer(report: ResearchReport) -> None:
    col1, col2 = st.columns([3, 1])
    col1.caption(
        "KPIs are identified from the thesis context (requires OpenRouter API key) "
        "and measured from live yfinance data. Re-check anytime to update current values."
    )
    if col2.button(
        "↻ Refresh KPIs",
        key="inline_refresh_kpi",
        type="primary",
        use_container_width=True,
    ):
        with st.spinner("Refreshing KPI values…"):
            try:
                updated = _data.refresh_layer(report, layer="kpi")
                _state.set_report(updated)
                _state.log_event(report.symbol, "refresh", {"layer": "kpi"})
            except Exception as exc:  # noqa: BLE001
                st.error(f"KPI refresh failed: {exc}")
        st.rerun()


# ── Helpers ───────────────────────────────────────────────────────────────────


def _fmt_value(v: float | None, unit: str) -> str:
    if v is None:
        return "—"
    if unit == "pct":
        return f"{v:.1%}"
    if unit in ("x", "ratio"):
        return f"{v:.2f}x"
    if unit == "usd":
        abs_v = abs(v)
        if abs_v >= 1e9:
            return f"${v / 1e9:,.2f}B"
        if abs_v >= 1e6:
            return f"${v / 1e6:,.1f}M"
        return f"${v:,.0f}"
    return f"{v:.3f}"
