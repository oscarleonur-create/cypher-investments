"""Catalysts & Risks tab — timeline (Gantt) + risk register + risk heatmap."""

from __future__ import annotations

from datetime import date, timedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from advisor.dashboard.theme import PALETTE, PLOTLY_TEMPLATE, SEVERITY_COLORS
from advisor.research.models import CatalystItem, ResearchReport

_CATALYST_COLORS = {
    "EARNINGS": "#60a5fa",
    "PRODUCT_LAUNCH": "#19c37d",
    "REGULATORY": "#f59e0b",
    "CONTRACT": "#a78bfa",
    "MACRO": "#ef4444",
    "OTHER": "#9ca3af",
}


def render_catalysts(report: ResearchReport) -> None:
    cr = report.catalyst_risk
    if cr is None or (not cr.catalysts and not cr.risks):
        st.warning("No catalyst or risk data available.")
        return

    if cr.fetched_at:
        st.caption(
            f"Catalyst/risk data fetched {cr.fetched_at:%Y-%m-%d %H:%M}. "
            "Use **Refresh layers → Catalysts** in the sidebar if it looks stale."
        )

    if cr.catalysts:
        _render_timeline(cr.catalysts)
        st.divider()

    if cr.risks:
        _render_risk_register(cr.risks)
        st.divider()
        _render_risk_heatmap(cr.risks)


# ── Catalyst timeline ───────────────────────────────────────────────────────


def _render_timeline(catalysts: list[CatalystItem]) -> None:
    st.subheader("Catalyst timeline (next 12 months)")
    today = date.today()
    horizon = today + timedelta(days=365)

    rows = []
    for c in catalysts:
        span = _date_span(c.expected_date, today)
        if span is None:
            continue
        start, end = span
        if end < today or start > horizon:
            continue
        rows.append(
            {
                "Description": _truncate(c.description, 60),
                "Type": c.catalyst_type.value,
                "Start": pd.Timestamp(start),
                "End": pd.Timestamp(end),
                "Source": c.source or "—",
            }
        )

    if not rows:
        st.caption("No catalysts in the next 12 months.")
        return

    df = pd.DataFrame(rows).sort_values("Start")
    fig = px.timeline(
        df,
        x_start="Start",
        x_end="End",
        y="Description",
        color="Type",
        color_discrete_map=_CATALYST_COLORS,
        hover_data=["Source"],
    )
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=max(280, 40 + 38 * len(df)),
        margin=dict(l=20, r=20, t=20, b=20),
        legend=dict(orientation="h", y=-0.2),
        xaxis=dict(range=[pd.Timestamp(today), pd.Timestamp(horizon)]),
    )
    # Plotly's add_vline with annotation_text averages x-coordinates internally
    # via `sum(X)/len(X)`, which crashes on pd.Timestamp inputs. Add the shape
    # without an annotation, then place the label manually with add_annotation.
    today_ts = pd.Timestamp(today)
    fig.add_shape(
        type="line",
        x0=today_ts,
        x1=today_ts,
        xref="x",
        y0=0,
        y1=1,
        yref="paper",
        line=dict(color=PALETTE["neutral"], dash="dash", width=1),
    )
    fig.add_annotation(
        x=today_ts,
        y=1,
        xref="x",
        yref="paper",
        text="Today",
        showarrow=False,
        yshift=10,
        font=dict(color=PALETTE["neutral"], size=11),
    )
    st.plotly_chart(fig, use_container_width=True)


def _date_span(expected_date: str, today: date) -> tuple[date, date] | None:
    """Parse `expected_date` into (start, end). Quarter strings span the quarter;
    ISO dates render as a 3-day bar (so they're visible on the chart)."""
    if not expected_date:
        return None
    try:
        d = date.fromisoformat(expected_date[:10])
        return (d, d + timedelta(days=3))
    except ValueError:
        pass
    if "Q" in expected_date:
        try:
            year_str, q = expected_date.replace(" ", "").split("-Q")
            year = int(year_str)
            q = int(q)
            month_start = {1: 1, 2: 4, 3: 7, 4: 10}[q]
            from calendar import monthrange

            month_end = month_start + 2
            start = date(year, month_start, 1)
            end = date(year, month_end, monthrange(year, month_end)[1])
            return (start, end)
        except (KeyError, ValueError):
            return None
    return None


# ── Risk register ───────────────────────────────────────────────────────────


def _render_risk_register(risks) -> None:  # type: ignore[no-untyped-def]
    st.subheader("Risk register")
    rows = []
    for r in risks:
        rows.append(
            {
                "Severity": r.severity.value,
                "Category": r.category or "—",
                "P": r.probability,
                "I": r.impact,
                "Score": (r.probability * r.impact) if r.risk_score is not None else None,
                "Description": r.description,
                "Source": r.source or "—",
            }
        )
    df = pd.DataFrame(rows).sort_values("Score", ascending=False, na_position="last")

    def _style_severity(s: str) -> str:
        color = SEVERITY_COLORS.get(s, PALETTE["neutral"])
        return f"background-color: {color}22; color: {color}; font-weight: bold"

    styler = df.style.format({"P": _fmt_pct_loose, "I": _fmt_pct_loose, "Score": _fmt_score}).map(
        _style_severity, subset=["Severity"]
    )
    st.dataframe(styler, use_container_width=True, hide_index=True)


def _fmt_pct_loose(v) -> str:  # type: ignore[no-untyped-def]
    if v is None or pd.isna(v):
        return "—"
    return f"{v:.0%}"


def _fmt_score(v) -> str:  # type: ignore[no-untyped-def]
    if v is None or pd.isna(v):
        return "—"
    return f"{v:.2f}"


# ── Risk heatmap (probability × impact) ─────────────────────────────────────


def _render_risk_heatmap(risks) -> None:  # type: ignore[no-untyped-def]
    st.subheader("Probability × Impact")
    pts_x = []
    pts_y = []
    labels = []
    colors = []
    for r in risks:
        if r.probability is None or r.impact is None:
            continue
        pts_x.append(r.probability)
        pts_y.append(r.impact)
        labels.append(_truncate(r.description, 80))
        colors.append(SEVERITY_COLORS.get(r.severity.value, PALETTE["neutral"]))

    if not pts_x:
        st.caption("Risks lack probability/impact estimates — heatmap skipped.")
        return

    fig = go.Figure()

    # Quadrant background
    fig.add_shape(
        type="rect",
        x0=0.5,
        y0=0.5,
        x1=1.0,
        y1=1.0,
        fillcolor=PALETTE["negative"],
        opacity=0.10,
        line_width=0,
    )
    fig.add_shape(
        type="rect",
        x0=0,
        y0=0,
        x1=0.5,
        y1=0.5,
        fillcolor=PALETTE["positive"],
        opacity=0.06,
        line_width=0,
    )

    fig.add_trace(
        go.Scatter(
            x=pts_x,
            y=pts_y,
            mode="markers",
            marker=dict(size=18, color=colors, line=dict(color="#1f2937", width=1)),
            text=labels,
            hovertemplate="Prob %{x:.0%} · Impact %{y:.0%}<br>%{text}<extra></extra>",
        )
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=400,
        margin=dict(l=40, r=40, t=20, b=40),
        xaxis=dict(title="Probability", range=[0, 1], tickformat=".0%"),
        yaxis=dict(title="Impact", range=[0, 1], tickformat=".0%"),
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Helpers ─────────────────────────────────────────────────────────────────


def _truncate(s: str, n: int) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[: n - 1] + "…"
