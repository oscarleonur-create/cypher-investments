"""Ratios & Quality tab — 4-panel time-series + heatmap + red-flags table."""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from advisor.dashboard.theme import PALETTE, PLOTLY_TEMPLATE, severity_chip
from advisor.research.models import RatioBundle, RedFlagList, ResearchReport

# Direction of each ratio: True = higher-is-better.
# Used by the heatmap to flip the diverging colorscale where appropriate.
_HIGHER_IS_BETTER: dict[str, bool] = {
    "gross_margin": True,
    "operating_margin": True,
    "net_margin": True,
    "roa": True,
    "roe": True,
    "roic": True,
    "current_ratio": True,
    "quick_ratio": True,
    "interest_coverage": True,
    "asset_turnover": True,
    "inventory_turns": True,
    "fcf_margin": True,
    "fcf_to_net_income": True,
    # lower-is-better
    "debt_to_equity": False,
    "debt_to_ebitda": False,
    "dso": False,
    "capex_intensity": False,
}


def render_ratios(report: ResearchReport) -> None:
    ratios = report.ratios
    if ratios is None or not ratios.periods:
        st.warning("No ratio data available.")
        return

    _render_time_series(ratios)
    st.divider()
    _render_heatmap(ratios)
    st.divider()
    _render_red_flags(report.red_flags)
    st.divider()
    _render_share_count(ratios)


# ── 4-panel time-series ─────────────────────────────────────────────────────


def _render_time_series(ratios: RatioBundle) -> None:
    st.subheader("Ratios over time")
    # Periods are most-recent-first; reverse for chronological x-axis.
    periods = list(reversed(ratios.periods))
    years = [p.fiscal_year for p in periods]

    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=(
            "Profitability",
            "Liquidity & Leverage",
            "Efficiency",
            "Cash Quality",
        ),
        vertical_spacing=0.14,
        horizontal_spacing=0.08,
        specs=[
            [{"secondary_y": False}, {"secondary_y": False}],
            [{"secondary_y": True}, {"secondary_y": False}],
        ],
    )

    # Profitability (row 1, col 1)
    _add_line(fig, years, periods, "gross_margin", "Gross margin", PALETTE["accent"], 1, 1)
    _add_line(fig, years, periods, "operating_margin", "Operating margin", PALETTE["warn"], 1, 1)
    _add_line(fig, years, periods, "net_margin", "Net margin", PALETTE["positive"], 1, 1)
    _add_line(fig, years, periods, "roe", "ROE", "#a78bfa", 1, 1)
    _add_line(fig, years, periods, "roic", "ROIC", "#ec4899", 1, 1)
    fig.update_yaxes(tickformat=".0%", row=1, col=1)

    # Liquidity & Leverage (row 1, col 2)
    _add_line(fig, years, periods, "current_ratio", "Current", PALETTE["accent"], 1, 2)
    _add_line(fig, years, periods, "quick_ratio", "Quick", PALETTE["positive"], 1, 2)
    _add_line(fig, years, periods, "debt_to_equity", "D/E", PALETTE["warn"], 1, 2)
    _add_line(fig, years, periods, "debt_to_ebitda", "D/EBITDA", PALETTE["negative"], 1, 2)
    _add_line(fig, years, periods, "interest_coverage", "Int cov", "#a78bfa", 1, 2)

    # Efficiency (row 2, col 1) — DSO on secondary y-axis (days, not turns)
    _add_line(fig, years, periods, "asset_turnover", "Asset turnover", PALETTE["accent"], 2, 1)
    _add_line(fig, years, periods, "inventory_turns", "Inventory turns", PALETTE["positive"], 2, 1)
    _add_line(
        fig,
        years,
        periods,
        "dso",
        "DSO (days)",
        PALETTE["warn"],
        2,
        1,
        secondary_y=True,
        dash="dot",
    )

    # Cash Quality (row 2, col 2)
    _add_line(fig, years, periods, "fcf_margin", "FCF margin", PALETTE["positive"], 2, 2)
    _add_line(fig, years, periods, "capex_intensity", "Capex / Rev", PALETTE["warn"], 2, 2)
    _add_line(fig, years, periods, "fcf_to_net_income", "FCF / NI", PALETTE["accent"], 2, 2)
    fig.update_yaxes(tickformat=".0%", row=2, col=2)

    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=620,
        margin=dict(l=20, r=20, t=50, b=20),
        legend=dict(orientation="h", y=-0.12),
        hovermode="x unified",
    )
    st.plotly_chart(fig, use_container_width=True)


def _add_line(
    fig,
    x: list,
    periods: list,
    field: str,
    name: str,
    color: str,
    row: int,
    col: int,
    secondary_y: bool = False,
    dash: str | None = None,
) -> None:
    y = [getattr(p, field, None) for p in periods]
    if not any(v is not None for v in y):
        return
    fig.add_trace(
        go.Scatter(
            x=x,
            y=y,
            mode="lines+markers",
            name=name,
            line=dict(color=color, dash=dash) if dash else dict(color=color),
        ),
        row=row,
        col=col,
        secondary_y=secondary_y,
    )


# ── Ratio heatmap ────────────────────────────────────────────────────────────


def _render_heatmap(ratios: RatioBundle) -> None:
    st.subheader("Ratio heatmap")
    st.caption(
        "Each row is normalized across the available years. "
        "Green = best year (in the right direction for that ratio), red = worst."
    )

    rows = [
        ("gross_margin", "Gross margin"),
        ("operating_margin", "Operating margin"),
        ("net_margin", "Net margin"),
        ("roe", "ROE"),
        ("roic", "ROIC"),
        ("current_ratio", "Current ratio"),
        ("debt_to_equity", "D / E"),
        ("debt_to_ebitda", "D / EBITDA"),
        ("interest_coverage", "Interest coverage"),
        ("asset_turnover", "Asset turnover"),
        ("dso", "DSO (days)"),
        ("fcf_margin", "FCF margin"),
        ("capex_intensity", "Capex / Rev"),
        ("fcf_to_net_income", "FCF / NI"),
    ]
    periods = list(reversed(ratios.periods))
    years = [str(p.fiscal_year) for p in periods]

    raw = np.array(
        [[getattr(p, attr, None) for p in periods] for attr, _ in rows],
        dtype=object,
    )

    # Per-row normalization → score in [-1, +1] where +1 = best, -1 = worst
    z = np.full(raw.shape, np.nan, dtype=float)
    text = np.full(raw.shape, "—", dtype=object)
    for i, (attr, _) in enumerate(rows):
        vals = [v for v in raw[i] if v is not None]
        if not vals:
            continue
        lo, hi = min(vals), max(vals)
        for j, v in enumerate(raw[i]):
            if v is None:
                continue
            text[i, j] = _format_ratio_cell(attr, v)
            if hi == lo:
                z[i, j] = 0.0
                continue
            score = 2 * (v - lo) / (hi - lo) - 1  # [-1, 1]
            if not _HIGHER_IS_BETTER.get(attr, True):
                score = -score
            z[i, j] = score

    fig = go.Figure(
        data=go.Heatmap(
            z=z,
            x=years,
            y=[label for _, label in rows],
            text=text,
            texttemplate="%{text}",
            textfont=dict(size=11),
            colorscale=[
                [0.0, PALETTE["negative"]],
                [0.5, "#1f2937"],
                [1.0, PALETTE["positive"]],
            ],
            zmin=-1,
            zmax=1,
            showscale=False,
            hovertemplate="%{y} · %{x}: %{text}<extra></extra>",
        )
    )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=420,
        margin=dict(l=20, r=20, t=20, b=20),
        xaxis=dict(side="top"),
    )
    fig.update_yaxes(autorange="reversed")
    st.plotly_chart(fig, use_container_width=True)


def _format_ratio_cell(attr: str, v: float) -> str:
    if attr in {
        "gross_margin",
        "operating_margin",
        "net_margin",
        "roa",
        "roe",
        "roic",
        "fcf_margin",
        "capex_intensity",
    }:
        return f"{v * 100:.1f}%"
    if attr == "dso":
        return f"{v:.0f}d"
    return f"{v:.2f}×"


# ── Red flags table ──────────────────────────────────────────────────────────


def _render_red_flags(red_flags: RedFlagList | None) -> None:
    st.subheader("Red flags")
    if red_flags is None or not red_flags.flags:
        green = PALETTE["positive"]
        st.markdown(
            f"<span style='color:{green}'>No red flags detected.</span>",
            unsafe_allow_html=True,
        )
        return

    chips = "".join(
        severity_chip(f.severity.value, f"{f.severity.value}: {f.code}") for f in red_flags.flags
    )
    st.markdown(chips, unsafe_allow_html=True)

    rows = []
    for f in red_flags.flags:
        rows.append(
            {
                "Severity": f.severity.value,
                "Code": f.code,
                "Title": f.title,
                "Period": str(f.period_end) if f.period_end else "—",
                "Detail": f.detail,
            }
        )
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


# ── Share-count CAGR badge ───────────────────────────────────────────────────


def _render_share_count(ratios: RatioBundle) -> None:
    cagr = ratios.share_count_cagr_3y
    if cagr is None:
        return
    st.subheader("Share count")
    if cagr > 0.03:
        color = PALETTE["negative"]
        verdict = "dilutive"
    elif cagr < -0.01:
        color = PALETTE["positive"]
        verdict = "buybacks"
    else:
        color = PALETTE["warn"]
        verdict = "roughly flat"
    st.markdown(
        f"3-year share-count CAGR: "
        f"<span style='color:{color};font-weight:600'>{cagr:+.1%}</span> · {verdict}",
        unsafe_allow_html=True,
    )
