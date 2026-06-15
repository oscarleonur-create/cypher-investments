"""Plotly figure builders for the Portfolio tab.

Kept separate from the renderer (mirrors ``simulator/charts.py``) so the figures
are easy to tweak in isolation. All use the shared dark template and palette.
"""

from __future__ import annotations

import plotly.graph_objects as go

from advisor.dashboard.theme import PALETTE, PLOTLY_TEMPLATE

# Discrete colourway for the concentration donuts.
_WEDGE_COLORS = [
    PALETTE["accent"],
    PALETTE["positive"],
    PALETTE["warn"],
    PALETTE["negative"],
    "#a78bfa",
    "#22d3ee",
    "#f472b6",
    "#84cc16",
    "#fb923c",
    "#38bdf8",
]

_FLAG_COLOR = {
    "OVERPRICED": PALETTE["negative"],
    "PRICED_IN": PALETTE["warn"],
    "UPSIDE": PALETTE["positive"],
    "UNKNOWN": PALETTE["neutral"],
}


def concentration_donut(labels: list[str], weights: list[float], title: str) -> go.Figure:
    """Donut of portfolio weights. ``weights`` are fractions (sum ≈ 1)."""
    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=weights,
            hole=0.55,
            sort=False,
            textinfo="label+percent",
            textposition="inside",
            marker=dict(colors=_WEDGE_COLORS * (len(labels) // len(_WEDGE_COLORS) + 1)),
            hovertemplate="%{label}: %{percent}<extra></extra>",
        )
    )
    fig.update_layout(
        title=title,
        template=PLOTLY_TEMPLATE,
        height=360,
        showlegend=False,
        margin=dict(t=48, b=8, l=8, r=8),
    )
    return fig


def valuation_bar(rows: list[dict]) -> go.Figure:
    """Horizontal bar of remaining upside per holding, coloured by flag."""
    rows = [r for r in rows if r.get("upside") is not None]
    symbols = [r["symbol"] for r in rows]
    upsides = [r["upside"] * 100 for r in rows]
    colors = [_FLAG_COLOR.get(r["flag"], PALETTE["neutral"]) for r in rows]

    fig = go.Figure(
        go.Bar(
            x=upsides,
            y=symbols,
            orientation="h",
            marker_color=colors,
            text=[f"{u:+.0f}%" for u in upsides],
            textposition="outside",
            hovertemplate="%{y}: %{x:+.1f}% upside<extra></extra>",
        )
    )
    fig.add_vline(x=0, line_width=1, line_color=PALETTE["neutral"])
    fig.update_layout(
        title="Remaining upside to fair value",
        xaxis_title="Upside %",
        template=PLOTLY_TEMPLATE,
        height=max(280, 28 * len(symbols) + 120),
        margin=dict(t=48, b=8, l=8, r=8),
    )
    return fig


def sector_rotation_bar(sectors: list[str], weights: list[float], rel_3m: list[float]) -> go.Figure:
    """Your sector weight (bars) overlaid with each sector's 3-mo relative momentum.

    Bars = portfolio weight (left axis, %); markers = sector ETF return vs SPY
    over 3 months (right axis, %). Leading sectors plot above zero.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=sectors,
            y=[w * 100 for w in weights],
            name="Your weight",
            marker_color=PALETTE["accent"],
            opacity=0.7,
            hovertemplate="%{x}: %{y:.1f}% of book<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=sectors,
            y=[(r * 100 if r is not None else None) for r in rel_3m],
            name="3-mo vs SPY",
            mode="markers",
            marker=dict(
                size=13,
                color=[
                    PALETTE["positive"] if (r is not None and r > 0) else PALETTE["negative"]
                    for r in rel_3m
                ],
                symbol="diamond",
            ),
            yaxis="y2",
            hovertemplate="%{x}: %{y:+.1f}% vs SPY (3m)<extra></extra>",
        )
    )
    fig.update_layout(
        title="Sector weight vs rotation (3-mo relative to SPY)",
        template=PLOTLY_TEMPLATE,
        height=380,
        yaxis=dict(title="Weight %"),
        yaxis2=dict(title="Rel. return %", overlaying="y", side="right", zeroline=True),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=64, b=8, l=8, r=8),
    )
    return fig


def vix_gauge(current: float, percentile_1y: float) -> go.Figure:
    """Indicator gauge for the current VIX level with calm/elevated/stress bands."""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=current,
            number={"suffix": ""},
            title={"text": f"VIX · {percentile_1y:.0%} of past year"},
            gauge={
                "axis": {"range": [0, 60]},
                "bar": {"color": PALETTE["accent"]},
                "steps": [
                    {"range": [0, 15], "color": "rgba(25,195,125,0.35)"},
                    {"range": [15, 25], "color": "rgba(245,158,11,0.30)"},
                    {"range": [25, 60], "color": "rgba(239,68,68,0.30)"},
                ],
            },
        )
    )
    fig.update_layout(template=PLOTLY_TEMPLATE, height=300, margin=dict(t=48, b=8, l=24, r=24))
    return fig


def vix_history(dates: list[str], values: list[float], sma20: float) -> go.Figure:
    """1-year VIX line with the 20-day SMA reference."""
    fig = go.Figure(
        go.Scatter(
            x=dates,
            y=values,
            mode="lines",
            line=dict(color=PALETTE["accent"], width=1.5),
            fill="tozeroy",
            fillcolor="rgba(96,165,250,0.12)",
            name="VIX",
            hovertemplate="%{x}: %{y:.1f}<extra></extra>",
        )
    )
    fig.add_hline(
        y=sma20,
        line_dash="dot",
        line_color=PALETTE["neutral"],
        annotation_text=f"20d SMA {sma20:.1f}",
        annotation_position="top left",
    )
    fig.update_layout(
        title="VIX — last 12 months",
        template=PLOTLY_TEMPLATE,
        height=300,
        margin=dict(t=48, b=8, l=8, r=8),
    )
    return fig
