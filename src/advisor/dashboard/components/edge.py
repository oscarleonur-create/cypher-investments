"""Edge tab — Variant Perception + Sell-Side Consensus.

Shows what the market believes, what we believe, and exactly why we differ.
This is the core hedge-fund alpha articulation layer (§0 of the research report).
"""

from __future__ import annotations

import plotly.graph_objects as go
import streamlit as st

from advisor.dashboard import data as _data
from advisor.dashboard import state as _state
from advisor.dashboard.theme import PALETTE, PLOTLY_TEMPLATE
from advisor.research.models import (
    AnalystConsensus,
    ResearchReport,
    VariantPerceptionResult,
)

# Threshold: stock more than this far above/below consensus = divergence alert
_DIVERGENCE_THRESHOLD = 0.15


def render_edge(report: ResearchReport) -> None:
    vp = report.variant_perception
    cn = report.consensus

    if vp is None and cn is None:
        _render_no_data(report)
        return

    # Detect stale / empty consensus data and offer inline refresh
    cn_empty = cn is not None and cn.n_analysts == 0 and cn.target_price_mean is None
    if cn_empty:
        _render_stale_warning(report)
        cn = None  # treat as absent so we don't render dashes

    # ── Key Insight callout ──────────────────────────────────────────────────
    if vp and vp.our_key_insight:
        st.info(f"**Edge:** {vp.our_key_insight}")

    # ── Divergence alert: stock vs. consensus target ─────────────────────────
    if cn and cn.current_price and cn.target_price_mean:
        gap = (cn.current_price / cn.target_price_mean) - 1
        if abs(gap) >= _DIVERGENCE_THRESHOLD:
            _render_divergence_alert(cn, gap)

    # ── Price target landscape chart ─────────────────────────────────────────
    if cn or (report.dcf and (report.dcf.base or report.dcf.bull or report.dcf.bear)):
        _render_price_target_chart(report, vp, cn)
        st.divider()

    # ── Market vs. Our View table ────────────────────────────────────────────
    _render_comparison_table(report, vp, cn)

    if vp:
        st.divider()
        _render_edge_narrative(vp)

    if cn and cn.n_analysts > 0:
        st.divider()
        _render_analyst_detail(cn)
    elif cn is None and vp is None:
        pass  # already handled above

    # Inline refresh footer
    st.divider()
    _render_refresh_footer(report)


# ── Empty / stale states ──────────────────────────────────────────────────────


def _render_no_data(report: ResearchReport) -> None:
    st.info(
        "No edge analysis yet for this report. Click **↻ Refresh Edge** below to fetch "
        "analyst consensus and run Variant Perception."
    )
    _render_refresh_footer(report)


def _render_stale_warning(report: ResearchReport) -> None:
    st.warning(
        "Analyst consensus data is empty — the report was likely cached before the Edge "
        "layer was added, or yfinance hit a transient error. "
        "Click **↻ Refresh Edge** below to re-fetch."
    )


def _render_refresh_footer(report: ResearchReport) -> None:
    col1, col2 = st.columns([3, 1])
    col1.caption(
        "Consensus data is fetched from yfinance (analyst price targets, forward EPS estimates, "
        "revision counts). Variant Perception narrative requires an OpenRouter API key."
    )
    if col2.button(
        "↻ Refresh Edge", key="inline_refresh_edge", type="primary", use_container_width=True
    ):
        with st.spinner("Refreshing consensus + variant perception…"):
            try:
                updated = _data.refresh_layer(report, layer="edge")
                _state.set_report(updated)
                _state.log_event(report.symbol, "refresh", {"layer": "edge"})
            except Exception as exc:  # noqa: BLE001
                st.error(f"Edge refresh failed: {exc}")
        st.rerun()


# ── Divergence alert ──────────────────────────────────────────────────────────


def _render_divergence_alert(cn: AnalystConsensus, gap: float) -> None:
    """Show a prominent alert when the stock is trading far above/below consensus targets."""
    color = PALETTE["warn"] if gap > 0 else PALETTE["negative"]

    if gap > 0:
        headline = (
            f"**{cn.symbol} is trading {gap:.1%} above the consensus mean target** "
            f"(${cn.current_price:.2f} vs. ${cn.target_price_mean:.2f})."
        )
        implication = (
            "This means the street's published targets are stale — the stock has moved "
            "ahead of analyst models. Either the consensus hasn't caught up to recent "
            "developments, or the market is pricing in upside that analysts haven't yet modelled."
        )
    else:
        headline = (
            f"**{cn.symbol} is trading {abs(gap):.1%} below the consensus mean target** "
            f"(${cn.current_price:.2f} vs. ${cn.target_price_mean:.2f})."
        )
        implication = (
            "The market is discounting the stock more aggressively than analyst models suggest. "
            "This could signal a genuine value opportunity or that consensus is too optimistic."
        )

    st.markdown(
        f"<div style='border-left:4px solid {color};padding:10px 16px;"
        f"background:{color}18;border-radius:4px;margin-bottom:12px'>"
        f"<p style='margin:0;color:{color}'>{headline}</p>"
        f"<p style='margin:6px 0 0;font-size:0.9em;color:#9ca3af'>{implication}</p>"
        f"</div>",
        unsafe_allow_html=True,
    )


# ── Price target chart ────────────────────────────────────────────────────────


def _render_price_target_chart(
    report: ResearchReport,
    vp: VariantPerceptionResult | None,
    cn: AnalystConsensus | None,
) -> None:
    st.subheader("Price target landscape")

    current = (cn.current_price if cn else None) or (
        report.dcf.current_price if report.dcf else None
    )

    bars: list[tuple[str, float, str, str]] = []  # (label, value, color, group)

    if cn:
        if cn.target_price_low:
            bars.append(("Street Low", cn.target_price_low, PALETTE["neutral"], "consensus"))
        if cn.target_price_median:
            bars.append(("Street Median", cn.target_price_median, "#d97706", "consensus"))
        if cn.target_price_mean:
            bars.append(("Street Mean", cn.target_price_mean, PALETTE["warn"], "consensus"))
        if cn.target_price_high:
            bars.append(("Street High", cn.target_price_high, PALETTE["neutral"], "consensus"))

    dcf = report.dcf
    if dcf:
        if dcf.bear:
            bars.append(("Our Bear", dcf.bear.implied_price, PALETTE["negative"], "ours"))
        if dcf.base:
            bars.append(("Our Base", dcf.base.implied_price, PALETTE["accent"], "ours"))
        if dcf.bull:
            bars.append(("Our Bull", dcf.bull.implied_price, PALETTE["positive"], "ours"))

    if not bars:
        st.caption("No price targets to chart.")
        return

    # Sort so consensus and ours are interleaved by price level
    bars.sort(key=lambda b: b[1])

    fig = go.Figure()
    fig.add_trace(
        go.Bar(
            x=[v for _, v, _, _ in bars],
            y=[label for label, _, _, _ in bars],
            orientation="h",
            marker=dict(color=[c for _, _, c, _ in bars]),
            text=[f"${v:,.2f}" for _, v, _, _ in bars],
            textposition="outside",
            hovertemplate="%{y}: $%{x:,.2f}<extra></extra>",
        )
    )
    if current:
        fig.add_vline(
            x=current,
            line_dash="dash",
            line_color="#f9fafb",
            annotation_text=f"Now ${current:,.2f}",
            annotation_position="top right",
        )
    fig.update_layout(
        template=PLOTLY_TEMPLATE,
        height=max(220, len(bars) * 48 + 60),
        margin=dict(l=20, r=80, t=30, b=20),
        xaxis_title="Price ($)",
        showlegend=False,
    )
    st.plotly_chart(fig, use_container_width=True)


# ── Comparison table ──────────────────────────────────────────────────────────


def _render_comparison_table(
    report: ResearchReport,
    vp: VariantPerceptionResult | None,
    cn: AnalystConsensus | None,
) -> None:
    st.subheader("Market vs. Our View")

    current = (cn.current_price if cn else None) or (
        report.dcf.current_price if report.dcf else None
    )
    dcf = report.dcf

    rows: list[dict] = []

    # Analyst recommendation
    if cn and cn.n_analysts > 0:
        rows.append(
            {
                "Metric": "Analyst recommendation",
                "Market / Consensus": (
                    f"{cn.recommendation_key.capitalize()} " f"({cn.n_analysts} analysts)"
                ),
                "Our Model": "—",
            }
        )

    # Target price
    cons_target = cn.target_price_mean if cn and cn.target_price_mean else None
    our_target = (
        vp.our_base_price
        if vp and vp.our_base_price
        else (dcf.base.implied_price if dcf and dcf.base else None)
    )
    if cons_target or our_target:
        rows.append(
            {
                "Metric": "Target price",
                "Market / Consensus": f"${cons_target:,.2f}" if cons_target else "—",
                "Our Model": f"${our_target:,.2f}" if our_target else "—",
            }
        )

    # Implied upside (relative to current price)
    cons_upside = cn.consensus_upside_pct if cn else None
    our_upside = (our_target / current - 1) if our_target and current and current > 0 else None
    if cons_upside is not None or our_upside is not None:
        rows.append(
            {
                "Metric": "Implied upside",
                "Market / Consensus": f"{cons_upside:+.1%}" if cons_upside is not None else "—",
                "Our Model": f"{our_upside:+.1%}" if our_upside is not None else "—",
            }
        )

    # Gap between our model and consensus
    if vp and vp.price_vs_consensus_pct is not None:
        direction = "above" if vp.price_vs_consensus_pct > 0 else "below"
        rows.append(
            {
                "Metric": "Our model vs. consensus",
                "Market / Consensus": "—",
                "Our Model": f"{abs(vp.price_vs_consensus_pct):.1%} {direction}",
            }
        )

    # Implied growth
    mkt_growth = (
        vp.market_implied_growth
        if vp and vp.market_implied_growth is not None
        else (dcf.implied_growth_rate if dcf else None)
    )
    our_growth = (
        vp.our_base_growth
        if vp and vp.our_base_growth is not None
        else (dcf.base.assumptions.revenue_growth_yr1_3 if dcf and dcf.base else None)
    )
    if mkt_growth is not None or our_growth is not None:
        rows.append(
            {
                "Metric": "Implied revenue growth",
                "Market / Consensus": f"{mkt_growth:.1%}" if mkt_growth is not None else "—",
                "Our Model": f"{our_growth:.1%}" if our_growth is not None else "—",
            }
        )

    # Estimate revision trend
    if cn and cn.n_analysts > 0:
        rows.append(
            {
                "Metric": "Estimate revisions (30d)",
                "Market / Consensus": (
                    f"{cn.revision_trend.capitalize()}  "
                    f"(↑{cn.eps_revisions_up_30d} / ↓{cn.eps_revisions_down_30d})"
                ),
                "Our Model": "—",
            }
        )
        if cn.upgrades_30d or cn.downgrades_30d:
            rows.append(
                {
                    "Metric": "Rating changes (30d)",
                    "Market / Consensus": (
                        f"↑{cn.upgrades_30d} upgrades / " f"↓{cn.downgrades_30d} downgrades"
                    ),
                    "Our Model": "—",
                }
            )

    if rows:
        import pandas as pd

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.caption("Run a force refresh or click ↻ Refresh Edge to populate this table.")

    if vp:
        mt = vp.mispricing_type.value.replace("_", " ").title()
        conv_color = {
            "HIGH": PALETTE["positive"],
            "MEDIUM": PALETTE["warn"],
            "LOW": PALETTE["neutral"],
        }.get(vp.conviction, PALETTE["neutral"])
        st.markdown(
            f"**Mispricing type:** `{mt}`  &nbsp;&nbsp; "
            f'<span style="color:{conv_color};font-weight:600">'
            f"Edge conviction: {vp.conviction}</span>",
            unsafe_allow_html=True,
        )


# ── Narrative ─────────────────────────────────────────────────────────────────


def _render_edge_narrative(vp: VariantPerceptionResult) -> None:
    if vp.our_edge:
        st.subheader("Why we differ")
        with st.container(border=True):
            st.markdown(vp.our_edge)

    if vp.consensus_misses:
        st.subheader("What consensus is missing")
        for miss in vp.consensus_misses:
            st.markdown(f"- {miss}")


# ── Analyst detail ────────────────────────────────────────────────────────────


def _render_analyst_detail(cn: AnalystConsensus) -> None:
    st.subheader("Analyst estimates")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("# Analysts", str(cn.n_analysts))

    col2.metric(
        "Recommendation",
        cn.recommendation_key.capitalize() if cn.recommendation_key else "—",
    )

    if cn.recommendation_mean is not None:
        label_map = {1: "Strong Buy", 2: "Buy", 3: "Hold", 4: "Sell", 5: "Strong Sell"}
        nearest = min(label_map, key=lambda k: abs(k - cn.recommendation_mean))
        col3.metric("Consensus score", f"{cn.recommendation_mean:.2f} — {label_map[nearest]}")
    else:
        col3.metric("Consensus score", "—")

    col4.metric(
        "Revision trend",
        cn.revision_trend.capitalize(),
        delta=f"↑{cn.eps_revisions_up_30d} / ↓{cn.eps_revisions_down_30d}",
    )

    # Consensus target range
    if cn.target_price_low and cn.target_price_high:
        st.caption(
            f"Target range: **${cn.target_price_low:,.2f}** (low) · "
            f"**${cn.target_price_mean:,.2f}** (mean) · "
            f"**${cn.target_price_high:,.2f}** (high)"
        )

    # Forward estimates table
    forward_rows: list[dict] = []
    if cn.eps_estimate_current_year is not None or cn.eps_estimate_next_year is not None:
        forward_rows.append(
            {
                "Metric": "EPS estimate",
                "Current Year": f"${cn.eps_estimate_current_year:.2f}"
                if cn.eps_estimate_current_year is not None
                else "—",
                "Next Year": f"${cn.eps_estimate_next_year:.2f}"
                if cn.eps_estimate_next_year is not None
                else "—",
            }
        )
    if cn.revenue_estimate_current_year is not None or cn.revenue_estimate_next_year is not None:
        forward_rows.append(
            {
                "Metric": "Revenue estimate",
                "Current Year": _abbrev(cn.revenue_estimate_current_year),
                "Next Year": _abbrev(cn.revenue_estimate_next_year),
            }
        )

    if forward_rows:
        import pandas as pd

        st.dataframe(pd.DataFrame(forward_rows), use_container_width=True, hide_index=True)

    if cn.fetched_at:
        st.caption(f"Consensus data fetched {cn.fetched_at:%Y-%m-%d %H:%M}.")


# ── Helpers ───────────────────────────────────────────────────────────────────


def _abbrev(v: float | None) -> str:
    if v is None:
        return "—"
    abs_v = abs(v)
    if abs_v >= 1e12:
        return f"${v / 1e12:,.2f}T"
    if abs_v >= 1e9:
        return f"${v / 1e9:,.2f}B"
    if abs_v >= 1e6:
        return f"${v / 1e6:,.1f}M"
    return f"${v:,.0f}"
