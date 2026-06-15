"""Portfolio review renderer — thesis health, concentration, valuation & market.

Organised into tabs:
  • Overview      — thesis health table + per-ticker detail (the original view).
  • Concentration — weight donuts (by holding & sector) + concentration metrics.
  • Valuation     — remaining upside per holding (overpriced / priced-in flags).
  • Rotation      — sector weights vs sector-ETF momentum.
  • Market (VIX)  — broad-market volatility regime (portfolio-independent).
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from advisor.dashboard.components import portfolio_charts as charts
from advisor.dashboard.theme import PALETTE
from advisor.research import portfolio_analytics as analytics
from advisor.research.portfolio_review import PortfolioReview

_ATTN_ICON = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}
_FLAG_LABEL = {
    "OVERPRICED": "🔴 Overpriced",
    "PRICED_IN": "🟡 Priced in",
    "UPSIDE": "🟢 Upside",
    "UNKNOWN": "⚪ No model",
}


def render_portfolio(review: PortfolioReview) -> None:
    if not review.positions:
        st.warning(
            "No equity holdings found. Check TastyTrade credentials in `.env` and that "
            "the selected accounts have open positions."
        )
        # Market tab is portfolio-independent — still worth showing.
        _render_market_tab()
        return

    tab_overview, tab_conc, tab_val, tab_rot, tab_mkt = st.tabs(
        ["Overview", "Concentration", "Valuation risk", "Sector rotation", "Market (VIX)"]
    )

    with tab_overview:
        _render_overview(review)
    with tab_conc:
        _render_concentration(review)
    with tab_val:
        _render_valuation(review)
    with tab_rot:
        _render_rotation(review)
    with tab_mkt:
        _render_market_tab()


# ── Overview (original view) ─────────────────────────────────────────────────


def _render_overview(review: PortfolioReview) -> None:
    has_value = review.total_market_value > 0

    cols = st.columns(5 if has_value else 4)
    cols[0].metric("Holdings", len(review.positions))
    cols[1].metric("Invalidated", review.n_invalidated)
    cols[2].metric("Caution", review.n_caution)
    cols[3].metric("Near-term events", review.n_near_term_events)
    if has_value:
        cols[4].metric("Book value", f"${review.total_market_value:,.0f}")

    st.divider()

    # ── Summary table ────────────────────────────────────────────────────────
    rows = []
    for p in review.positions:
        row = {
            "": _ATTN_ICON.get(p.attention, ""),
            "Ticker": p.symbol,
        }
        if has_value:
            row["Weight"] = f"{p.weight:.1%}" if p.weight else "—"
        row.update(
            {
                "Accounts": ", ".join(p.accounts) or "—",
                "Thesis": p.thesis_status.replace("_", " "),
                "Conviction": p.conviction or "—",
                "Base upside": f"{p.base_upside:+.0%}" if p.base_upside is not None else "—",
                "Next earnings": p.next_earnings_date or "—",
                "Events": len(p.near_term_catalysts),
            }
        )
        rows.append(row)
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()

    # ── Per-ticker detail ────────────────────────────────────────────────────
    st.subheader("Detail")
    for p in review.positions:
        icon = _ATTN_ICON.get(p.attention, "")
        title = f"{icon} {p.symbol}"
        if p.company_name and p.company_name != p.symbol:
            title += f" — {p.company_name}"
        title += f"  ·  {p.thesis_status.replace('_', ' ')}"

        with st.expander(title, expanded=(p.attention == "HIGH")):
            meta = []
            if p.conviction:
                meta.append(f"**Conviction:** {p.conviction}")
            if p.base_target is not None:
                up = f" ({p.base_upside:+.0%})" if p.base_upside is not None else ""
                meta.append(f"**DCF base target:** ${p.base_target:,.2f}{up}")
            if meta:
                st.markdown(" · ".join(meta))

            if p.report_was_built:
                st.caption("New research report built for this run.")
            elif not p.has_report:
                st.caption("No research report on file — thesis monitoring limited.")

            if p.kpi_alerts:
                st.markdown("**KPI alerts**")
                for a in p.kpi_alerts:
                    alert_color = PALETTE["negative"] if a.startswith("⚠") else PALETTE["warn"]
                    st.markdown(
                        f"<div style='border-left:3px solid {alert_color};padding:6px 10px;"
                        f"margin-bottom:4px;background:{alert_color}18;border-radius:3px'>"
                        f"{a}</div>",
                        unsafe_allow_html=True,
                    )

            if p.next_earnings_date:
                st.markdown(f"**Next earnings:** {p.next_earnings_date}")

            if p.near_term_catalysts:
                st.markdown("**Near-term catalysts**")
                for c in p.near_term_catalysts:
                    st.markdown(f"- {c}")

            if p.error:
                st.warning(p.error)


# ── Concentration ────────────────────────────────────────────────────────────


def _needs_refresh() -> None:
    st.info(
        "No position values on this review. Hit **Refresh from TastyTrade** to pull "
        "live market values and compute portfolio weights."
    )


def _render_concentration(review: PortfolioReview) -> None:
    if review.total_market_value <= 0:
        _needs_refresh()
        return

    metrics = analytics.concentration_metrics(review)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Holdings", metrics["n_holdings"])
    c2.metric("Top holding", f"{metrics['top_weight']:.1%}")
    c3.metric("Top 5", f"{metrics['top5_weight']:.1%}")
    c4.metric("Effective N", f"{metrics['effective_n']:.1f}", help="1 / HHI — diversification")

    st.divider()

    col_h, col_s = st.columns(2)
    with col_h:
        held = [(p.symbol, p.weight) for p in review.positions if p.weight > 0]
        held.sort(key=lambda kv: kv[1], reverse=True)
        labels = [s for s, _ in held]
        weights = [w for _, w in held]
        st.plotly_chart(
            charts.concentration_donut(labels, weights, "By holding"),
            use_container_width=True,
        )
    with col_s:
        sectors = analytics.sector_breakdown(review)
        st.plotly_chart(
            charts.concentration_donut(list(sectors.keys()), list(sectors.values()), "By sector"),
            use_container_width=True,
        )


# ── Valuation risk ───────────────────────────────────────────────────────────


def _render_valuation(review: PortfolioReview) -> None:
    rows = analytics.valuation_table(review)
    priced = [r for r in rows if r["upside"] is not None]

    n_over = sum(1 for r in priced if r["flag"] == "OVERPRICED")
    n_in = sum(1 for r in priced if r["flag"] == "PRICED_IN")
    c1, c2, c3 = st.columns(3)
    c1.metric("Overpriced", n_over, help="Trading above DCF/Bayesian fair value")
    c2.metric("Priced in", n_in, help="Less than 5% upside left")
    c3.metric("No model", sum(1 for r in rows if r["upside"] is None))

    if priced:
        st.plotly_chart(charts.valuation_bar(rows), use_container_width=True)

    table = []
    for r in rows:
        table.append(
            {
                "Ticker": r["symbol"],
                "Weight": f"{r['weight']:.1%}" if r["weight"] else "—",
                "Flag": _FLAG_LABEL.get(r["flag"], r["flag"]),
                "DCF upside": f"{r['base_upside']:+.0%}" if r["base_upside"] is not None else "—",
                "Bayes upside": (
                    f"{r['bayes_upside']:+.0%}" if r["bayes_upside"] is not None else "—"
                ),
                "P(undervalued)": (
                    f"{r['bayes_prob_undervalued']:.0%}"
                    if r["bayes_prob_undervalued"] is not None
                    else "—"
                ),
            }
        )
    st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)
    st.caption(
        "Upside prefers the DCF base case, falling back to the Bayesian posterior. "
        "Names with no research report show as *No model* — run research to value them."
    )


# ── Sector rotation ──────────────────────────────────────────────────────────


def _render_rotation(review: PortfolioReview) -> None:
    if review.total_market_value <= 0:
        _needs_refresh()
        return

    sectors = analytics.sector_breakdown(review)
    known = {s: w for s, w in sectors.items() if s != "Unknown"}
    if not known:
        st.info("No classified sectors yet — sectors are filled in on the next refresh.")
        return

    with st.spinner("Fetching sector-ETF momentum…"):
        rotation = _cached_sector_rotation(tuple(known.keys()))

    if not rotation:
        st.warning("Could not fetch sector momentum (market data unavailable).")
        # Still show the weights so the tab isn't empty.
        st.bar_chart(pd.Series(known, name="Weight"))
        return

    ordered = [s for s in known if s in rotation]
    weights = [known[s] for s in ordered]
    rel_3m = [rotation[s]["rel_3m"] for s in ordered]
    st.plotly_chart(
        charts.sector_rotation_bar(ordered, weights, rel_3m),
        use_container_width=True,
    )

    table = []
    for s in ordered:
        r = rotation[s]
        table.append(
            {
                "Sector": s,
                "ETF": r["etf"],
                "Your weight": f"{known[s]:.1%}",
                "ETF 3-mo": f"{r['etf_return_3m']:+.1%}" if r["etf_return_3m"] is not None else "—",
                "vs SPY (3m)": f"{r['rel_3m']:+.1%}" if r["rel_3m"] is not None else "—",
                "vs SPY (1m)": f"{r['rel_1m']:+.1%}" if r["rel_1m"] is not None else "—",
                "Trend": "📈 Leading" if r["leading"] else "📉 Lagging",
            }
        )
    st.dataframe(pd.DataFrame(table), use_container_width=True, hide_index=True)
    st.caption(
        "Positive **vs SPY** means the sector is outperforming the market — watch for being "
        "overweight a *lagging* sector or underweight a *leading* one."
    )


# ── Market (VIX) ─────────────────────────────────────────────────────────────


def _render_market_tab() -> None:
    ctx = _cached_market_context()
    vix = ctx.get("vix")
    regime = ctx.get("regime")

    if vix is None:
        st.warning("VIX data unavailable right now.")
        return

    badge_color = PALETTE["neutral"]
    if regime:
        badge_color = {
            "Calm": PALETTE["positive"],
            "Normal": PALETTE["accent"],
            "Stressed": PALETTE["negative"],
        }.get(regime["label"], PALETTE["neutral"])
        st.markdown(
            f"<span style='background:{badge_color}22;border:1px solid {badge_color};"
            f"color:{badge_color};padding:4px 12px;border-radius:12px;font-weight:600'>"
            f"Regime: {regime['label']}</span>",
            unsafe_allow_html=True,
        )
        st.caption(
            f"HMM read as of {regime.get('date', '—')} · model VIX {regime.get('vix', 0):.1f}"
        )
    else:
        st.caption(
            "No trained regime model on disk — showing VIX only. "
            "Train one with `advisor ml regime` to enable the calm/normal/stressed read."
        )

    col_g, col_h = st.columns([1, 2])
    with col_g:
        st.plotly_chart(
            charts.vix_gauge(vix["current"], vix["percentile_1y"]),
            use_container_width=True,
        )
    with col_h:
        dates = [h["date"] for h in vix["history"]]
        vals = [h["vix"] for h in vix["history"]]
        st.plotly_chart(charts.vix_history(dates, vals, vix["sma20"]), use_container_width=True)


# ── Cached data fetches ──────────────────────────────────────────────────────


@st.cache_data(ttl=900, show_spinner=False)
def _cached_market_context() -> dict:
    from advisor.research.market_context import get_market_context

    return get_market_context()


@st.cache_data(ttl=900, show_spinner=False)
def _cached_sector_rotation(sectors: tuple[str, ...]) -> dict:
    return analytics.sector_rotation(list(sectors))
