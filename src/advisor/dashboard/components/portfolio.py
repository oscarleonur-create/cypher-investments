"""Portfolio review renderer — thesis health + events across held companies."""

from __future__ import annotations

import pandas as pd
import streamlit as st

from advisor.dashboard.theme import PALETTE
from advisor.research.portfolio_review import PortfolioReview

_ATTN_ICON = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}


def render_portfolio(review: PortfolioReview) -> None:
    if not review.positions:
        st.warning(
            "No equity holdings found. Check TastyTrade credentials in `.env` and that "
            "the selected accounts have open positions."
        )
        return

    # ── Top metrics ──────────────────────────────────────────────────────────
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Holdings", len(review.positions))
    c2.metric("Invalidated", review.n_invalidated)
    c3.metric("Caution", review.n_caution)
    c4.metric("Near-term events", review.n_near_term_events)

    st.divider()

    # ── Summary table ────────────────────────────────────────────────────────
    rows = []
    for p in review.positions:
        rows.append(
            {
                "": _ATTN_ICON.get(p.attention, ""),
                "Ticker": p.symbol,
                "Accounts": ", ".join(p.accounts) or "—",
                "Thesis": p.thesis_status.replace("_", " "),
                "Conviction": p.conviction or "—",
                "Base upside": f"{p.base_upside:+.0%}" if p.base_upside is not None else "—",
                "Next earnings": p.next_earnings_date or "—",
                "Events": len(p.near_term_catalysts),
            }
        )
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
