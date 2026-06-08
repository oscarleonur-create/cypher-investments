"""Memo tab — Investment Committee format (90-second read).

§0 Executive Summary + Edge
§1 Business Quality (moat + management)
§2 Valuation snapshot (bear / base / bull vs. consensus)
§3 Key Catalysts
§4 Downside-First
§5 Position Sizing
§6 Exit Triggers
"""

from __future__ import annotations

import streamlit as st

from advisor.dashboard import data as _data
from advisor.dashboard import state as _state
from advisor.dashboard.theme import PALETTE
from advisor.research.models import InvestmentMemo, ResearchReport


def render_memo(report: ResearchReport) -> None:
    memo = report.investment_memo

    if memo is None:
        _render_no_memo(report)
        return

    _render_executive_summary(memo)
    st.divider()
    _render_business_quality(memo, report)
    st.divider()
    _render_valuation(memo)
    st.divider()
    _render_catalysts(memo)
    st.divider()
    _render_downside(memo)
    st.divider()
    _render_sizing(memo)
    st.divider()
    _render_exit_triggers(memo)
    st.divider()
    _render_refresh_footer(report)


# ── §0 Executive Summary ──────────────────────────────────────────────────────


def _render_executive_summary(memo: InvestmentMemo) -> None:
    col_left, col_right = st.columns([3, 1])

    with col_left:
        st.subheader(f"Investment Memo — {memo.symbol}")
        if memo.company_name and memo.company_name != memo.symbol:
            st.caption(memo.company_name)

    conviction_color = {
        "HIGH": PALETTE["positive"],
        "MEDIUM": PALETTE["warn"],
        "LOW": PALETTE["neutral"],
    }.get(memo.conviction, PALETTE["neutral"])

    col_right.markdown(
        f"<div style='text-align:right;padding-top:8px'>"
        f"<span style='font-size:0.8em;color:#9ca3af'>Conviction</span><br>"
        f"<span style='font-size:1.4em;font-weight:700;color:{conviction_color}'>"
        f"{memo.conviction}</span></div>",
        unsafe_allow_html=True,
    )

    if memo.executive_summary:
        st.markdown(f"> {memo.executive_summary}")

    if memo.key_insight:
        st.info(f"**Edge:** {memo.key_insight}")

    if memo.mispricing_type and memo.mispricing_type not in ("none", ""):
        label = memo.mispricing_type.replace("_", " ").title()
        st.caption(f"Mispricing type: `{label}`")


# ── §1 Business Quality ───────────────────────────────────────────────────────


def _render_business_quality(memo: InvestmentMemo, report: ResearchReport | None = None) -> None:
    st.subheader("Business Quality")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Moat**")
        moat_label = memo.moat_type.replace("_", " ").title() if memo.moat_type else "Unknown"
        st.markdown(f"`{moat_label}`")

        ind = report.industry if report else None
        if ind is not None and ind.moat_strength is not None:
            tier = (
                "Wide"
                if ind.moat_strength >= 7.5
                else "Narrow"
                if ind.moat_strength >= 4
                else "None"
            )
            st.markdown(
                f"<span style='font-size:1.3em;font-weight:700'>{ind.moat_strength:.1f}</span>"
                f"<span style='color:#9ca3af'> / 10 — {tier}</span>",
                unsafe_allow_html=True,
            )
        if memo.moat_description:
            st.caption(memo.moat_description)

        if ind is not None and ind.competitive_position:
            _POSITION_COLOR = {
                "leader": "#16a34a",
                "challenger": "#2563eb",
                "niche": "#6b7280",
                "laggard": "#dc2626",
            }
            color = _POSITION_COLOR.get(ind.competitive_position, "#6b7280")
            st.markdown(
                f"<span style='background:{color};color:#fff;padding:1px 8px;"
                f"border-radius:10px;font-size:0.8em'>{ind.competitive_position.title()}</span>",
                unsafe_allow_html=True,
            )

        if ind is not None and ind.competitive_advantages:
            st.markdown("**Advantages**")
            for adv in ind.competitive_advantages:
                st.markdown(f"- {adv}")

    with col2:
        st.markdown("**Management Quality**")
        if memo.management_quality_score is not None:
            tier_color = {
                "HIGH": PALETTE["positive"],
                "MEDIUM": PALETTE["warn"],
                "LOW": PALETTE["negative"],
            }.get(memo.management_quality_tier, PALETTE["neutral"])
            st.markdown(
                f"<span style='font-size:1.6em;font-weight:700;color:{tier_color}'>"
                f"{memo.management_quality_score:.0f}</span>"
                f"<span style='color:#9ca3af'> / 100 — {memo.management_quality_tier}</span>",
                unsafe_allow_html=True,
            )
        else:
            st.caption("Management quality not computed — run a full refresh.")


# ── §2 Valuation ─────────────────────────────────────────────────────────────


def _render_valuation(memo: InvestmentMemo) -> None:
    st.subheader("Valuation")

    cols = st.columns(4)
    current = memo.current_price

    def _price_metric(col, label: str, target: float | None, upside: float | None) -> None:
        if target is None:
            col.metric(label, "—")
            return
        delta_str = f"{upside:+.1%}" if upside is not None else None
        col.metric(label, f"${target:,.2f}", delta=delta_str)

    cols[0].metric("Current Price", f"${current:,.2f}" if current else "—")
    _price_metric(cols[1], "Bear Target", memo.bear_target, memo.bear_upside)
    _price_metric(cols[2], "Base Target", memo.base_target, memo.base_upside)
    _price_metric(cols[3], "Bull Target", memo.bull_target, memo.bull_upside)

    if memo.consensus_target or memo.n_analysts:
        cn_upside = f"{memo.consensus_upside:+.1%}" if memo.consensus_upside is not None else ""
        st.caption(
            f"Street consensus: **${memo.consensus_target:,.2f}** {cn_upside} "
            f"({memo.n_analysts} analysts)"
            if memo.consensus_target
            else f"Street consensus: {memo.n_analysts} analysts"
        )


# ── §3 Catalysts ─────────────────────────────────────────────────────────────


def _render_catalysts(memo: InvestmentMemo) -> None:
    st.subheader("Key Catalysts")

    if memo.key_catalysts:
        for cat in memo.key_catalysts:
            st.markdown(f"- {cat}")
    else:
        st.caption("No catalysts captured — run a full refresh to populate.")


# ── §4 Downside-First ─────────────────────────────────────────────────────────


def _render_downside(memo: InvestmentMemo) -> None:
    st.subheader("Downside-First")

    if memo.max_loss_scenario:
        neg = PALETTE["negative"]
        st.markdown(
            f"<div style='border-left:4px solid {neg};padding:10px 16px;"
            f"background:{neg}18;border-radius:4px'>"
            f"<p style='margin:0;font-weight:600;color:{neg}'>Max-Loss Scenario</p>"
            f"<p style='margin:6px 0 0'>{memo.max_loss_scenario}</p>"
            f"</div>",
            unsafe_allow_html=True,
        )

    if memo.bear_case_deep_dive:
        with st.expander("Bear case deep-dive"):
            st.markdown(memo.bear_case_deep_dive)


# ── §5 Position Sizing ────────────────────────────────────────────────────────


def _render_sizing(memo: InvestmentMemo) -> None:
    st.subheader("Position Sizing")

    col1, col2 = st.columns(2)

    conviction_color = {
        "HIGH": PALETTE["positive"],
        "MEDIUM": PALETTE["warn"],
        "LOW": PALETTE["neutral"],
    }.get(memo.conviction, PALETTE["neutral"])

    col1.markdown(
        f"<div style='padding:12px;background:#1f2937;border-radius:6px;text-align:center'>"
        f"<p style='margin:0;font-size:0.85em;color:#9ca3af'>Suggested range</p>"
        f"<p style='margin:4px 0;font-size:1.8em;font-weight:700;color:{conviction_color}'>"
        f"{memo.position_size_pct_low:.1f}% – {memo.position_size_pct_high:.1f}%</p>"
        f"<p style='margin:0;font-size:0.8em;color:#6b7280'>of portfolio</p>"
        f"</div>",
        unsafe_allow_html=True,
    )

    if memo.sizing_rationale:
        col2.markdown(f"**Rationale:** {memo.sizing_rationale}")


# ── §6 Exit Triggers ──────────────────────────────────────────────────────────


def _render_exit_triggers(memo: InvestmentMemo) -> None:
    st.subheader("Exit Triggers")

    if memo.exit_triggers:
        warn = PALETTE["warn"]
        for trigger in memo.exit_triggers:
            st.markdown(
                f"<div style='border-left:3px solid {warn};padding:6px 12px;"
                f"margin-bottom:6px;background:#1f293780;border-radius:3px'>"
                f"⚠️ {trigger}</div>",
                unsafe_allow_html=True,
            )
    else:
        st.caption("Exit triggers not generated. Run a full refresh.")


# ── Empty state / refresh ─────────────────────────────────────────────────────


def _render_no_memo(report: ResearchReport) -> None:
    st.info(
        "Investment memo not yet generated for this report. "
        "Click **↻ Rebuild Memo** below to synthesise it from the cached report data."
    )
    _render_refresh_footer(report)


def _render_refresh_footer(report: ResearchReport) -> None:
    col1, col2 = st.columns([3, 1])
    col1.caption(
        "The memo synthesises all research layers into IC-presentation format. "
        "Requires an OpenRouter API key for the executive summary and exit triggers. "
        "Deterministic fields (valuation, sizing, catalysts) populate without an API key."
    )
    if col2.button(
        "↻ Rebuild Memo",
        key="inline_refresh_memo",
        type="primary",
        use_container_width=True,
    ):
        with st.spinner("Rebuilding investment memo…"):
            try:
                updated = _data.refresh_layer(report, layer="memo")
                _state.set_report(updated)
                _state.log_event(report.symbol, "refresh", {"layer": "memo"})
            except Exception as exc:  # noqa: BLE001
                st.error(f"Memo rebuild failed: {exc}")
        st.rerun()
