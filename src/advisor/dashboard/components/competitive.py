"""Competitive position dashboard component.

Renders sector context, moat assessment, Porter's 5 Forces, and competitive
advantages for the Research workstation.
"""

from __future__ import annotations

import streamlit as st

from advisor.research.models import PortersForceStrength, ResearchReport

_POSITION_COLOR = {
    "leader": "#16a34a",
    "challenger": "#2563eb",
    "niche": "#6b7280",
    "laggard": "#dc2626",
}

_FORCE_ICON = {
    PortersForceStrength.LOW: "🟢",
    PortersForceStrength.HIGH: "🔴",
    PortersForceStrength.MEDIUM: "🟡",
}

_FORCE_LABEL = {
    PortersForceStrength.LOW: "Low",
    PortersForceStrength.HIGH: "High",
    PortersForceStrength.MEDIUM: "Medium",
}


def _moat_tier(strength: float) -> str:
    if strength >= 7.5:
        return "Wide"
    if strength >= 4.0:
        return "Narrow"
    return "None"


def render_competitive(report: ResearchReport) -> None:
    ind = report.industry

    if ind is None:
        st.info("Run the full report to generate competitive analysis.")
        return

    # ── Header: sector / industry / competitive position ─────────────────────
    hcols = st.columns([2, 2, 2, 2])
    with hcols[0]:
        st.markdown("**Sector**")
        st.markdown(f"`{ind.sector or '—'}`")
    with hcols[1]:
        st.markdown("**Industry**")
        st.markdown(f"`{ind.industry or '—'}`")
    with hcols[2]:
        st.markdown("**Position**")
        pos = ind.competitive_position or ""
        color = _POSITION_COLOR.get(pos, "#6b7280")
        label = pos.title() if pos else "—"
        st.markdown(
            f"<span style='background:{color};color:#fff;padding:2px 10px;"
            f"border-radius:12px;font-weight:600'>{label}</span>",
            unsafe_allow_html=True,
        )
    with hcols[3]:
        st.markdown("**Moat Type**")
        moat_label = ind.moat_type.value.replace("_", " ").title()
        st.markdown(f"`{moat_label}`")

    st.divider()

    # ── Moat panel ───────────────────────────────────────────────────────────
    st.subheader("Moat Assessment")
    mcols = st.columns(2)

    with mcols[0]:
        if ind.moat_strength is not None:
            tier = _moat_tier(ind.moat_strength)
            st.metric(
                label=f"Moat Strength — {tier}",
                value=f"{ind.moat_strength:.1f} / 10",
            )
            st.progress(ind.moat_strength / 10.0)
        if ind.moat_description:
            st.caption(ind.moat_description)

    with mcols[1]:
        if ind.competitive_advantages:
            st.markdown("**Competitive Advantages**")
            for adv in ind.competitive_advantages:
                st.markdown(f"- {adv}")
        elif ind.market_share_narrative:
            st.caption(ind.market_share_narrative)

    if ind.market_share_narrative and ind.competitive_advantages:
        with st.expander("Market share narrative"):
            st.write(ind.market_share_narrative)

    st.divider()

    # ── Sector context: tailwinds / headwinds ─────────────────────────────────
    st.subheader("Sector Context")
    tcols = st.columns(2)

    with tcols[0]:
        st.markdown(
            "<span style='color:#16a34a;font-weight:700'>▲ Tailwinds</span>",
            unsafe_allow_html=True,
        )
        if ind.sector_tailwinds:
            for t in ind.sector_tailwinds:
                st.markdown(f"- {t}")
        else:
            st.caption("No tailwinds data — refresh industry layer.")

    with tcols[1]:
        st.markdown(
            "<span style='color:#dc2626;font-weight:700'>▼ Headwinds</span>",
            unsafe_allow_html=True,
        )
        if ind.sector_headwinds:
            for h in ind.sector_headwinds:
                st.markdown(f"- {h}")
        else:
            st.caption("No headwinds data — refresh industry layer.")

    st.divider()

    # ── Porter's 5 Forces ────────────────────────────────────────────────────
    st.subheader("Porter's 5 Forces")
    pf = ind.porters_forces
    if pf is None:
        st.caption("Porter's forces not available.")
    else:
        forces = [
            ("Competitive Rivalry", pf.competitive_rivalry),
            ("Supplier Power", pf.supplier_power),
            ("Buyer Power", pf.buyer_power),
            ("Threat of New Entrants", pf.threat_of_new_entrants),
            ("Threat of Substitutes", pf.threat_of_substitutes),
        ]
        for force_name, strength in forces:
            icon = _FORCE_ICON[strength]
            label = _FORCE_LABEL[strength]
            st.markdown(f"{icon} **{force_name}** — {label}")

        if pf.rationale:
            with st.expander("Porter's rationale"):
                st.write(pf.rationale)

    # ── Key competitors ───────────────────────────────────────────────────────
    if ind.key_competitors:
        st.divider()
        st.subheader("Key Competitors")
        st.markdown(" · ".join(f"`{c}`" for c in ind.key_competitors))
