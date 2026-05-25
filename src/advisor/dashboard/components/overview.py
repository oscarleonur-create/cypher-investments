"""Overview tab — thesis, quick stats, red-flag chips, near-term catalysts."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import streamlit as st

from advisor.dashboard import ai
from advisor.dashboard.persistence import get_store
from advisor.dashboard.theme import PALETTE, severity_chip
from advisor.research.models import CatalystItem, ResearchReport


def render_overview(report: ResearchReport, session_id: str | None = None) -> None:
    _render_thesis(report)
    st.divider()
    _render_quick_stats(report)
    st.divider()
    _render_red_flags(report)
    st.divider()
    _render_catalyst_preview(report)
    st.divider()
    _render_ai_thesis(report, session_id)


def _render_thesis(report: ResearchReport) -> None:
    st.subheader("Investment thesis")
    thesis = report.thesis
    if thesis is None:
        st.info("No thesis generated yet.")
        return
    if thesis.summary:
        st.markdown(thesis.summary)
    cols = st.columns(3)
    for col, label, scenario in zip(
        cols,
        ["Bear", "Base", "Bull"],
        [thesis.bear, thesis.base, thesis.bull],
    ):
        with col:
            if scenario is None:
                st.caption(f"**{label}** — not modeled")
                continue
            tgt = f"${scenario.target_price:,.2f}" if scenario.target_price else "—"
            ups = f" ({scenario.upside_pct:+.1%})" if scenario.upside_pct is not None else ""
            st.markdown(f"**{label}** · target {tgt}{ups}")
            if scenario.description:
                st.caption(scenario.description)
    if thesis.conviction:
        st.caption(f"Conviction: **{thesis.conviction}**")


def _render_quick_stats(report: ResearchReport) -> None:
    st.subheader("Quick stats")
    latest_ratio = report.ratios.latest() if report.ratios else None
    snap = report.multiples.subject if report.multiples else None
    dcf_base_upside = report.dcf.base.upside_pct if report.dcf and report.dcf.base else None
    rev_cagr = _revenue_cagr(report)
    shares_cagr = report.ratios.share_count_cagr_3y if report.ratios else None
    debt_ebitda = latest_ratio.debt_to_ebitda if latest_ratio else None
    fcf_ni = latest_ratio.fcf_to_net_income if latest_ratio else None

    row1 = st.columns(3)
    row1[0].metric("Revenue 5y CAGR", _pct(rev_cagr))
    row1[1].metric("FCF margin", _pct(latest_ratio.fcf_margin if latest_ratio else None))
    row1[2].metric("ROIC", _pct(latest_ratio.roic if latest_ratio else None))

    row2 = st.columns(3)
    row2[0].metric("Debt / EBITDA", _ratio(debt_ebitda))
    row2[1].metric("FCF / Net income", _ratio(fcf_ni))
    row2[2].metric("Share count CAGR (3y)", _pct(shares_cagr))

    row3 = st.columns(3)
    row3[0].metric("Trailing P/E", _ratio(snap.pe_trailing if snap else None))
    row3[1].metric("EV / EBITDA", _ratio(snap.ev_to_ebitda if snap else None))
    row3[2].metric(
        "DCF upside (base)",
        _pct(dcf_base_upside),
        delta=_pct(dcf_base_upside) if dcf_base_upside is not None else None,
    )


def _render_red_flags(report: ResearchReport) -> None:
    st.subheader("Red flags")
    rf = report.red_flags
    if rf is None or not rf.flags:
        green = PALETTE["positive"]
        st.markdown(
            f"<span style='color:{green}'>No red flags detected.</span>",
            unsafe_allow_html=True,
        )
        return
    chips = "".join(severity_chip(f.severity.value, f.code) for f in rf.flags)
    st.markdown(chips, unsafe_allow_html=True)
    with st.expander(f"Flag details ({len(rf.flags)})", expanded=False):
        for f in rf.flags:
            st.markdown(f"**[{f.severity.value}] {f.title}** — {f.detail}")


def _render_catalyst_preview(report: ResearchReport) -> None:
    st.subheader("Near-term catalysts (next 90 days)")
    cr = report.catalyst_risk
    if cr is None or not cr.catalysts:
        st.caption("No catalysts on file.")
        return

    today = date.today()
    horizon = today + timedelta(days=90)
    # Don't trust `is_near_term` — it was captured when the catalyst was first
    # ingested and goes stale. Re-derive from the actual `expected_date`.
    parsed: list[tuple[date | None, CatalystItem]] = []
    for c in cr.catalysts:
        parsed.append((_parse_catalyst_date(c.expected_date), c))

    upcoming = [(d, c) for d, c in parsed if d is not None and today <= d <= horizon]
    quarter_only = [
        (None, c)
        for d, c in parsed
        if d is None and _quarter_in_window(c.expected_date, today, horizon)
    ]
    items = sorted(upcoming, key=lambda t: t[0]) + quarter_only

    past = sum(1 for d, _ in parsed if d is not None and d < today)
    far = sum(1 for d, _ in parsed if d is not None and d > horizon)

    if not items:
        st.caption("No catalysts in the next 90 days.")
    else:
        for d, c in items:
            when = d.isoformat() if d else (c.expected_date or "TBD")
            days_str = f" · in {(d - today).days}d" if d else ""
            st.markdown(f"- **{when}**{days_str} · `{c.catalyst_type.value}` · {c.description}")

    extras = []
    if past:
        extras.append(f"{past} past")
    if far:
        extras.append(f"{far} beyond 90d")
    if extras:
        st.caption(
            f"Hidden: {', '.join(extras)}. "
            f"Full list lives in the Catalysts & Risks tab (step 5)."
        )
    if cr.fetched_at:
        st.caption(
            f"Catalyst data fetched {cr.fetched_at:%Y-%m-%d %H:%M}. "
            "Use **Force refresh** in the sidebar if it looks stale."
        )


def _render_ai_thesis(report: ResearchReport, session_id: str | None) -> None:
    st.subheader("AI thesis")

    if not ai.is_configured():
        st.info(
            "Set `RESEARCH_AGENT_OPENROUTER_API_KEY` (in your shell or in a `.env` "
            "at the project root) to enable on-demand thesis generation."
        )
        return

    if session_id is None:
        st.info("Need an active session to cache AI theses.")
        return

    # Look up the most recent cached generation for this (session, symbol)
    cached: dict | None = None
    store = get_store()
    try:
        events = store.list_events(session_id, symbol=report.symbol, limit=100)
    finally:
        store.close()
    for e in events:
        if e["event_type"] == "ai_thesis" and e["payload"].get("thesis_md"):
            cached = e
            break

    cols = st.columns([1, 1, 4])
    generate = cols[0].button("✨ Generate AI thesis", type="primary", use_container_width=True)
    regenerate = cols[1].button("Regenerate", use_container_width=True) if cached else False

    if generate or regenerate:
        try:
            with st.spinner("Synthesizing thesis…"):
                thesis_md = ai.generate_thesis(report)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Generation failed: {exc}")
            return

        store = get_store()
        try:
            store.log_event(
                session_id,
                report.symbol,
                "ai_thesis",
                {
                    "thesis_md": thesis_md,
                    "model": _model_label(),
                    "generated_at": datetime.now().isoformat(),
                },
            )
        finally:
            store.close()
        st.rerun()

    # Render the most recent cached thesis (or just-generated, after rerun)
    if cached:
        thesis_md = cached["payload"].get("thesis_md", "")
        generated_at = cached["payload"].get("generated_at") or cached["ts"]
        model = cached["payload"].get("model", "")
        st.caption(f"Generated {generated_at[:19]} · model: {model or 'configured default'}")
        with st.container(border=True):
            st.markdown(thesis_md)
        if st.button("📝 Append to notes", key="append_thesis"):
            _append_thesis_to_notes(session_id, report.symbol, thesis_md, generated_at)
    elif not generate and not regenerate:
        st.caption(
            "Click **✨ Generate AI thesis** to synthesize a balanced bull/bear "
            "narrative from the data on this page. Results cache per session."
        )


def _append_thesis_to_notes(
    session_id: str, symbol: str, thesis_md: str, generated_at: str
) -> None:
    store = get_store()
    try:
        existing = store.load_note(session_id, symbol)
        body = existing["note_md"] if existing else ""
        header = f"\n\n---\n\n## AI thesis (generated {generated_at[:19]})\n\n"
        store.save_note(session_id, symbol, body + header + thesis_md)
        store.log_event(session_id, symbol, "note_added", {"source": "ai_thesis"})
    finally:
        store.close()
    st.success("Appended to notes.")


def _model_label() -> str:
    try:
        from research_agent.config import ResearchConfig

        return ResearchConfig().llm_model
    except Exception:  # noqa: BLE001
        return ""


# ── Helpers ──────────────────────────────────────────────────────────────────


def _revenue_cagr(report: ResearchReport) -> float | None:
    """Compound annual growth rate of revenue over the available period."""
    if report.statements is None or not report.statements.income:
        return None
    series = [p.revenue for p in report.statements.income if p.revenue is not None]
    if len(series) < 2 or series[-1] in (None, 0):
        return None
    # `income` is ordered most-recent-first by the extractor
    latest = series[0]
    oldest = series[-1]
    years = len(series) - 1
    if oldest <= 0 or years <= 0:
        return None
    try:
        return (latest / oldest) ** (1.0 / years) - 1.0
    except (ZeroDivisionError, ValueError):
        return None


def _parse_catalyst_date(expected_date: str) -> date | None:
    """Parse an ISO date prefix from `expected_date`. Returns None for quarter
    strings ('2026-Q2') or unparseable values."""
    if not expected_date:
        return None
    try:
        return date.fromisoformat(expected_date[:10])
    except ValueError:
        return None


_QUARTER_MONTHS = {"Q1": (1, 3), "Q2": (4, 6), "Q3": (7, 9), "Q4": (10, 12)}


def _quarter_in_window(expected_date: str, today: date, horizon: date) -> bool:
    """Return True if a quarter-string catalyst (e.g. '2026-Q2') overlaps the window."""
    if not expected_date or "Q" not in expected_date:
        return False
    try:
        year_str, q = expected_date.replace(" ", "").split("-Q")
        year = int(year_str)
        m_start, m_end = _QUARTER_MONTHS[f"Q{q}"]
    except (KeyError, ValueError):
        return False
    from calendar import monthrange

    q_start = date(year, m_start, 1)
    q_end = date(year, m_end, monthrange(year, m_end)[1])
    return q_start <= horizon and q_end >= today


def _pct(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v * 100:+.1f}%"


def _ratio(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v:.2f}×"
