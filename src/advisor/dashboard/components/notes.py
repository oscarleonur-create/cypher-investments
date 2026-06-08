"""Notes & Sources tab — markdown notes editor + saved scenarios + exports."""

from __future__ import annotations

import io

import pandas as pd
import streamlit as st

from advisor.dashboard.persistence import get_store
from advisor.research.models import ResearchReport


def render_notes(report: ResearchReport, session_id: str) -> None:
    _render_notes_editor(report, session_id)
    st.divider()
    _render_saved_scenarios(report.symbol)
    st.divider()
    _render_filings(report)
    st.divider()
    _render_transcripts(report)
    st.divider()
    _render_exports(report)


# ── Notes editor ────────────────────────────────────────────────────────────


def _render_notes_editor(report: ResearchReport, session_id: str) -> None:
    st.subheader(f"Notes — {report.symbol}")

    note_key = f"note_{session_id}_{report.symbol}"
    if note_key not in st.session_state:
        store = get_store()
        try:
            existing = store.load_note(session_id, report.symbol)
        finally:
            store.close()
        st.session_state[note_key] = existing["note_md"] if existing else ""

    note_md = st.text_area(
        "Markdown notes (autosaved when you click Save)",
        value=st.session_state[note_key],
        height=240,
        key=f"{note_key}_input",
        placeholder=(
            "# Thesis\n"
            "- Key bull point\n"
            "- Key bear point\n\n"
            "## To do\n"
            "- [ ] Check 10-K page 47 for revenue recognition policy\n"
        ),
    )

    col1, col2, col3 = st.columns([1, 1, 4])
    if col1.button("💾 Save note", type="primary"):
        store = get_store()
        try:
            store.save_note(session_id, report.symbol, note_md)
            store.log_event(session_id, report.symbol, "note_added", {"chars": len(note_md)})
        finally:
            store.close()
        st.session_state[note_key] = note_md
        st.success("Note saved.")
    if col2.button("Reload"):
        st.session_state.pop(note_key, None)
        st.rerun()

    # Preview
    if note_md.strip():
        with st.expander("Preview", expanded=False):
            st.markdown(note_md)


# ── Saved scenarios (mirror of Valuation tab list) ──────────────────────────


def _render_saved_scenarios(symbol: str) -> None:
    st.subheader("Saved scenarios")
    store = get_store()
    try:
        saved = store.list_scenarios(symbol)
    finally:
        store.close()
    if not saved:
        st.caption("No scenarios saved yet. Build one in the Valuation tab.")
        return
    rows = []
    for sc in saved:
        a = sc["assumptions"]
        r = sc["result"]
        rows.append(
            {
                "Name": sc["name"],
                "Saved": sc["saved_at"][:16],
                "WACC": f"{a.wacc:.2%}",
                "Rev grw 1-3": f"{a.revenue_growth_yr1_3:.1%}",
                "FCF margin": f"{a.target_fcf_margin:.1%}",
                "Implied $": f"${r.implied_price:,.2f}",
                "Upside": f"{r.upside_pct:+.1%}",
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


# ── Filings ─────────────────────────────────────────────────────────────────


def _render_filings(report: ResearchReport) -> None:
    st.subheader("Filings")
    if not report.filings:
        st.caption(
            "No filings on record. Use **↻ Filings** in the sidebar (or Force "
            "refresh) to pull the SEC EDGAR filing index for this ticker."
        )
        return
    rows = [
        {
            "Form": f.form.value,
            "Filed": str(f.filing_date),
            "Period": str(f.period_of_report) if f.period_of_report else "—",
            "Accession": f.accession_number,
            "URL": f.url or "—",
        }
        for f in report.filings
    ]
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Full-text reader ─────────────────────────────────────────────────
    st.markdown("**Read a filing in full**")
    labels = {f"{f.form.value} · {f.filing_date} · {f.accession_number}": f for f in report.filings}
    picked = st.selectbox(
        "Choose a filing to open",
        options=list(labels.keys()),
        key=f"filing_pick_{report.symbol}",
        label_visibility="collapsed",
    )
    sel = labels[picked]
    col1, col2 = st.columns([1, 4])
    load = col1.button("📄 Load full text", key=f"filing_load_{report.symbol}")
    if sel.url:
        col2.markdown(f"[Open on SEC.gov ↗]({sel.url})")

    state_key = f"filing_text_{sel.accession_number}"
    if load:
        from advisor.dashboard import data

        st.session_state[state_key] = data.load_filing_text(sel.accession_number, as_markdown=True)

    text = st.session_state.get(state_key)
    if text:
        st.download_button(
            "Download filing (Markdown)",
            data=text.encode(),
            file_name=f"{report.symbol}_{sel.form.value.replace(' ', '')}_{sel.filing_date}.md",
            mime="text/markdown",
            key=f"filing_dl_{sel.accession_number}",
        )
        st.caption(f"{len(text):,} characters")
        with st.container(height=600, border=True):
            st.markdown(text)
    elif text == "":
        st.warning(
            "Couldn't retrieve the filing text from EDGAR. "
            "Use the SEC.gov link above to read it on the source site."
        )


# ── Transcripts ─────────────────────────────────────────────────────────────


def _render_transcripts(report: ResearchReport) -> None:
    st.subheader("Earnings transcripts")
    if report.transcripts is None or not report.transcripts.summaries:
        st.caption("No transcript summaries on record.")
        return
    for ts in report.transcripts.summaries:
        with st.expander(f"{ts.quarter} · tone: {ts.tone.value}", expanded=False):
            if ts.management_guidance:
                st.markdown(f"**Management guidance:** {ts.management_guidance}")
            if ts.analyst_concerns:
                st.markdown(f"**Analyst concerns:** {ts.analyst_concerns}")
            if ts.key_topics:
                st.markdown("**Key topics:** " + " · ".join(ts.key_topics))
            if ts.highlight_quote:
                st.markdown(f"> {ts.highlight_quote}")


# ── Exports ─────────────────────────────────────────────────────────────────


def _render_exports(report: ResearchReport) -> None:
    st.subheader("Export")
    col1, col2, col3 = st.columns(3)

    # JSON
    json_bytes = report.model_dump_json(indent=2).encode()
    col1.download_button(
        "Download JSON",
        data=json_bytes,
        file_name=f"{report.symbol}_research.json",
        mime="application/json",
        use_container_width=True,
    )

    # Markdown — use research.render module if available
    try:
        from advisor.research.render import render_report

        md = render_report(report)
        if hasattr(md, "markup"):
            md_text = md.markup
        else:
            md_text = str(md)
    except Exception as exc:  # noqa: BLE001
        md_text = f"# {report.symbol} — render failed\n\n{exc}"
    col2.download_button(
        "Download Markdown",
        data=md_text.encode(),
        file_name=f"{report.symbol}_research.md",
        mime="text/markdown",
        use_container_width=True,
    )

    # CSV — flatten statements
    if report.statements and report.statements.income:
        csv_buf = io.StringIO()
        rows = []
        for p in report.statements.income:
            rows.append(
                {
                    "fiscal_year": p.fiscal_year,
                    "period_end": p.period_end,
                    "revenue": p.revenue,
                    "operating_income": p.operating_income,
                    "net_income": p.net_income,
                    "eps_diluted": p.eps_diluted,
                    "ebitda": p.ebitda,
                }
            )
        pd.DataFrame(rows).to_csv(csv_buf, index=False)
        col3.download_button(
            "Download statements CSV",
            data=csv_buf.getvalue().encode(),
            file_name=f"{report.symbol}_statements.csv",
            mime="text/csv",
            use_container_width=True,
        )
    else:
        col3.caption("No statements to export.")
