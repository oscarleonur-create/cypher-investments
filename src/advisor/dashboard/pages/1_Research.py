"""Research workstation — single-ticker deep fundamental analysis."""

from __future__ import annotations

import streamlit as st
from advisor.dashboard import data, state
from advisor.dashboard.components.activity import render_activity
from advisor.dashboard.components.catalysts import render_catalysts
from advisor.dashboard.components.ecosystem import render_ecosystem
from advisor.dashboard.components.edge import render_edge
from advisor.dashboard.components.financials import render_financials
from advisor.dashboard.components.header import render_header
from advisor.dashboard.components.notes import render_notes
from advisor.dashboard.components.overview import render_overview
from advisor.dashboard.components.ratios import render_ratios
from advisor.dashboard.components.valuation import render_valuation
from advisor.dashboard.persistence import get_store
from advisor.dashboard.theme import apply_page_config

apply_page_config("Research")
state.init_state()
session_id = state.get_session_id()

st.title("Fundamental Research")
st.caption("Run the full research pipeline, browse the results, follow connections.")


# ── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    # ── Session picker ───────────────────────────────────────────────────
    st.subheader("Session")
    _store = get_store()
    try:
        _sessions = _store.list_sessions(limit=20)
    finally:
        _store.close()
    _options = {s["id"]: s for s in _sessions}
    _labels = [f"{s['name']} · {s.get('current_symbol') or '—'}" for s in _sessions]
    if _sessions:
        _ids = [s["id"] for s in _sessions]
        try:
            _idx = _ids.index(session_id)
        except ValueError:
            _idx = 0
        _picked = st.selectbox(
            "Active session",
            options=range(len(_ids)),
            index=_idx,
            format_func=lambda i: _labels[i],
            label_visibility="collapsed",
        )
        if _ids[_picked] != session_id:
            state.set_session_id(_ids[_picked])
            st.rerun()
    if st.button("➕ New session", use_container_width=True):
        _store = get_store()
        try:
            _new_id = _store.new_session()
        finally:
            _store.close()
        state.set_session_id(_new_id)
        st.rerun()

    st.markdown("---")
    st.subheader("Ticker")
    # The text input is keyed on `symbol_input` so we can override it
    # programmatically when the user clicks Back / Follow / Watchlist.
    pending = state.consume_pending_symbol()
    if pending is not None:
        st.session_state["symbol_input"] = pending
    elif "symbol_input" not in st.session_state:
        st.session_state["symbol_input"] = state.get_current_symbol() or "AAPL"

    symbol_input = st.text_input("Symbol", key="symbol_input").strip().upper()

    st.markdown("---")
    st.subheader("Parameters")
    n_years = st.slider("Years of history", 1, 10, value=5)
    peers_input = st.text_input(
        "Custom peers (comma-separated)",
        value="",
        help="Override the auto-built peer set used in valuation.",
    )
    force_refresh = st.checkbox(
        "Force refresh (ignore cache)",
        value=False,
        help="Rebuild every layer from scratch. The first run on a ticker takes 1–3 minutes.",
    )

    st.markdown("---")
    run_clicked = st.button("Run / Switch", type="primary", use_container_width=True)

    st.markdown("---")
    st.subheader("History")
    hist = st.session_state.get("history") or []
    if len(hist) >= 2:
        back_clicked = st.button("← Back", use_container_width=True)
        if back_clicked:
            prev = state.pop_history()
            if prev:
                state.enter_research(prev)
                st.rerun()
    if hist:
        for i, prior in enumerate(reversed(hist[-10:])):
            if st.button(prior, key=f"hist_{i}_{prior}", use_container_width=True):
                state.enter_research(prior)
                st.rerun()
    else:
        st.caption("_No prior tickers this session._")

    # ── Per-layer refresh ────────────────────────────────────────────────
    report_current = state.get_report()
    if report_current is not None:
        st.markdown("---")
        st.subheader("Refresh layers")
        st.caption("Rebuild one slice without re-running the full pipeline.")
        layer_cols = st.columns(2)
        peers_list = [p.strip().upper() for p in peers_input.split(",") if p.strip()] or None
        layer_buttons = [
            ("Statements", "statements"),
            ("Multiples", "multiples"),
            ("Ecosystem", "ecosystem"),
            ("Catalysts", "catalysts"),
            ("Transcripts", "transcripts"),
            ("Edge", "edge"),
        ]
        for i, (label, key) in enumerate(layer_buttons):
            col = layer_cols[i % 2]
            extra = "(custom peers)" if key == "multiples" and peers_list else ""
            if col.button(
                f"↻ {label} {extra}".strip(),
                key=f"refresh_{key}",
                use_container_width=True,
            ):
                with st.spinner(f"Refreshing {label.lower()}…"):
                    try:
                        updated = data.refresh_layer(
                            report_current,
                            layer=key,
                            n_years=n_years,
                            explicit_peers=peers_list if key == "multiples" else None,
                        )
                        state.set_report(updated)
                        state.log_event(
                            updated.symbol,
                            "refresh",
                            {"layer": key, "peers": peers_list},
                        )
                        st.success(f"{label} refreshed.")
                    except Exception as exc:  # noqa: BLE001
                        st.error(f"{label} refresh failed: {exc}")
                st.rerun()

    st.markdown("---")
    st.subheader("Watchlist")
    watch = st.session_state.get("watchlist") or []
    if watch:
        for sym in watch:
            cols = st.columns([4, 1])
            if cols[0].button(sym, key=f"watch_open_{sym}", use_container_width=True):
                state.enter_research(sym)
                st.rerun()
            if cols[1].button("✕", key=f"watch_rm_{sym}", help=f"Remove {sym}"):
                state.remove_from_watchlist(sym)
                st.rerun()
    else:
        st.caption("_Pin tickers from the header to build a watchlist._")

    st.markdown("---")
    st.caption(
        "**Tip**: set `RESEARCH_AGENT_OPENROUTER_API_KEY` to unlock the "
        "AI thesis generator on the Overview tab."
    )


# ── Run handler ─────────────────────────────────────────────────────────────


def _on_run(symbol: str, n_years: int, peers: list[str] | None, force: bool) -> None:
    label = f"Building research for **{symbol}**"
    if force:
        label += " (force refresh)"
    with st.status(label, expanded=True) as status:

        def progress(event: str, layer: str) -> None:
            icon = {"started": "•", "done": "✓", "failed": "✗"}.get(event, "•")
            status.update(label=f"{label} — {layer}")
            st.write(f"{icon} {layer}")

        try:
            report = data.load_report(
                symbol=symbol,
                n_years=n_years,
                explicit_peers=peers,
                force_refresh=force,
                progress=progress,
            )
            if data.is_report_empty(report):
                # Every layer succeeded but produced no data — typically an
                # invalid ticker. Purge the cache so it doesn't pollute the
                # recent list and surface a clear error.
                data.purge_cached_report(symbol)
                status.update(
                    label=f"No data found for {symbol} — likely an invalid ticker.",
                    state="error",
                )
                st.error(
                    f"**{symbol}** doesn't appear to be a valid ticker — yfinance, "
                    f"EDGAR, peers, and holders all returned empty. "
                    f"Common typos: `APPL` → `AAPL`, `GOOG` vs `GOOGL`, `BRK.B` → `BRK-B`."
                )
                # Pop the failed symbol off history so Back doesn't loop us here.
                hist = st.session_state.get("history") or []
                if hist and hist[-1] == symbol:
                    hist.pop()
                return
            state.set_report(report)
            state.set_current_symbol(symbol)
            state.push_history(symbol)
            state.log_event(symbol, "view", {"force": force, "n_years": n_years})
            # Persist current symbol on the session row so reopening tomorrow
            # lands back here.
            try:
                _s = get_store()
                _s.open_session(session_id, current_symbol=symbol)
                _s.close()
            except Exception:  # noqa: BLE001
                pass
            status.update(label=f"Done — {symbol}", state="complete")
        except Exception as exc:  # noqa: BLE001
            status.update(label=f"Failed — {exc}", state="error")
            st.exception(exc)


# Auto-run when the user clicks Run/Switch, OR when the active symbol differs
# from what's currently displayed (e.g. a follow-button just fired).
report = state.get_report()
auto_switch = (
    symbol_input and (report is None or report.symbol != symbol_input) and len(symbol_input) <= 8
)

if run_clicked and symbol_input:
    peers_list = [p.strip().upper() for p in peers_input.split(",") if p.strip()] or None
    _on_run(symbol_input, n_years, peers_list, force_refresh)
    report = state.get_report()
elif auto_switch and symbol_input != (report.symbol if report else None):
    # Triggered by Follow / Back / Watchlist clicks.
    _on_run(symbol_input, n_years, None, False)
    report = state.get_report()


# ── Main canvas ─────────────────────────────────────────────────────────────

if report is None:
    st.info(
        "Enter a ticker in the sidebar and click **Run / Switch** to build a report. "
        "Cached reports load instantly; first-time tickers run the full pipeline (1–3 min)."
    )
else:
    # Header bar with a pin-to-watchlist action
    header_cols = st.columns([5, 1])
    with header_cols[0]:
        render_header(report, last_run_at=st.session_state.get("last_run_at"))
    with header_cols[1]:
        already_pinned = report.symbol in (st.session_state.get("watchlist") or [])
        if not already_pinned:
            if st.button("📌 Pin to watchlist", use_container_width=True):
                state.add_to_watchlist(report.symbol)
                st.rerun()
        else:
            st.caption("📌 In watchlist")
    st.divider()

    tabs = st.tabs(
        [
            "Overview",
            "Edge",
            "Financials",
            "Ratios",
            "Valuation",
            "Ecosystem",
            "Catalysts & Risks",
            "Notes & Sources",
            "Activity",
        ]
    )

    with tabs[0]:
        render_overview(report, session_id=session_id)
    with tabs[1]:
        render_edge(report)
    with tabs[2]:
        render_financials(report)
    with tabs[3]:
        render_ratios(report)
    with tabs[4]:
        render_valuation(report)
    with tabs[5]:
        render_ecosystem(report)
    with tabs[6]:
        render_catalysts(report)
    with tabs[7]:
        render_notes(report, session_id)
    with tabs[8]:
        render_activity(session_id, report.symbol)
