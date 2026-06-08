"""Session-state helpers for the research workstation.

Step 1 keeps state in-memory only (per Streamlit session). Persistence to
SQLite arrives in step 5 (sessions, history, notes, scenarios).
"""

from __future__ import annotations

import streamlit as st


def init_state() -> None:
    """Initialize the session-state keys used by the Research page."""
    st.session_state.setdefault("current_symbol", "")
    st.session_state.setdefault("report", None)
    st.session_state.setdefault("history", [])
    st.session_state.setdefault("watchlist", [])
    st.session_state.setdefault("last_run_at", None)
    st.session_state.setdefault("session_id", None)


def get_session_id() -> str:
    """Return the active research session, auto-creating one on first call.

    Sessions persist to SQLite and let the user pick up tomorrow where they
    left off (notes, scenarios, history of visited tickers).
    """
    from advisor.dashboard.persistence import get_store

    sid = st.session_state.get("session_id")
    if sid:
        return sid

    store = get_store()
    try:
        sessions = store.list_sessions(limit=1)
        if sessions:
            sid = sessions[0]["id"]
            store.open_session(sid)
        else:
            sid = store.new_session()
    finally:
        store.close()

    st.session_state["session_id"] = sid
    return sid


def set_session_id(session_id: str) -> None:
    """Switch the active research session and clear UI state for the new one."""
    from advisor.dashboard.persistence import get_store

    st.session_state["session_id"] = session_id
    st.session_state["history"] = []
    st.session_state["report"] = None
    st.session_state["current_symbol"] = ""

    store = get_store()
    try:
        opened = store.open_session(session_id)
        if opened and opened.get("current_symbol"):
            sym = opened["current_symbol"]
            st.session_state["pending_symbol"] = sym
            st.session_state["current_symbol"] = sym
    finally:
        store.close()


def log_event(symbol: str, event_type: str, payload: dict | None = None) -> None:
    """Best-effort: log an event into the active session."""
    sid = st.session_state.get("session_id")
    if not sid or not symbol:
        return
    try:
        from advisor.dashboard.persistence import get_store

        store = get_store()
        try:
            store.log_event(sid, symbol, event_type, payload)
        finally:
            store.close()
    except Exception:  # noqa: BLE001
        pass


def set_current_symbol(symbol: str) -> None:
    st.session_state["current_symbol"] = symbol.strip().upper()


def get_current_symbol() -> str:
    return st.session_state.get("current_symbol", "")


def set_report(report) -> None:  # type: ignore[no-untyped-def]
    from datetime import datetime

    st.session_state["report"] = report
    st.session_state["last_run_at"] = datetime.now()


def get_report():  # type: ignore[no-untyped-def]
    return st.session_state.get("report")


def push_history(symbol: str) -> None:
    """Record a visit to `symbol`. Used by the back-button."""
    sym = symbol.strip().upper()
    if not sym:
        return
    hist = st.session_state.setdefault("history", [])
    if not hist or hist[-1] != sym:
        hist.append(sym)


def pop_history() -> str | None:
    """Drop the current symbol from the history stack and return the previous one."""
    hist = st.session_state.get("history") or []
    if len(hist) < 2:
        return None
    hist.pop()  # drop current
    return hist[-1]


def enter_research(new_symbol: str) -> None:
    """Switch the active research subject.

    Pushes the new symbol onto history and clears the cached report so the
    page's run handler picks up the change on the next rerun. Caller is
    responsible for calling `st.rerun()` after this returns.
    """
    sym = new_symbol.strip().upper()
    if not sym:
        return
    st.session_state["pending_symbol"] = sym
    push_history(sym)
    # Drop the displayed report; the page will detect the mismatch with
    # `pending_symbol` and rebuild on the next render cycle.
    st.session_state["report"] = None


def consume_pending_symbol() -> str | None:
    """Read-and-clear the pending symbol set by `enter_research`."""
    sym = st.session_state.pop("pending_symbol", None)
    return sym


def add_to_watchlist(symbol: str) -> None:
    sym = symbol.strip().upper()
    if not sym:
        return
    watch = st.session_state.setdefault("watchlist", [])
    if sym not in watch:
        watch.append(sym)


def remove_from_watchlist(symbol: str) -> None:
    sym = symbol.strip().upper()
    watch = st.session_state.get("watchlist") or []
    if sym in watch:
        watch.remove(sym)
