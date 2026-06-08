"""Portfolio review — thesis health + events across your TastyTrade holdings.

Reachable from your phone: launch with `advisor dashboard ui --host 0.0.0.0`
and open the printed Network URL on the same WiFi.
"""

from __future__ import annotations

import streamlit as st
from advisor.dashboard import state
from advisor.dashboard.components.portfolio import render_portfolio
from advisor.dashboard.theme import apply_page_config
from advisor.research.portfolio_review import (
    DEFAULT_ACCOUNTS,
    build_portfolio_review,
    load_latest_review,
    render_portfolio_review,
    save_review,
)

apply_page_config("Portfolio")
state.init_state()

st.title("Portfolio review")
st.caption(
    "For every company you hold across your TastyTrade accounts: has the thesis "
    "changed, and are there events coming up? Pulls live holdings, checks KPI "
    "thesis health, and surfaces earnings + catalysts."
)

# ── Controls ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.subheader("Refresh options")
    selected = st.multiselect(
        "Accounts",
        options=list(DEFAULT_ACCOUNTS),
        default=list(DEFAULT_ACCOUNTS),
    )
    rebuild = st.checkbox(
        "Auto-build uncovered tickers",
        value=True,
        help="Run full research for holdings with no report yet (slow — minutes each).",
    )
    with_catalysts = st.checkbox("Fetch earnings/catalysts", value=True)
    concurrency = st.slider("Parallelism", 1, 6, 3)

col_a, col_b = st.columns([1, 3])
do_refresh = col_a.button("🔄 Refresh from TastyTrade", type="primary", use_container_width=True)

if do_refresh:
    if rebuild:
        st.warning(
            "Auto-building uncovered tickers can take several minutes and uses "
            "LLM/Tavily credits. Sit tight…"
        )
    with st.spinner("Pulling holdings and reviewing each company…"):
        try:
            review = build_portfolio_review(
                selected or None,
                rebuild_uncovered=rebuild,
                with_catalysts=with_catalysts,
                concurrency=concurrency,
            )
            save_review(review)
            st.session_state["_portfolio_review"] = review
        except Exception as exc:  # noqa: BLE001
            st.error(f"Refresh failed: {exc}")

# ── Load the review (session → cached) ─────────────────────────────────────────

review = st.session_state.get("_portfolio_review")
fetched_at = None
if review is None:
    loaded = load_latest_review()
    if loaded is not None:
        review, fetched_at = loaded
        st.session_state["_portfolio_review"] = review

if review is None:
    st.info("No portfolio review yet. Hit **Refresh from TastyTrade** to build one.")
    st.stop()

stamp = (fetched_at or review.generated_at).strftime("%Y-%m-%d %H:%M")
st.caption(f"Last reviewed: {stamp} · accounts {', '.join(review.account_numbers) or '—'}")

# ── Downloads ──────────────────────────────────────────────────────────────────

d1, d2, _ = st.columns([1, 1, 3])
d1.download_button(
    "⬇ Markdown",
    data=render_portfolio_review(review),
    file_name="portfolio_review.md",
    mime="text/markdown",
    use_container_width=True,
)
d2.download_button(
    "⬇ JSON",
    data=review.model_dump_json(indent=2),
    file_name="portfolio_review.json",
    mime="application/json",
    use_container_width=True,
)

st.divider()

render_portfolio(review)
