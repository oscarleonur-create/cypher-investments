"""X.com / Twitter social sentiment display component."""

from __future__ import annotations

import streamlit as st

from advisor.research.models import ResearchReport


def render_x_sentiment(report: ResearchReport) -> None:
    st.subheader("X.com / Twitter Sentiment")

    import os

    xs = report.x_sentiment
    if xs is None or xs.post_count == 0:
        if not os.environ.get("ADVISOR_RESEARCH_X_BEARER_TOKEN"):
            st.info(
                "**X sentiment not configured.** "
                "Set `ADVISOR_RESEARCH_X_BEARER_TOKEN` in your environment, "
                "then re-run the pipeline.  \n"
                "Get a Bearer Token at [developer.x.com](https://developer.x.com) "
                "(free tier works)."
            )
        else:
            st.info(
                f"No recent tweets found for **${report.symbol}** "
                "(last 7 days). Try again after the next earnings event or news catalyst."
            )
        return

    # ── Score overview ────────────────────────────────────────────────────────
    score_label = "Bullish" if xs.score >= 65 else "Bearish" if xs.score <= 35 else "Neutral"
    delta_str = f"{xs.score - 50:+.0f} vs neutral"

    col1, col2, col3 = st.columns(3)
    col1.metric("Sentiment Score", f"{xs.score:.0f} / 100", delta=delta_str)
    col2.metric("Bullish Posts", f"{xs.positive_pct:.0f}%")
    col3.metric("Posts Analyzed", str(xs.post_count))

    st.caption(f"Overall signal: **{score_label}**")
    st.divider()

    # ── Themes ────────────────────────────────────────────────────────────────
    theme_col1, theme_col2 = st.columns(2)

    with theme_col1:
        st.markdown("**Bullish themes**")
        if xs.top_bullish_themes:
            for theme in xs.top_bullish_themes:
                st.markdown(f"- {theme}")
        else:
            st.caption("None identified")

    with theme_col2:
        st.markdown("**Bearish themes**")
        if xs.top_bearish_themes:
            for theme in xs.top_bearish_themes:
                st.markdown(f"- {theme}")
        else:
            st.caption("None identified")

    # ── Key posts ─────────────────────────────────────────────────────────────
    if xs.key_posts:
        st.divider()
        with st.expander(f"Key posts ({len(xs.key_posts)} shown)", expanded=False):
            _BADGE = {"bullish": "🟢", "bearish": "🔴", "neutral": "⚪"}
            for post in xs.key_posts:
                badge = _BADGE.get(post.sentiment, "⚪")
                themes_str = " · ".join(post.themes) if post.themes else ""
                st.markdown(
                    f"{badge} **{post.sentiment.capitalize()}**"
                    + (f" — _{themes_str}_" if themes_str else "")
                )
                st.markdown(f"> {post.text}")
                if post.url:
                    st.markdown(f"[View on X]({post.url})")
                st.markdown("---")
