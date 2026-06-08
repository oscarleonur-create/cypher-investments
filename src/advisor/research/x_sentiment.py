"""X.com social sentiment via the X API v2.

Searches recent tweets (last 7 days) for the ticker using Bearer Token
app-only auth. Falls back to a neutral default when the token is absent
or the API returns no results.

Required env var:
    ADVISOR_RESEARCH_X_BEARER_TOKEN  — X Developer App Bearer Token
"""

from __future__ import annotations

import logging
import os
from typing import Any

import httpx
from pydantic import BaseModel, Field

from advisor.research.models import XPost, XSentimentResult

logger = logging.getLogger(__name__)

_X_SEARCH_URL = "https://api.twitter.com/2/tweets/search/recent"
_X_TWEET_FIELDS = "text,created_at,public_metrics,author_id"
_MAX_RESULTS = 50  # 10–100 per request on Basic tier; free tier allows 10


# ── Structured LLM response ──────────────────────────────────────────────────


class _XSentimentOut(BaseModel):
    score: float = Field(
        description="Composite sentiment score 0-100 (50=neutral, >70=bullish, <30=bearish)"
    )
    positive_pct: float = Field(
        description="Percentage of tweets with bullish/positive sentiment 0-100"
    )
    top_bullish_themes: list[str] = Field(
        default_factory=list,
        description="Top 2-4 bullish narrative themes recurring across tweets",
    )
    top_bearish_themes: list[str] = Field(
        default_factory=list,
        description="Top 2-4 bearish narrative themes recurring across tweets",
    )
    key_posts: list[dict[str, Any]] = Field(
        default_factory=list,
        description=(
            "Up to 6 representative tweets. Each dict must have: "
            "text (str), sentiment ('bullish'|'bearish'|'neutral'), themes (list[str]), url (str)"
        ),
    )


_X_SENTIMENT_PROMPT = """\
You are a financial sentiment analyst specializing in retail investor signals on X (Twitter).
Given a set of recent tweets about a stock ticker, extract structured sentiment.

Tweets are labeled [t1], [t2], … for reference.

Provide:
- score: 0-100 composite sentiment (50=neutral, >70=bullish, <30=bearish)
- positive_pct: percentage of tweets that are bullish/positive (0-100)
- top_bullish_themes: 2-4 recurring bullish narrative themes
  (e.g. "strong earnings beat", "AI growth optionality")
- top_bearish_themes: 2-4 recurring bearish narrative themes
  (e.g. "guidance cut", "margin pressure")
- key_posts: up to 6 representative tweets; each must have:
    - text: verbatim tweet text
    - sentiment: "bullish" | "bearish" | "neutral"
    - themes: list of 1-3 theme labels
    - url: leave empty string "" (URLs are not included in tweet text)

Rules:
- Weight tweets with high engagement (likes, retweets) more heavily
- Ignore spam, bots, and off-topic posts
- Never fabricate tweets not present in the input
- Financial analysis tweets carry more signal than emotional reactions"""


# ── X API v2 client ──────────────────────────────────────────────────────────


def _fetch_tweets(symbol: str, bearer_token: str) -> list[dict]:
    """Call X API v2 search/recent and return raw tweet dicts."""
    query = f"${symbol} lang:en -is:retweet"
    params = {
        "query": query,
        "max_results": min(_MAX_RESULTS, 100),
        "tweet.fields": _X_TWEET_FIELDS,
    }
    headers = {"Authorization": f"Bearer {bearer_token}"}

    resp = httpx.get(
        _X_SEARCH_URL,
        params=params,
        headers=headers,
        timeout=30,
    )
    resp.raise_for_status()
    data = resp.json()
    return data.get("data") or []


# ── Main function ─────────────────────────────────────────────────────────────


def build_x_sentiment(symbol: str, company_name: str = "") -> XSentimentResult:
    """Fetch recent X tweets for `symbol` and score sentiment via Claude.

    Returns XSentimentResult with neutral defaults when the Bearer Token is
    absent or the API returns no results.
    """
    bearer_token = os.environ.get("ADVISOR_RESEARCH_X_BEARER_TOKEN", "").strip()
    if not bearer_token:
        logger.info("ADVISOR_RESEARCH_X_BEARER_TOKEN not set — skipping X sentiment")
        return XSentimentResult(symbol=symbol)

    from research_agent.config import ResearchConfig
    from research_agent.llm import OpenRouterLLM

    config = ResearchConfig()
    llm = OpenRouterLLM(config)

    try:
        tweets = _fetch_tweets(symbol, bearer_token)

        if not tweets:
            logger.info("No recent tweets found for $%s", symbol)
            return XSentimentResult(symbol=symbol)

        # Build a labeled block for the LLM
        parts: list[str] = []
        for i, t in enumerate(tweets, 1):
            metrics = t.get("public_metrics") or {}
            engagement = (
                f"likes={metrics.get('like_count', 0)} " f"RT={metrics.get('retweet_count', 0)}"
            )
            parts.append(f"[t{i}] ({engagement})\n{t.get('text', '')}")
        tweet_block = "\n\n---\n\n".join(parts)

        user_prompt = (
            f"Analyze sentiment of these recent X tweets about ${symbol} "
            f"({company_name or symbol}):\n\n{tweet_block}"
        )

        scored: _XSentimentOut = llm.complete(
            system_prompt=_X_SENTIMENT_PROMPT,
            user_prompt=user_prompt,
            response_model=_XSentimentOut,
        )

        key_posts = [
            XPost(
                text=p.get("text", ""),
                sentiment=p.get("sentiment", "neutral"),
                themes=p.get("themes", []),
                url=p.get("url", ""),
            )
            for p in (scored.key_posts or [])[:6]
        ]

        return XSentimentResult(
            symbol=symbol,
            score=max(0.0, min(100.0, scored.score)),
            positive_pct=max(0.0, min(100.0, scored.positive_pct)),
            top_bullish_themes=scored.top_bullish_themes[:4],
            top_bearish_themes=scored.top_bearish_themes[:4],
            key_posts=key_posts,
            post_count=len(tweets),
        )

    except httpx.HTTPStatusError as exc:
        logger.warning(
            "X API error for %s: %s %s", symbol, exc.response.status_code, exc.response.text[:200]
        )
        return XSentimentResult(symbol=symbol)
    except Exception as exc:
        logger.warning("X sentiment failed for %s: %s", symbol, exc)
        return XSentimentResult(symbol=symbol)
