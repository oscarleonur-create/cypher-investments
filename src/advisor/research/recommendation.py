"""Symbol-scoped position recommendation: BUY/SELL/INCREASE/DECREASE/HOLD.

Decision support only -- this module never places an order. Position facts
(quantity, cost basis, mark, P&L%) are computed here in Python from live
account data and handed to the LLM as given facts; the LLM only writes the
action/reasoning/factors, following the same anti-hallucination pattern
established in investment_memo.py/thesis.py this session: inject today's
date, "use ONLY the given facts", qualitative language over invented
precision, LOW conviction when context is sparse.
"""

from __future__ import annotations

import logging
from datetime import date

from advisor.research.models import ActionRecommendation, PositionContext, ResearchReport

logger = logging.getLogger(__name__)


async def fetch_position_context(symbol: str) -> PositionContext:
    """Live equity + option positions for `symbol`, merged across accounts.

    Mirrors advisor.api.routers.portfolio.holdings()'s account-looping and
    weighted-average-cost math (deliberately duplicated rather than
    refactoring that working endpoint) -- extended to also surface option
    legs, since this app trades options as its primary strategy and the
    /holdings endpoint deliberately excludes them for the equity grid.
    """
    from advisor.api import deps
    from advisor.market.tastytrade_client import get_positions
    from advisor.research.portfolio_review import DEFAULT_ACCOUNTS

    sym = symbol.upper()
    session = await deps.try_get_tt_session()
    if session is None:
        return PositionContext(error="No TastyTrade session available.")

    equity_qty = 0.0
    equity_avg = 0.0
    mark = 0.0
    option_legs: list[str] = []
    accounts: list[str] = []

    for acct in DEFAULT_ACCOUNTS:
        try:
            positions = await get_positions(session, acct)
        except Exception as exc:  # noqa: BLE001
            logger.warning("recommendation: positions fetch failed for %s: %s", acct, exc)
            continue

        for p in positions:
            if str(p.get("underlying_symbol", "")).upper() != sym:
                continue
            if acct not in accounts:
                accounts.append(acct)

            instrument = str(p.get("instrument_type", "")).upper()
            direction = -1 if str(p.get("quantity_direction", "")).lower() == "short" else 1
            qty = float(p.get("quantity", 0)) * direction

            if "OPTION" in instrument:
                side = "SELL" if direction < 0 else "BUY"
                option_legs.append(
                    f"{side} {abs(qty):g}x {p.get('symbol', sym)} "
                    f"@ {float(p.get('average_open_price', 0) or 0):.2f}"
                )
                continue

            new_qty = equity_qty + qty
            if new_qty != 0:
                equity_avg = (
                    equity_avg * equity_qty + float(p.get("average_open_price", 0) or 0) * qty
                ) / new_qty
            equity_qty = new_qty
            mark = float(p.get("mark_price", 0) or p.get("mark", 0) or mark)

    pnl_pct = None
    if equity_qty != 0 and equity_avg > 0 and mark > 0:
        pnl_pct = (mark - equity_avg) / equity_avg * (1 if equity_qty > 0 else -1)

    return PositionContext(
        has_position=bool(equity_qty != 0 or option_legs),
        equity_quantity=equity_qty,
        average_open_price=equity_avg,
        mark=mark,
        unrealized_pnl_pct=pnl_pct,
        option_legs=option_legs,
        accounts=accounts,
    )


def build_recommendation(
    symbol: str,
    report: ResearchReport | None,
    position: PositionContext,
) -> ActionRecommendation:
    llm_out = _llm_recommend(symbol, report, position)
    return ActionRecommendation(
        symbol=symbol.upper(),
        action=llm_out.get("action", "HOLD"),
        conviction=llm_out.get("conviction", "LOW"),
        reasoning=llm_out.get("reasoning", _fallback_reasoning(report, position)),
        key_factors=llm_out.get("key_factors", []),
        risks=llm_out.get("risks", []),
        position=position,
    )


def _llm_recommend(symbol: str, report: ResearchReport | None, position: PositionContext) -> dict:
    try:
        from pydantic import BaseModel as PydanticBase
        from research_agent.config import ResearchConfig
        from research_agent.llm import OpenRouterLLM

        config = ResearchConfig()
        if not config.openrouter_api_key:
            return {}

        context = _build_context(symbol, report, position)
        today = date.today().isoformat()
        valid_actions = (
            "HOLD, INCREASE, DECREASE, or SELL (never BUY -- a position is already open)"
            if position.has_position
            else "BUY or HOLD (never SELL/INCREASE/DECREASE -- there is no open position)"
        )

        class RecommendationOut(PydanticBase):
            action: str
            conviction: str
            reasoning: str
            key_factors: list[str]
            risks: list[str]

        llm = OpenRouterLLM(config)
        result = llm.complete(
            system_prompt=(
                f"Today's date is {today}. You are providing data-driven decision "
                "support for the user's own trading system, not personalized "
                "financial advice -- frame reasoning as 'the data shows X' rather "
                "than directives. "
                "Use ONLY the facts given in the context below. Do NOT invent "
                "numbers, catalysts, price targets, or other facts not explicitly "
                "stated or directly computable from it -- a fabricated-but-"
                "plausible-sounding figure is worse than an honest 'insufficient "
                "data', since this drives a real position decision. "
                f"Given the current position state, valid actions are: {valid_actions}. "
                "conviction: HIGH | MEDIUM | LOW -- use LOW whenever the context is "
                "sparse (e.g. no cached research report, or few data points). "
                "reasoning: 2-4 sentences, grounded only in the given context. "
                "key_factors: 2-4 short strings, each citing a specific given fact. "
                "risks: 1-3 short strings on what would make this call wrong."
            ),
            user_prompt=context,
            response_model=RecommendationOut,
        )
        return result.model_dump()

    except Exception as exc:  # noqa: BLE001
        logger.warning("Recommendation LLM call failed for %s: %s", symbol, exc)
        return {}


def _build_context(symbol: str, report: ResearchReport | None, position: PositionContext) -> str:
    parts: list[str] = [f"Symbol: {symbol.upper()}"]

    if position.has_position:
        pos_desc = []
        if position.equity_quantity:
            pos_desc.append(
                f"{position.equity_quantity:g} shares @ avg ${position.average_open_price:.2f}, "
                f"mark ${position.mark:.2f}"
            )
            if position.unrealized_pnl_pct is not None:
                pos_desc.append(f"unrealized {position.unrealized_pnl_pct:+.1%}")
        if position.option_legs:
            pos_desc.append("option legs: " + "; ".join(position.option_legs))
        parts.append("Current position: " + ", ".join(pos_desc))
    else:
        parts.append("Current position: none.")

    if report is None:
        parts.append("No cached research report is available for this symbol.")
        return "\n".join(parts)

    if report.business_model:
        parts.append(f"Business: {report.business_model[:300]}")

    vp = report.variant_perception
    if vp and vp.our_key_insight:
        parts.append(f"Our edge: {vp.our_key_insight}")

    dcf = report.dcf
    if dcf:
        parts.append(f"DCF: current ${dcf.current_price:.2f}, WACC {dcf.wacc:.1%}")
        if dcf.base:
            parts.append(f"Base case: ${dcf.base.implied_price:.2f} ({dcf.base.upside_pct:+.1%})")
        if dcf.bear:
            parts.append(f"Bear case: ${dcf.bear.implied_price:.2f} ({dcf.bear.upside_pct:+.1%})")

    if report.thesis and report.thesis.summary:
        parts.append(f"Thesis summary: {report.thesis.summary[:300]}")

    if report.catalyst_risk and report.catalyst_risk.catalysts:
        near_term = [c for c in report.catalyst_risk.catalysts if c.is_near_term]
        if near_term:
            parts.append(
                "Near-term catalysts: "
                + "; ".join(f"{c.description} ({c.expected_date})" for c in near_term[:3])
            )

    if report.catalyst_risk and report.catalyst_risk.high_risks:
        parts.append(
            "Key risks: "
            + "; ".join(r.description[:80] for r in report.catalyst_risk.high_risks[:3])
        )

    return "\n".join(parts)


def _fallback_reasoning(report: ResearchReport | None, position: PositionContext) -> str:
    if report is None:
        return "No cached research report available -- build research for this symbol first."
    if position.has_position:
        return "Existing position found; LLM synthesis unavailable, no automated recommendation."
    return "No existing position; LLM synthesis unavailable, no automated recommendation."
