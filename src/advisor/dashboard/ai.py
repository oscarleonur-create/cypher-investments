"""AI thesis generator — synthesizes a balanced narrative from a ResearchReport.

Uses `research_agent.OpenRouterLLM` (already in the project) so the dashboard
doesn't need its own OpenAI client. The thesis is grounded in the data the
pipeline already produced — no extra web search, no fact-checking step. If
the user wants a deeper LLM workflow, they should run `research_agent` from
the CLI directly.

Required env var: `RESEARCH_AGENT_OPENROUTER_API_KEY` (loaded via
ResearchConfig). When absent, `is_configured()` returns False and the UI
shows setup instructions instead of trying to call the API.
"""

from __future__ import annotations

import logging
from datetime import date

from advisor.research.models import ResearchReport

logger = logging.getLogger(__name__)


SYSTEM_PROMPT = """\
You are a senior buy-side analyst writing an investment thesis card.

Synthesize a balanced thesis from the structured data provided. Be specific
and numeric — quote the actual figures. Do not invent data the prompt
doesn't contain; if a section is missing, say so plainly ("FCF data not
available") rather than guessing.

Output format (clean GitHub-flavored markdown, ~400-700 words):

## Verdict
**ONE-WORD** — BUY / WATCH / AVOID — followed by a single sentence rationale.

## Bull case (top 3)
- Bullet 1 (anchored to a specific number from the data)
- Bullet 2
- Bullet 3

## Bear case (top 3)
- Bullet 1 (anchored to a specific number)
- Bullet 2
- Bullet 3

## Key metrics
| Metric | Value |
|---|---|
| Revenue (latest) | … |
| FCF / Net income | … |
| ROIC | … |
| Net debt | … |
| Trailing P/E | … |
| DCF base upside | … |

## Risks
The two or three highest-conviction risks, each in one sentence.

## What would invalidate this thesis
One sentence on the specific data point or event that would flip the verdict.

## Next 30-90 days to watch
Bulleted list of near-term catalysts from the data, with dates."""


THESIS_CHAT_SYSTEM_PROMPT = """\
You are a senior buy-side analyst and investment thesis collaborator.

The user is building or stress-testing an investment thesis for the ticker shown
in the research data below. Your job is to help them think through the thesis
rigorously — answer their questions, challenge weak assumptions, surface overlooked
risks, and help draft thesis sections on demand.

Guidelines:
- Always ground your answers in the provided research data. Quote specific figures.
- If the data doesn't cover something, say so clearly rather than guessing.
- Be direct. One crisp paragraph beats three vague ones.
- When the user asks you to "draft" something (a bull case, an invalidation clause,
  a position sizing rationale), produce a concise, investor-ready block of text.
- Push back on overconfident framing. A good thesis acknowledges its own blind spots.

{context}"""


def is_configured() -> bool:
    """Return True if the OpenRouter API key is set in the environment."""
    try:
        from research_agent.config import ResearchConfig

        return bool(ResearchConfig().openrouter_api_key)
    except Exception:  # noqa: BLE001
        return False


def generate_thesis(report: ResearchReport) -> str:
    """Call OpenRouter with a balanced thesis prompt; return markdown."""
    from research_agent.config import ResearchConfig
    from research_agent.llm import OpenRouterLLM

    config = ResearchConfig()
    if not config.openrouter_api_key:
        raise RuntimeError(
            "RESEARCH_AGENT_OPENROUTER_API_KEY is not set. "
            "Add it to your environment or .env file."
        )

    llm = OpenRouterLLM(config)
    user_prompt = format_data_block(report)
    text = llm.complete(system_prompt=SYSTEM_PROMPT, user_prompt=user_prompt)
    # Strip stray markdown fences the model sometimes wraps in
    cleaned = text.strip()
    if cleaned.startswith("```"):
        lines = cleaned.split("\n")
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        cleaned = "\n".join(lines[1:])
    return cleaned


def build_chat_context(report: ResearchReport) -> str:
    """Build the system prompt for the thesis chat, injecting all available data."""
    data_block = format_data_block(report)
    lines = [data_block]

    if report.x_sentiment and report.x_sentiment.post_count > 0:
        xs = report.x_sentiment
        lines.append("## X.com / Twitter sentiment")
        lines.append(
            f"- Score: {xs.score:.0f}/100 · Positive: {xs.positive_pct:.0f}% · "
            f"Posts analyzed: {xs.post_count}"
        )
        if xs.top_bullish_themes:
            lines.append("- Bullish themes: " + ", ".join(xs.top_bullish_themes))
        if xs.top_bearish_themes:
            lines.append("- Bearish themes: " + ", ".join(xs.top_bearish_themes))
        lines.append("")

    context = "\n".join(lines)
    return THESIS_CHAT_SYSTEM_PROMPT.format(context=context)


# ── Data digest ─────────────────────────────────────────────────────────────


def format_data_block(report: ResearchReport) -> str:
    """Render a compact text summary of the report for the LLM prompt."""
    lines: list[str] = []
    lines.append(f"# {report.symbol} — research data as of {report.as_of}")
    if report.business_model:
        lines.append(f"Company: {report.business_model}")
    if report.industry:
        ind = report.industry
        bits = []
        if ind.sector:
            bits.append(f"sector={ind.sector}")
        if ind.industry:
            bits.append(f"industry={ind.industry}")
        if ind.moat_type and ind.moat_type.value != "none":
            bits.append(f"moat={ind.moat_type.value}")
        if ind.moat_strength is not None:
            tier = (
                "wide"
                if ind.moat_strength >= 7.5
                else "narrow"
                if ind.moat_strength >= 4
                else "none"
            )
            bits.append(f"moat_strength={ind.moat_strength:.1f}/10 ({tier})")
        if ind.competitive_position:
            bits.append(f"position={ind.competitive_position}")
        if bits:
            lines.append("Industry: " + ", ".join(bits))
        if ind.competitive_advantages:
            lines.append("Competitive advantages:")
            for adv in ind.competitive_advantages:
                lines.append(f"  - {adv}")
        if ind.sector_tailwinds:
            lines.append("Sector tailwinds: " + "; ".join(ind.sector_tailwinds))
        if ind.sector_headwinds:
            lines.append("Sector headwinds: " + "; ".join(ind.sector_headwinds))
    lines.append("")

    # Market snapshot (from multiples.subject — usually the freshest)
    if report.multiples and report.multiples.subject:
        s = report.multiples.subject
        lines.append("## Market snapshot")
        lines.append(
            f"- Price: {_money(s.price)} · Market cap: {_abbrev(s.market_cap)} "
            f"· EV: {_abbrev(s.enterprise_value)}"
        )
        lines.append(
            f"- P/E (TTM): {_x(s.pe_trailing)} · Fwd P/E: {_x(s.pe_forward)} "
            f"· EV/EBITDA: {_x(s.ev_to_ebitda)} · EV/Sales: {_x(s.ev_to_sales)} "
            f"· P/FCF: {_x(s.p_to_fcf)} · PEG: {_x(s.peg)} · Beta: {_x(s.beta)}"
        )
        if s.revenue_growth_yoy is not None:
            lines.append(f"- TTM Revenue growth YoY: {s.revenue_growth_yoy * 100:+.1f}%")
        lines.append("")

    # 5-year statements summary
    if report.statements and report.statements.income:
        lines.append("## 5-year financial trend (latest first)")
        inc = report.statements.income
        cf = report.statements.cashflow or []
        cf_by_year = {p.fiscal_year: p for p in cf}
        for p in inc[:5]:
            cf_p = cf_by_year.get(p.fiscal_year)
            row = (
                f"- FY{p.fiscal_year}: "
                f"Revenue {_money(p.revenue)} · "
                f"OpInc {_money(p.operating_income)} · "
                f"NetInc {_money(p.net_income)}"
            )
            if cf_p:
                row += f" · FCF {_money(cf_p.free_cash_flow)}"
            if p.eps_diluted is not None:
                row += f" · EPS {p.eps_diluted:.2f}"
            lines.append(row)
        lines.append("")

    # Ratios — latest period
    if report.ratios and report.ratios.latest():
        r = report.ratios.latest()
        lines.append("## Ratios (latest period)")
        lines.append(
            f"- Gross margin: {_pct(r.gross_margin)} · "
            f"Operating margin: {_pct(r.operating_margin)} · "
            f"Net margin: {_pct(r.net_margin)}"
        )
        lines.append(
            f"- ROE: {_pct(r.roe)} · ROIC: {_pct(r.roic)} · "
            f"Asset turnover: {_x(r.asset_turnover)}"
        )
        lines.append(
            f"- Current ratio: {_x(r.current_ratio)} · "
            f"D/E: {_x(r.debt_to_equity)} · D/EBITDA: {_x(r.debt_to_ebitda)} · "
            f"Interest coverage: {_x(r.interest_coverage)}"
        )
        lines.append(
            f"- FCF margin: {_pct(r.fcf_margin)} · "
            f"FCF/NI: {_x(r.fcf_to_net_income)} · "
            f"Capex intensity: {_pct(r.capex_intensity)} · "
            f"DSO: {_x(r.dso)} days"
        )
        if report.ratios.share_count_cagr_3y is not None:
            lines.append(
                f"- Share count 3y CAGR: {report.ratios.share_count_cagr_3y * 100:+.1f}% "
                f"({'dilution' if report.ratios.share_count_cagr_3y > 0 else 'buybacks'})"
            )
        lines.append("")

    # DCF
    if report.dcf:
        d = report.dcf
        lines.append("## DCF valuation")
        lines.append(
            f"- WACC: {d.wacc * 100:.2f}% · Beta: {_x(d.beta)} · "
            f"Risk-free rate: {d.risk_free_rate * 100:.2f}%"
        )
        lines.append(
            f"- Current price: {_money(d.current_price)} · " f"Net debt: {_money(d.net_debt)}"
        )
        for name, sc in [("Bear", d.bear), ("Base", d.base), ("Bull", d.bull)]:
            if sc is None:
                continue
            lines.append(
                f"- {name}: implied {_money(sc.implied_price)} "
                f"({sc.upside_pct:+.1%} vs current) · "
                f"growth Yr1-3 {sc.assumptions.revenue_growth_yr1_3 * 100:.1f}% · "
                f"FCF margin {sc.assumptions.target_fcf_margin * 100:.1f}%"
            )
        if d.implied_growth_rate is not None:
            lines.append(
                f"- Reverse-DCF implied growth (market expects): "
                f"{d.implied_growth_rate * 100:.1f}%"
            )
        lines.append("")

    # Peer comp summary
    if report.multiples and report.multiples.peers:
        peers = report.multiples.peers[:5]
        lines.append("## Peer comp (subject + up to 5 peers)")
        lines.append("Ticker | P/E | EV/EBITDA | EV/Sales | Op margin | Rev grw YoY")
        for snap in [report.multiples.subject, *peers]:
            lines.append(
                f"- {snap.symbol}: "
                f"P/E {_x(snap.pe_trailing)} · "
                f"EV/EBITDA {_x(snap.ev_to_ebitda)} · "
                f"EV/Sales {_x(snap.ev_to_sales)} · "
                f"Op mgn {_pct(snap.operating_margin)} · "
                f"Rev grw {_pct(snap.revenue_growth_yoy)}"
            )
        lines.append("")

    # Red flags
    if report.red_flags and report.red_flags.flags:
        lines.append("## Red flags")
        for f in report.red_flags.flags[:10]:
            lines.append(f"- [{f.severity.value}] {f.code}: {f.detail}")
        lines.append("")

    # Ecosystem
    if report.ecosystem:
        eco = report.ecosystem
        lines.append("## Ownership & ecosystem")
        if eco.holders:
            lines.append(
                f"- Institutional ownership: {_pct(eco.holders.pct_institutional)} · "
                f"Insider ownership: {_pct(eco.holders.pct_insider)}"
            )
            top = eco.holders.top_holders[:5]
            if top:
                names = ", ".join(f"{h.name} ({_pct(h.pct_held)})" for h in top)
                lines.append(f"- Top holders: {names}")
        if eco.insiders:
            sign = "buying" if eco.insiders.net_buying_usd > 0 else "selling"
            lines.append(
                f"- Insider activity ({eco.insiders.lookback_days}d): "
                f"net {sign} {_money(abs(eco.insiders.net_buying_usd))} · "
                f"C-suite buying: {'yes' if eco.insiders.c_suite_buying else 'no'}"
            )
        if eco.top_customers:
            lines.append(
                "- Key customers: "
                + ", ".join(
                    (f"{c.name} (~{c.concentration_pct:.0f}%)" if c.concentration_pct else c.name)
                    for c in eco.top_customers[:5]
                )
            )
        if eco.top_suppliers:
            lines.append("- Key suppliers: " + ", ".join(s.name for s in eco.top_suppliers[:5]))
        lines.append("")

    # Catalysts & risks
    if report.catalyst_risk:
        cr = report.catalyst_risk
        today = date.today()
        upcoming = []
        for c in cr.catalysts:
            try:
                d_ = date.fromisoformat(c.expected_date[:10])
                if d_ >= today:
                    upcoming.append((d_.isoformat(), c))
            except (ValueError, AttributeError):
                upcoming.append((c.expected_date or "TBD", c))
        if upcoming:
            lines.append("## Upcoming catalysts")
            for when, c in upcoming[:8]:
                lines.append(f"- {when}: [{c.catalyst_type.value}] {c.description}")
            lines.append("")
        if cr.risks:
            lines.append("## Risk register (top by probability×impact)")
            ranked = sorted(
                cr.risks,
                key=lambda r: (r.risk_score or 0),
                reverse=True,
            )[:8]
            for rk in ranked:
                p = f"{rk.probability:.0%}" if rk.probability is not None else "—"
                i = f"{rk.impact:.0%}" if rk.impact is not None else "—"
                lines.append(
                    f"- [{rk.severity.value}] {rk.category}: {rk.description} " f"(P={p}, I={i})"
                )
            lines.append("")

    # Pipeline's own thesis if present (useful prior)
    if report.thesis and (report.thesis.summary or report.thesis.base):
        lines.append("## Pipeline's existing thesis (prior — feel free to disagree)")
        if report.thesis.summary:
            lines.append(report.thesis.summary)
        for label, sc in [
            ("Bear", report.thesis.bear),
            ("Base", report.thesis.base),
            ("Bull", report.thesis.bull),
        ]:
            if sc is None:
                continue
            tgt = f"${sc.target_price:,.2f}" if sc.target_price else "—"
            lines.append(f"- {label} ({sc.probability:.0%}): target {tgt} — {sc.description}")
        lines.append("")

    return "\n".join(lines)


# ── Tiny formatters ─────────────────────────────────────────────────────────


def _money(v: float | None) -> str:
    if v is None:
        return "—"
    return f"${_abbrev_num(v)}"


def _abbrev(v: float | None) -> str:
    if v is None:
        return "—"
    return f"${_abbrev_num(v)}"


def _abbrev_num(v: float) -> str:
    abs_v = abs(v)
    if abs_v >= 1e12:
        return f"{v / 1e12:,.2f}T"
    if abs_v >= 1e9:
        return f"{v / 1e9:,.2f}B"
    if abs_v >= 1e6:
        return f"{v / 1e6:,.1f}M"
    if abs_v >= 1e3:
        return f"{v / 1e3:,.1f}K"
    return f"{v:,.2f}"


def _pct(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v * 100:+.1f}%"


def _x(v: float | None) -> str:
    if v is None:
        return "—"
    return f"{v:.2f}x"
