"""Markdown renderer for ResearchReport.

Sections map 1:1 to the 7-layer analyst blueprint:
  §1 Business
  §2 Ecosystem
  §3 Industry & Competitive Position
  §4 Financial Analysis
  §5 Valuation
  §6 Catalysts & Risks
  §7 Investment Thesis
"""

from __future__ import annotations

from advisor.research.models import ResearchReport


def render_report(report: ResearchReport) -> str:
    """Return the full Markdown string for a ResearchReport.

    Sections are ordered by blueprint layer depth:
      §0 Variant Perception (edge vs. consensus)
      §1 Business  → §2 Ecosystem  → §3 Industry
      → §4 Financials → §5 Valuation → §6 Catalysts → §7 Thesis
    Each section increases analytical depth and data dependency.
    """
    parts: list[str] = []

    parts.append(f"# Research Report: {report.symbol}")
    parts.append(f"*As of {report.as_of}  |  Generated {report.created_at:%Y-%m-%d %H:%M}*\n")
    parts.append(_render_level_index(report))

    parts.append(_render_variant_perception(report))
    parts.append(_render_business(report))
    parts.append(_render_ecosystem(report))
    parts.append(_render_industry(report))
    parts.append(_render_financials(report))
    parts.append(_render_valuation(report))
    parts.append(_render_catalysts_risks(report))
    parts.append(_render_thesis(report))
    parts.append(_render_network_hint(report))

    if report.notes:
        parts.append("## Notes\n")
        for n in report.notes:
            parts.append(f"- {n}")

    return "\n".join(p for p in parts if p)


# ── Section renderers ─────────────────────────────────────────────────────────


def _render_level_index(report: ResearchReport) -> str:
    """One-line index showing which layers have data."""
    layers = [
        ("L0 Edge", report.variant_perception),
        ("L1 Business", report.business_model or report.market_data),
        ("L2 Ecosystem", report.ecosystem),
        ("L3 Industry", report.industry or report.transcripts),
        ("L4 Financials", report.ratios or report.statements),
        ("L5 Valuation", report.multiples or report.dcf),
        ("L6 Catalysts", report.catalyst_risk),
        ("L7 Thesis", report.thesis),
    ]
    items = []
    for label, has_data in layers:
        mark = "✓" if has_data else "·"
        items.append(f"{mark} {label}")
    return "**Depth index:** " + "  →  ".join(items) + "\n"


def _render_variant_perception(report: ResearchReport) -> str:
    """§0 Variant Perception — our view vs. market consensus."""
    lines = ["## §0 Variant Perception\n"]

    vp = report.variant_perception
    cn = report.consensus

    if vp is None and cn is None:
        lines.append("*No consensus or variant perception data.*\n")
        return "\n".join(lines)

    # Key insight headline
    if vp and vp.our_key_insight:
        lines.append(f"> **Edge:** {vp.our_key_insight}\n")

    # Side-by-side comparison table
    if cn or vp:
        lines.append("### Market vs. Our View\n")
        lines.append("| Metric | Market / Consensus | Our Model |")
        lines.append("|--------|-------------------|-----------|")

        if cn:
            rec = cn.recommendation_key.capitalize() if cn.recommendation_key else "—"
            n = f" ({cn.n_analysts} analysts)" if cn.n_analysts else ""
            lines.append(f"| Recommendation | {rec}{n} | — |")

        if (cn and cn.target_price_mean) or (vp and vp.our_base_price):
            cons_t = f"${cn.target_price_mean:.2f}" if cn and cn.target_price_mean else "—"
            our_t = f"${vp.our_base_price:.2f}" if vp and vp.our_base_price else "—"
            lines.append(f"| Target Price | {cons_t} | {our_t} |")

        if cn and cn.consensus_upside_pct is not None:
            cons_up = f"{cn.consensus_upside_pct:+.1%}"
            our_up = (
                f"{((vp.our_base_price / (cn.current_price or 1)) - 1):+.1%}"
                if vp and vp.our_base_price and cn and cn.current_price
                else "—"
            )
            lines.append(f"| Upside (from current) | {cons_up} | {our_up} |")

        if vp and vp.market_implied_growth is not None:
            our_g = f"{vp.our_base_growth:.1%}" if vp.our_base_growth is not None else "—"
            lines.append(f"| Implied Revenue Growth | {vp.market_implied_growth:.1%} | {our_g} |")

        if cn:
            lines.append(f"| Estimate Revision Trend | {cn.revision_trend.capitalize()} | — |")
            if cn.upgrades_30d or cn.downgrades_30d:
                lines.append(
                    f"| Rating Changes (30d) | "
                    f"↑{cn.upgrades_30d} upgrades / ↓{cn.downgrades_30d} downgrades | — |"
                )

        if cn and cn.target_price_low and cn.target_price_high:
            lines.append(
                f"| Consensus Target Range | "
                f"${cn.target_price_low:.2f} – ${cn.target_price_high:.2f} | — |"
            )

        lines.append("")

    # Our edge narrative
    if vp and vp.our_edge:
        lines.append("### Why We Differ\n")
        lines.append(f"{vp.our_edge}\n")

    # What the market misses
    if vp and vp.consensus_misses:
        lines.append("### What Consensus Is Missing\n")
        for miss in vp.consensus_misses:
            lines.append(f"- {miss}")
        lines.append("")

    # Mispricing type + conviction
    if vp:
        mt = vp.mispricing_type.value.replace("_", " ").title()
        lines.append(f"**Mispricing type:** {mt}  |  " f"**Edge conviction:** {vp.conviction}\n")

    return "\n".join(lines)


def _render_business(report: ResearchReport) -> str:
    lines = ["## §1 Business\n"]

    if report.business_model:
        lines.append(f"**Company:** {report.business_model} ({report.symbol})\n")

    if report.market_data:
        md = report.market_data
        items = []
        if md.float_shares:
            items.append(f"Float: {_big(md.float_shares)}")
        if md.short_interest_pct is not None:
            items.append(f"Short interest: {md.short_interest_pct:.1f}%")
        if md.days_to_cover is not None:
            items.append(f"Days-to-cover: {md.days_to_cover:.1f}d")
        if md.avg_volume_10d:
            items.append(f"Avg vol 10d: {_big(md.avg_volume_10d)}")
        if items:
            lines.append("**Market data:** " + "  |  ".join(items) + "\n")

    return "\n".join(lines)


def _render_ecosystem(report: ResearchReport) -> str:
    lines = ["## §2 Ecosystem\n"]
    eco = report.ecosystem

    if eco is None:
        lines.append("*No ecosystem data.*\n")
        return "\n".join(lines)

    # Institutional holders
    if eco.holders:
        h = eco.holders
        lines.append("### Institutional Ownership\n")
        if h.pct_institutional is not None:
            lines.append(f"- **Institutional:** {h.pct_institutional:.1%}")
        if h.pct_insider is not None:
            lines.append(f"- **Insider:** {h.pct_insider:.1%}")
        if h.top_holders:
            lines.append("\n| # | Holder | Shares | % Out |")
            lines.append("|---|--------|--------|-------|")
            for i, holder in enumerate(h.top_holders[:10], 1):
                lines.append(
                    f"| {i} | {holder.name} | {_big(holder.shares)} " f"| {_pct(holder.pct_held)} |"
                )
        lines.append("")

    # Insider activity
    if eco.insiders:
        ins = eco.insiders
        lines.append("### Insider Activity (trailing 12 mo)\n")
        lines.append(f"- **Net buying:** {_big(ins.net_buying_usd, signed=True)}")
        lines.append(f"- **C-suite buying:** {'Yes' if ins.c_suite_buying else 'No'}")
        if ins.transactions:
            lines.append("\n| Date | Insider | Title | Type | Value |")
            lines.append("|------|---------|-------|------|-------|")
            for tx in ins.transactions[:8]:
                lines.append(
                    f"| {tx.transaction_date or ''} | {tx.insider_name[:25]} "
                    f"| {tx.title[:15]} | {tx.transaction_type} "
                    f"| {_big(tx.value_usd)} |"
                )
        lines.append("")

    # Customers
    if eco.top_customers:
        lines.append("### Key Customers\n")
        lines.append("| Customer | Revenue % | Note |")
        lines.append("|----------|-----------|------|")
        for c in eco.top_customers:
            pct = f"{c.concentration_pct:.0f}%" if c.concentration_pct else "—"
            lines.append(f"| {c.name} | {pct} | {c.note} |")
        lines.append("")

    # Suppliers
    if eco.top_suppliers:
        lines.append("### Key Suppliers\n")
        lines.append("| Supplier | Category | Note |")
        lines.append("|----------|----------|------|")
        for s in eco.top_suppliers:
            lines.append(f"| {s.name} | {s.category} | {s.note} |")
        lines.append("")

    return "\n".join(lines)


def _render_industry(report: ResearchReport) -> str:
    lines = ["## §3 Industry & Competitive Position\n"]
    ind = report.industry

    if ind is None:
        lines.append("*No industry analysis.*\n")
        return "\n".join(lines)

    if ind.sector or ind.industry:
        lines.append(f"**Sector:** {ind.sector}  |  **Industry:** {ind.industry}\n")

    if ind.moat_type and ind.moat_type.value != "none":
        lines.append(f"**Moat:** {ind.moat_type.value.replace('_', ' ').title()}")
        if ind.moat_description:
            lines.append(f"— {ind.moat_description}")
        lines.append("")

    if ind.market_share_narrative:
        lines.append(f"**Market position:** {ind.market_share_narrative}\n")

    if ind.porters_forces:
        pf = ind.porters_forces
        lines.append("### Porter's 5 Forces\n")
        lines.append("| Force | Threat/Power |")
        lines.append("|-------|--------------|")
        forces = [
            ("Competitive Rivalry", pf.competitive_rivalry),
            ("Supplier Power", pf.supplier_power),
            ("Buyer Power", pf.buyer_power),
            ("Threat of New Entrants", pf.threat_of_new_entrants),
            ("Threat of Substitutes", pf.threat_of_substitutes),
        ]
        for label, strength in forces:
            lines.append(f"| {label} | {strength.value} |")
        if pf.rationale:
            lines.append(f"\n*{pf.rationale}*")
        lines.append("")

    if ind.key_competitors:
        lines.append(f"**Key competitors:** {', '.join(ind.key_competitors)}\n")

    # Transcripts
    if report.transcripts and report.transcripts.summaries:
        lines.append("### Earnings Call Sentiment\n")
        if report.transcripts.tone_trend:
            lines.append(f"*{report.transcripts.tone_trend}*\n")
        lines.append("| Quarter | Tone | Key Topics |")
        lines.append("|---------|------|------------|")
        for s in report.transcripts.summaries:
            topics = ", ".join(s.key_topics[:3])
            lines.append(f"| {s.quarter} | {s.tone.value} | {topics} |")
        lines.append("")

    return "\n".join(lines)


def _render_financials(report: ResearchReport) -> str:
    lines = ["## §4 Financial Analysis\n"]

    # Red flags first (most actionable)
    if report.red_flags and report.red_flags.flags:
        lines.append("### Red Flags\n")
        for f in report.red_flags.flags:
            lines.append(f"- **[{f.severity.value}]** {f.title}: {f.detail}")
        lines.append("")
    elif report.red_flags:
        lines.append("*No red flags detected.*\n")

    # Ratios
    if report.ratios and report.ratios.periods:
        lines.append("### Key Ratios\n")

        # Header row: FY years
        periods = report.ratios.periods[:5]
        header = "| Metric | " + " | ".join(str(p.fiscal_year) for p in periods) + " |"
        sep = "|--------|" + "|".join("-" * 8 for _ in periods) + "|"
        lines.append(header)
        lines.append(sep)

        ratio_rows = [
            ("Gross Margin", "gross_margin", "pct"),
            ("Op Margin", "operating_margin", "pct"),
            ("Net Margin", "net_margin", "pct"),
            ("ROIC", "roic", "pct"),
            ("ROE", "roe", "pct"),
            ("Debt/EBITDA", "debt_to_ebitda", "x"),
            ("FCF Margin", "fcf_margin", "pct"),
            ("DSO", "dso", "d"),
        ]
        for label, field, unit in ratio_rows:
            vals = " | ".join(_rfmt(getattr(p, field, None), unit) for p in periods)
            lines.append(f"| {label} | {vals} |")

        if report.ratios.share_count_cagr_3y is not None:
            lines.append(f"\n*Share count 3y CAGR: " f"{report.ratios.share_count_cagr_3y:+.1%}*")
        lines.append("")

    # Income snapshot
    if report.statements and report.statements.income:
        lines.append("### Income Trend (B USD)\n")
        lines.append("| FY | Revenue | Gross Profit | Op Income | Net Income | EPS |")
        lines.append("|----|---------|--------------|-----------|------------|-----|")
        for p in report.statements.income[:5]:
            lines.append(
                f"| {p.fiscal_year} "
                f"| {_big(p.revenue)} "
                f"| {_big(p.gross_profit)} "
                f"| {_big(p.operating_income)} "
                f"| {_big(p.net_income)} "
                f"| {_num(p.eps_diluted)} |"
            )
        lines.append("")

    return "\n".join(lines)


def _render_valuation(report: ResearchReport) -> str:
    lines = ["## §5 Valuation\n"]

    # Peer comp table
    if report.multiples:
        m = report.multiples
        lines.append("### Peer Comp Table\n")
        lines.append("| Ticker | Mkt Cap | Rev Grw | Op Mgn | EV/S | EV/EBITDA | P/E | P/FCF |")
        lines.append("|--------|---------|---------|--------|------|-----------|-----|-------|")
        for snap in m.all_snapshots:
            bold = "**" if snap.symbol == m.subject.symbol else ""
            lines.append(
                f"| {bold}{snap.symbol}{bold} "
                f"| {_big(snap.market_cap)} "
                f"| {_rfmt(snap.revenue_growth_yoy, 'pct')} "
                f"| {_rfmt(snap.operating_margin, 'pct')} "
                f"| {_rfmt(snap.ev_to_sales, 'x')} "
                f"| {_rfmt(snap.ev_to_ebitda, 'x')} "
                f"| {_rfmt(snap.pe_trailing, 'x')} "
                f"| {_rfmt(snap.p_to_fcf, 'x')} |"
            )
        lines.append("")

        # Own history
        if m.own_history:
            lines.append("### Own Multiple History\n")
            lines.append("| FY | EV/Sales | EV/EBITDA | P/E | P/FCF |")
            lines.append("|----|----------|-----------|-----|-------|")
            for h in m.own_history:
                lines.append(
                    f"| {h.fiscal_year} "
                    f"| {_rfmt(h.ev_to_sales, 'x')} "
                    f"| {_rfmt(h.ev_to_ebitda, 'x')} "
                    f"| {_rfmt(h.pe, 'x')} "
                    f"| {_rfmt(h.p_to_fcf, 'x')} |"
                )
            lines.append("")

    # DCF
    if report.dcf:
        dcf = report.dcf
        lines.append("### DCF Analysis\n")
        lines.append(
            f"Current price: **${dcf.current_price:.2f}**  |  "
            f"WACC: {dcf.wacc:.1%}  |  "
            f"β: {dcf.beta or 0:.2f}  |  "
            f"Rf: {dcf.risk_free_rate:.1%}"
        )
        if dcf.implied_growth_rate is not None:
            lines.append(
                f"\nMarket-implied growth rate (reverse-DCF): " f"**{dcf.implied_growth_rate:.1%}**"
            )
        lines.append("\n| Scenario | Growth Yr1-3 | FCF Margin | Implied Price | Upside |")
        lines.append("|----------|-------------|------------|---------------|--------|")
        for scenario in [dcf.base, dcf.bull, dcf.bear]:
            if scenario is None:
                continue
            a = scenario.assumptions
            lines.append(
                f"| {a.scenario.upper()} "
                f"| {a.revenue_growth_yr1_3:.1%} "
                f"| {a.target_fcf_margin:.1%} "
                f"| ${scenario.implied_price:.2f} "
                f"| {scenario.upside_pct:+.1%} |"
            )
        lines.append("")

    return "\n".join(lines)


def _render_catalysts_risks(report: ResearchReport) -> str:
    lines = ["## §6 Catalysts & Risks\n"]
    cr = report.catalyst_risk

    if cr is None:
        lines.append("*No catalyst/risk data.*\n")
        return "\n".join(lines)

    if cr.catalysts:
        lines.append("### Upcoming Catalysts\n")
        lines.append("| Type | Date | Description |")
        lines.append("|------|------|-------------|")
        for c in cr.catalysts:
            near = " ⚡" if c.is_near_term else ""
            lines.append(f"| {c.catalyst_type.value}{near} | {c.expected_date} | {c.description} |")
        lines.append("")

    if cr.risks:
        lines.append("### Risk Register\n")
        lines.append("| Severity | Category | P | I | Description |")
        lines.append("|----------|----------|---|---|-------------|")
        for r in cr.risks:
            p = f"{r.probability:.0%}" if r.probability is not None else "—"
            imp = f"{r.impact:.0%}" if r.impact is not None else "—"
            lines.append(
                f"| {r.severity.value} | {r.category} | {p} | {imp} " f"| {r.description[:90]} |"
            )
        lines.append("")

    return "\n".join(lines)


def _render_thesis(report: ResearchReport) -> str:
    lines = ["## §7 Investment Thesis\n"]
    th = report.thesis

    if th is None:
        lines.append("*No thesis data.*\n")
        return "\n".join(lines)

    if th.summary:
        lines.append(f"{th.summary}\n")

    lines.append(f"**Conviction:** {th.conviction}")
    if th.current_price:
        lines.append(f"  |  **Current price:** ${th.current_price:.2f}\n")

    lines.append("| Scenario | Prob | Target | Upside |")
    lines.append("|----------|------|--------|--------|")
    for s in [th.bull, th.base, th.bear]:
        if s is None:
            continue
        target = f"${s.target_price:.2f}" if s.target_price else "—"
        upside = f"{s.upside_pct:+.1%}" if s.upside_pct is not None else "—"
        lines.append(f"| {s.scenario.upper()} | {s.probability:.0%} | {target} | {upside} |")
    lines.append("")

    for s in [th.bull, th.base, th.bear]:
        if s is None or not (s.description or s.key_assumptions):
            continue
        lines.append(f"### {s.scenario.capitalize()} Case\n")
        if s.description:
            lines.append(f"{s.description}\n")
        if s.key_assumptions:
            lines.append("**What must be true:**")
            for a in s.key_assumptions:
                lines.append(f"- {a}")
        if s.what_proves_wrong:
            lines.append("\n**What proves you wrong:**")
            for w in s.what_proves_wrong:
                lines.append(f"- {w}")
        lines.append("")

    return "\n".join(lines)


def _render_network_hint(report: ResearchReport) -> str:
    """Footer showing Level-2 connections and drill-down instructions."""
    net = report.network
    if not net or not net.nodes:
        return (
            "---\n"
            f"*Level 2 network not built yet. "
            f"Run: `advisor research network {report.symbol}`*\n"
        )

    by_rel = net.by_relationship
    lines = ["---", "## Company Network\n"]
    lines.append(f"**Level 1 (subject):** {report.symbol}\n")
    lines.append("**Level 2 connections:**\n")

    rel_labels = {
        "peer": "Peers",
        "competitor": "Competitors",
        "top_holder": "Public Holders",
        "customer": "Customers",
        "supplier": "Suppliers",
    }
    for key, label in rel_labels.items():
        nodes = by_rel.get(key, [])
        if not nodes:
            continue
        tickers = ", ".join(n.symbol for n in nodes)
        lines.append(f"- **{label}:** {tickers}")

    lines.append(
        "\n**Level 3** → drill into any connection:\n"
        "```\n"
        "advisor research ticker BLK       # full 7-layer report\n"
        "advisor research network BLK      # BLK's own Level-2 map\n"
        "```"
    )
    return "\n".join(lines)


# ── Formatting helpers ────────────────────────────────────────────────────────


def _big(v: float | None, signed: bool = False) -> str:
    if v is None:
        return "—"
    sign = "+" if signed and v > 0 else ""
    abs_v = abs(v)
    if abs_v >= 1e12:
        return f"{sign}{v / 1e12:,.1f}T"
    if abs_v >= 1e9:
        return f"{sign}{v / 1e9:,.2f}B"
    if abs_v >= 1e6:
        return f"{sign}{v / 1e6:,.1f}M"
    if abs_v >= 1e3:
        return f"{sign}{v / 1e3:,.1f}K"
    return f"{sign}{v:,.2f}"


def _pct(v: float | None) -> str:
    return f"{v:.1%}" if v is not None else "—"


def _num(v: float | None) -> str:
    return f"{v:.2f}" if v is not None else "—"


def _rfmt(v: float | None, unit: str) -> str:
    if v is None:
        return "—"
    if unit == "pct":
        return f"{v * 100:+.1f}%"
    if unit == "x":
        return f"{v:.2f}x"
    if unit == "d":
        return f"{v:.0f}d"
    return f"{v:.2f}"
