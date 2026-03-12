"""LLM prompt templates for the strategy case synthesis."""

CASE_SYNTHESIS_SYSTEM = """\
You are a senior options strategist. Given the analysis data below, synthesize \
a structured trade case for an options position.

Your output must be a JSON object with these exact fields:
- thesis_summary: 2-3 sentence pitch explaining why this trade makes sense
- entry_criteria: list of 2-4 conditions that should hold before entering
- exit_plan: list of 2-3 exit rules (profit target, time-based, stop)
- invalidation: list of 2-3 conditions that would kill the thesis
- risks: list of 2-4 key risks specific to this trade
- management_plan: list of 2-3 ongoing management rules
- verdict: one of STRONG, MODERATE, WEAK, REJECT
- conviction_score: integer 0-100

Verdict guidelines:
- STRONG (75-100): High-confidence setup — strong scenario fit, good IV, solid fundamentals
- MODERATE (50-74): Reasonable setup with some caveats — worth monitoring
- WEAK (25-49): Marginal setup — too many concerns or poor risk/reward
- REJECT (0-24): Do not trade — fundamental problems, bad timing, or no edge

Be specific and actionable. Reference actual numbers from the data."""


def build_synthesis_prompt(
    symbol: str,
    scenario_summary: str,
    strategy_name: str,
    strategy_reasoning: str,
    strike_details: str,
    risk_details: str,
    research_summary: str | None = None,
) -> str:
    """Build the user prompt for case synthesis."""
    sections = [
        f"## Symbol: {symbol}",
        f"\n## Scenario\n{scenario_summary}",
        f"\n## Selected Strategy: {strategy_name}\n{strategy_reasoning}",
        f"\n## Strike Analysis\n{strike_details}",
        f"\n## Risk Profile\n{risk_details}",
    ]

    if research_summary:
        sections.append(f"\n## Fundamental Research\n{research_summary}")

    sections.append(
        "\n## Instructions\n"
        "Synthesize the above into a trade case. "
        "Weight your conviction score based on:\n"
        "- Scenario confidence (20 pts)\n"
        "- Strategy fit (15 pts)\n"
        "- Research verdict (20 pts — redistribute if no research)\n"
        "- Options sell score (20 pts)\n"
        "- Risk profile POP + EV (25 pts)"
    )

    return "\n".join(sections)
