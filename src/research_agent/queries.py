"""Mode-dependent search queries and prompt labels."""

from __future__ import annotations

from datetime import date

from research_agent.models import InputMode, ResearchInput


def step1_queries(input: ResearchInput) -> list[str]:
    """Return 2 trigger-detection search queries appropriate for the input mode."""
    year = date.today().year
    if input.mode == InputMode.TICKER:
        ticker = input.value.upper()
        return [
            f"{ticker} stock price drop reason {year}",
            f"{ticker} earnings catalyst decline recent",
        ]
    if input.mode == InputMode.SECTOR:
        sector = input.value
        return [
            f"{sector} sector selloff reason {year}",
            f"{sector} sector decline catalyst recent news",
        ]
    # THESIS
    thesis = input.value
    return [
        f"{thesis} market selloff {year}",
        f"{thesis} investment thesis decline recent",
    ]


def step3_queries(input: ResearchInput) -> dict[str, str]:
    """Return 6 category queries for deep-dive fact research."""
    year = date.today().year
    if input.mode == InputMode.TICKER:
        ticker = input.value.upper()
        return {
            "earnings": f"{ticker} earnings results revenue EPS {year}",
            "guidance": f"{ticker} forward guidance outlook forecast {year}",
            "competitive": f"{ticker} competitive position market share industry",
            "balance_sheet": f"{ticker} balance sheet cash debt free cash flow",
            "valuation": f"{ticker} valuation PE ratio compared peers historical",
            "bear_case": f"{ticker} risks bear case concerns problems {year}",
        }
    if input.mode == InputMode.SECTOR:
        sector = input.value
        return {
            "earnings": f"{sector} sector earnings trends revenue {year}",
            "guidance": f"{sector} sector outlook forecast guidance {year}",
            "competitive": f"{sector} sector leaders market share competition",
            "balance_sheet": f"{sector} sector balance sheets cash debt levels",
            "valuation": f"{sector} sector valuation multiples compared historical",
            "bear_case": f"{sector} sector risks bear case headwinds {year}",
        }
    # THESIS
    thesis = input.value
    return {
        "earnings": f"{thesis} company earnings revenue impact {year}",
        "guidance": f"{thesis} forward outlook analyst forecast {year}",
        "competitive": f"{thesis} competitive landscape winners losers",
        "balance_sheet": f"{thesis} balance sheet cash flow funding",
        "valuation": f"{thesis} valuation multiples pricing {year}",
        "bear_case": f"{thesis} risks bear case problems concerns {year}",
    }


def subject_label(input: ResearchInput) -> str:
    """Return a labelled string for LLM prompts, e.g. 'Ticker: AAPL'."""
    if input.mode == InputMode.TICKER:
        return f"Ticker: {input.value.upper()}"
    if input.mode == InputMode.SECTOR:
        return f"Sector: {input.value}"
    return f"Thesis: {input.value}"
