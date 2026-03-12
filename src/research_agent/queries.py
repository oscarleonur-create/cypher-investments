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


def step3_sec_queries(input: ResearchInput) -> dict[str, str]:
    """Return SEC-specific queries for filing-sourced data.

    Only meaningful for TICKER mode — SEC filings are company-specific.
    Returns empty dict for SECTOR and THESIS modes.
    """
    if input.mode != InputMode.TICKER:
        return {}

    ticker = input.value.upper()
    year = date.today().year
    return {
        "earnings_sec": f"{ticker} 10-Q quarterly earnings revenue net income {year}",
        "balance_sheet_sec": f"{ticker} 10-K balance sheet total assets liabilities cash",
        "guidance_sec": f"{ticker} 8-K forward guidance outlook management commentary {year}",
        "valuation_sec": f"{ticker} 10-K annual report revenue segments operating income",
    }


TRANSCRIPT_DOMAINS = ["seekingalpha.com", "fool.com", "nasdaq.com"]


def _infer_latest_quarter(today: date | None = None) -> tuple[str, int]:
    """Return (quarter label, year) for the most recent completed fiscal quarter.

    Q1 = Jan-Mar, Q2 = Apr-Jun, Q3 = Jul-Sep, Q4 = Oct-Dec.
    If today is in Q1 (Jan-Mar), the latest completed quarter is Q4 of the prior year.
    """
    if today is None:
        today = date.today()
    month = today.month
    if month <= 3:
        return "Q4", today.year - 1
    if month <= 6:
        return "Q1", today.year
    if month <= 9:
        return "Q2", today.year
    return "Q3", today.year


def step3_transcript_queries(input: ResearchInput) -> list[str]:
    """Return transcript-targeted search queries. Only meaningful for TICKER mode."""
    if input.mode != InputMode.TICKER:
        return []

    ticker = input.value.upper()
    quarter, year = _infer_latest_quarter()
    return [
        f"{ticker} earnings call transcript {quarter} {year}",
        f"{ticker} earnings call management guidance commentary {year}",
        f"{ticker} earnings call Q&A analyst questions {year}",
    ]


def subject_label(input: ResearchInput) -> str:
    """Return a labelled string for LLM prompts, e.g. 'Ticker: AAPL'."""
    if input.mode == InputMode.TICKER:
        return f"Ticker: {input.value.upper()}"
    if input.mode == InputMode.SECTOR:
        return f"Sector: {input.value}"
    return f"Thesis: {input.value}"
