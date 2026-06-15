"""Interactive, tool-calling research agent.

A thin tool-calling loop over OpenRouter that lets the user interrogate and
extend a cached :class:`~advisor.research.models.ResearchReport`: it can read
report sections, search the web, and re-run project compute (DCF, transcripts,
deep research, Bayesian, filings, options flow, live portfolio data) on demand.

This is distinct from ``research_agent`` (the fixed 4-step dip-card pipeline);
it reuses that package's LLM/search config and clients.
"""
