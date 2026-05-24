"""Deep fundamental research engine.

Implements the 7-layer analyst blueprint (business, ecosystem, industry,
financials, valuation, catalysts/risks, thesis). Phase 1 ships the financial
foundation: SEC EDGAR ingestion, 5-yr statements, ratio library, red flags.
"""

from advisor.research.config import ResearchSettings, get_settings

__all__ = ["ResearchSettings", "get_settings"]
