"""Top-level pipeline orchestrator."""

from __future__ import annotations

import logging

from research_agent.agent import run_loop
from research_agent.config import ResearchConfig
from research_agent.evidence import SourceRegistry
from research_agent.llm import ClaudeLLM
from research_agent.models import AgentState, OpportunityCard, ResearchInput
from research_agent.search import TavilyClient
from research_agent.store import Store

logger = logging.getLogger(__name__)


def run(input: ResearchInput, config: ResearchConfig) -> OpportunityCard:
    """Run the full research pipeline for a given input."""

    store = Store(config.db_path)
    search = TavilyClient(config, store)
    llm = ClaudeLLM(config)
    registry = SourceRegistry()

    state = AgentState(input=input)

    try:
        card = run_loop(state, search, llm, registry, config)
    finally:
        store.close()

    return card
