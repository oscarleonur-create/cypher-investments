"""Output writer: JSON, Markdown, and SQLite persistence."""

from __future__ import annotations

from pathlib import Path

from research_agent.card import render_markdown
from research_agent.config import ResearchConfig
from research_agent.models import OpportunityCard
from research_agent.store import Store


def write_outputs(card: OpportunityCard, config: ResearchConfig) -> tuple[Path, Path]:
    """Write card to JSON and Markdown files, and persist to SQLite.

    Returns (json_path, md_path).
    """
    out_dir = config.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    json_path = out_dir / f"{card.id}.json"
    md_path = out_dir / f"{card.id}.md"

    # Write JSON
    json_path.write_text(card.model_dump_json(indent=2))

    # Write Markdown
    md_path.write_text(render_markdown(card))

    # Persist to SQLite
    store = Store(config.db_path)
    try:
        store.save_run(card)
    finally:
        store.close()

    return json_path, md_path
