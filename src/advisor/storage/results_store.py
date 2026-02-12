"""JSON file storage for backtest results."""

from __future__ import annotations

import json
import logging
from pathlib import Path

from advisor.engine.results import BacktestResult

logger = logging.getLogger(__name__)

DEFAULT_RESULTS_DIR = Path("data/results")


class ResultsStore:
    """Persists and retrieves backtest results as JSON files."""

    def __init__(self, results_dir: Path | str = DEFAULT_RESULTS_DIR):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)

    def save(self, result: BacktestResult) -> Path:
        """Save a backtest result. Returns the file path."""
        filename = f"{result.run_id}.json"
        path = self.results_dir / filename
        path.write_text(result.model_dump_json(indent=2))
        logger.info(f"Saved result: {path}")
        return path

    def load(self, run_id: str) -> BacktestResult:
        """Load a backtest result by run_id."""
        path = self.results_dir / f"{run_id}.json"
        if not path.exists():
            raise FileNotFoundError(f"Result not found: {run_id}")
        data = json.loads(path.read_text())
        return BacktestResult(**data)

    def list_results(
        self,
        strategy_name: str | None = None,
        limit: int | None = None,
    ) -> list[BacktestResult]:
        """List stored results, optionally filtered by strategy name."""
        results = []
        for path in sorted(self.results_dir.glob("*.json"), reverse=True):
            try:
                data = json.loads(path.read_text())
                result = BacktestResult(**data)
                if strategy_name and result.strategy_name != strategy_name:
                    continue
                results.append(result)
            except Exception as e:
                logger.warning(f"Failed to load {path}: {e}")

        if limit:
            results = results[:limit]
        return results

    def delete(self, run_id: str) -> bool:
        """Delete a result by run_id."""
        path = self.results_dir / f"{run_id}.json"
        if path.exists():
            path.unlink()
            return True
        return False
