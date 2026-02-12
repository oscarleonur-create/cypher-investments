"""Disk cache for market data API calls."""

from __future__ import annotations

import hashlib
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

logger = logging.getLogger(__name__)

DEFAULT_CACHE_DIR = Path("data/cache")


class DiskCache:
    """Simple disk cache for DataFrames and JSON data."""

    def __init__(self, cache_dir: Path | str = DEFAULT_CACHE_DIR, ttl_hours: int = 24):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.ttl = timedelta(hours=ttl_hours)

    def _key(self, *parts: str) -> str:
        raw = ":".join(parts)
        return hashlib.sha256(raw.encode()).hexdigest()[:16]

    def _path(self, key: str, ext: str = ".parquet") -> Path:
        return self.cache_dir / f"{key}{ext}"

    def _is_valid(self, path: Path) -> bool:
        if not path.exists():
            return False
        mtime = datetime.fromtimestamp(path.stat().st_mtime)
        return datetime.now() - mtime < self.ttl

    def get_dataframe(self, *parts: str) -> pd.DataFrame | None:
        key = self._key(*parts)
        path = self._path(key, ".parquet")
        if self._is_valid(path):
            logger.debug(f"Cache hit: {parts}")
            return pd.read_parquet(path)
        return None

    def set_dataframe(self, df: pd.DataFrame, *parts: str) -> None:
        key = self._key(*parts)
        path = self._path(key, ".parquet")
        df.to_parquet(path)
        logger.debug(f"Cached: {parts}")

    def get_json(self, *parts: str) -> dict | None:
        key = self._key(*parts)
        path = self._path(key, ".json")
        if self._is_valid(path):
            logger.debug(f"Cache hit: {parts}")
            return json.loads(path.read_text())
        return None

    def set_json(self, data: dict, *parts: str) -> None:
        key = self._key(*parts)
        path = self._path(key, ".json")
        path.write_text(json.dumps(data, default=str))
        logger.debug(f"Cached: {parts}")

    def clear(self) -> int:
        """Clear all cached files. Returns count of removed files."""
        count = 0
        for f in self.cache_dir.iterdir():
            if f.is_file():
                f.unlink()
                count += 1
        return count
