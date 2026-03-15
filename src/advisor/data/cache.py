"""Disk cache for market data API calls."""

from __future__ import annotations

import hashlib
import json
import logging
import os
import tempfile
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

    def _is_valid(self, path: Path, ttl_seconds: int | None = None) -> bool:
        if not path.exists():
            return False
        if ttl_seconds is not None:
            age = datetime.now().timestamp() - path.stat().st_mtime
            return age < ttl_seconds
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

    def get_json(self, *parts: str, ttl_seconds: int | None = None) -> dict | None:
        """Return cached JSON or ``None`` if missing / expired.

        Parameters
        ----------
        *parts:
            Cache key components (joined and hashed).
        ttl_seconds:
            Optional per-call TTL override (in seconds).  When provided this
            takes precedence over the instance-level ``ttl``.
        """
        key = self._key(*parts)
        path = self._path(key, ".json")
        if self._is_valid(path, ttl_seconds=ttl_seconds):
            logger.debug(f"Cache hit: {parts}")
            return json.loads(path.read_text())
        return None

    def set_json(self, data: dict, *parts: str) -> None:
        """Atomically write *data* as JSON to the cache.

        Uses a temporary file + ``os.replace`` so that readers never see a
        partially-written file.
        """
        key = self._key(*parts)
        path = self._path(key, ".json")
        fd, tmp = tempfile.mkstemp(dir=self.cache_dir, suffix=".tmp")
        closed = False
        try:
            os.write(fd, json.dumps(data, default=str).encode())
            os.close(fd)
            closed = True
            os.replace(tmp, path)
            logger.debug(f"Cached: {parts}")
        except Exception as e:
            logger.debug("Cache write failed: %s", e)
            if not closed:
                os.close(fd)
            Path(tmp).unlink(missing_ok=True)
            raise

    def clear(self) -> int:
        """Clear all cached files. Returns count of removed files."""
        count = 0
        for f in self.cache_dir.iterdir():
            if f.is_file():
                f.unlink()
                count += 1
        return count
