"""Model versioning — snapshot, list, rollback, and prune model artifacts."""

from __future__ import annotations

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parents[3] / "data"
_MODEL_DIR = _DATA_DIR / "ml_models"
_VERSIONS_DIR = _MODEL_DIR / "versions"

# Files that constitute a "model version"
_MODEL_FILES = [
    "current_model.joblib",
    "meta_model.joblib",
    "hmm_regime.joblib",
]

_MANIFEST = "manifest.json"
_MAX_VERSIONS = 10


def snapshot(metrics: dict[str, Any] | None = None, tag: str = "") -> str:
    """Create a versioned snapshot of current model files.

    Args:
        metrics: Optional metrics dict to store in the manifest.
        tag: Optional human-readable tag (e.g., "weekly-retrain", "manual").

    Returns:
        Version ID (timestamp string).
    """
    now = datetime.now()
    version_id = now.strftime("%Y%m%d_%H%M%S")
    version_dir = _VERSIONS_DIR / version_id
    version_dir.mkdir(parents=True, exist_ok=True)

    copied = []
    for fname in _MODEL_FILES:
        src = _MODEL_DIR / fname
        if src.exists():
            shutil.copy2(src, version_dir / fname)
            copied.append(fname)

    if not copied:
        shutil.rmtree(version_dir)
        raise FileNotFoundError("No model files found to snapshot")

    # Build manifest
    manifest = {
        "version_id": version_id,
        "created_at": now.isoformat(),
        "tag": tag,
        "files": copied,
        "metrics": metrics or {},
    }

    # Try to extract metadata from the primary model
    primary = _MODEL_DIR / "current_model.joblib"
    if primary.exists():
        try:
            import joblib

            payload = joblib.load(primary)
            meta = payload.get("metadata", {})
            manifest["model_metadata"] = {
                "model_type": meta.get("model_type", "unknown"),
                "n_samples": meta.get("n_samples"),
                "n_features": meta.get("n_features"),
                "train_cutoff": meta.get("train_cutoff"),
                "trained_at": meta.get("trained_at"),
                "version": payload.get("version"),
            }
            manifest["cv_metrics"] = payload.get("metrics", {})
        except Exception:
            pass

    (version_dir / _MANIFEST).write_text(json.dumps(manifest, indent=2))
    logger.info("Model snapshot created: %s (%d files)", version_id, len(copied))
    return version_id


def list_versions() -> list[dict[str, Any]]:
    """List all saved model versions, newest first."""
    if not _VERSIONS_DIR.exists():
        return []

    versions = []
    for d in sorted(_VERSIONS_DIR.iterdir(), reverse=True):
        if not d.is_dir():
            continue
        manifest_path = d / _MANIFEST
        if manifest_path.exists():
            try:
                manifest = json.loads(manifest_path.read_text())
                versions.append(manifest)
            except (json.JSONDecodeError, OSError):
                versions.append(
                    {
                        "version_id": d.name,
                        "created_at": None,
                        "files": [f.name for f in d.iterdir() if f.is_file()],
                        "error": "corrupt manifest",
                    }
                )
        else:
            versions.append(
                {
                    "version_id": d.name,
                    "created_at": None,
                    "files": [f.name for f in d.iterdir() if f.is_file()],
                }
            )

    return versions


def get_version(version_id: str) -> dict[str, Any]:
    """Get full manifest for a specific version."""
    version_dir = _VERSIONS_DIR / version_id
    if not version_dir.exists():
        raise FileNotFoundError(f"Version {version_id} not found")

    manifest_path = version_dir / _MANIFEST
    if manifest_path.exists():
        return json.loads(manifest_path.read_text())
    raise FileNotFoundError(f"No manifest for version {version_id}")


def rollback(version_id: str) -> dict[str, Any]:
    """Restore model files from a specific version.

    Creates a snapshot of the current state first (tagged 'pre-rollback'),
    then copies the versioned files back to the active model directory.

    Returns:
        Dict with rollback details.
    """
    version_dir = _VERSIONS_DIR / version_id
    if not version_dir.exists():
        raise FileNotFoundError(f"Version {version_id} not found")

    # Snapshot current state before overwriting
    try:
        backup_id = snapshot(tag=f"pre-rollback-to-{version_id}")
    except FileNotFoundError:
        backup_id = None

    # Copy versioned files to active directory
    restored = []
    for fname in _MODEL_FILES:
        src = version_dir / fname
        if src.exists():
            shutil.copy2(src, _MODEL_DIR / fname)
            restored.append(fname)

    logger.info("Rolled back to version %s (%d files)", version_id, len(restored))

    return {
        "rolled_back_to": version_id,
        "backup_version": backup_id,
        "restored_files": restored,
    }


def prune(keep: int = _MAX_VERSIONS) -> list[str]:
    """Remove old versions, keeping the most recent `keep` versions.

    Returns:
        List of removed version IDs.
    """
    if not _VERSIONS_DIR.exists():
        return []

    dirs = sorted(
        [d for d in _VERSIONS_DIR.iterdir() if d.is_dir()],
        key=lambda d: d.name,
        reverse=True,
    )

    removed = []
    for d in dirs[keep:]:
        shutil.rmtree(d)
        removed.append(d.name)
        logger.info("Pruned model version: %s", d.name)

    return removed
