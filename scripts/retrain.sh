#!/usr/bin/env bash
# Daily ML model retrain — run via cron or launchd.
#
# Usage:
#   ./scripts/retrain.sh              # normal run
#   ./scripts/retrain.sh --dry-run    # just log what would happen
#
# Cron example (4 AM weekdays):
#   0 4 * * 1-5 /Users/oscarleonuribe/Documents/trading/options/scripts/retrain.sh
#
# What it does:
#   1. Trains primary LightGBM + meta-labeling model on full history
#   2. Re-fits HMM regime detector
#   3. Snapshots all model files into a versioned directory
#   4. Logs results to data/ml_models/retrain_log.jsonl
#   5. Keeps last 30 days of logs

set -euo pipefail

# ── Paths ─────────────────────────────────────────────────────────────
PROJECT_DIR="/Users/oscarleonuribe/Documents/trading/options"
VENV="/Users/oscarleonuribe/Library/Caches/pypoetry/virtualenvs/options-zzc4wflY-py3.12"
PYTHON="${VENV}/bin/python"
ADVISOR="${VENV}/bin/advisor"
LOG_DIR="${PROJECT_DIR}/data/ml_models"
LOG_FILE="${LOG_DIR}/retrain_log.jsonl"
LOCK_FILE="${LOG_DIR}/.retrain.lock"

# ── Ensure log dir ────────────────────────────────────────────────────
mkdir -p "${LOG_DIR}"

# ── Lock (prevent concurrent runs) ───────────────────────────────────
if [ -f "${LOCK_FILE}" ]; then
    pid=$(cat "${LOCK_FILE}")
    if kill -0 "$pid" 2>/dev/null; then
        echo "Retrain already running (PID ${pid}), exiting."
        exit 0
    fi
    rm -f "${LOCK_FILE}"
fi
echo $$ > "${LOCK_FILE}"
trap 'rm -f "${LOCK_FILE}"' EXIT

# ── Dry run check ────────────────────────────────────────────────────
DRY_RUN=false
if [ "${1:-}" = "--dry-run" ]; then
    DRY_RUN=true
    echo "[DRY RUN] Would retrain model at $(date -Iseconds)"
    exit 0
fi

# ── Timestamp ─────────────────────────────────────────────────────────
STARTED_AT=$(date -Iseconds)
echo "[${STARTED_AT}] Starting ML retrain..."

cd "${PROJECT_DIR}"

# ── Step 1: Train primary + meta-labeling ─────────────────────────────
echo "  Training primary + meta model..."
TRAIN_OUTPUT=$("${ADVISOR}" ml train-meta \
    --model lightgbm \
    --label-mode fixed \
    --threshold 3.0 \
    --horizon 10 \
    --lookback 5y \
    --decay 365 \
    --output json 2>&1) || true

# Extract key metrics from JSON output
TRAIN_METRICS=$("${PYTHON}" -c "
import json, sys
try:
    data = json.loads('''${TRAIN_OUTPUT}''')
    cv = data.get('cv_metrics', {})
    meta = data.get('metadata', {})
    ml = data.get('meta_labeling', {}).get('metrics', {})
    print(json.dumps({
        'cv_auc': cv.get('cv_auc_mean', 0),
        'n_samples': meta.get('n_samples', 0),
        'n_features': meta.get('n_features', 0),
        'meta_auc': ml.get('meta_auc', 0),
        'train_cutoff': meta.get('train_cutoff', ''),
    }))
except:
    print(json.dumps({'error': 'parse_failed'}))
" 2>/dev/null) || TRAIN_METRICS='{"error": "extract_failed"}'

echo "  Train metrics: ${TRAIN_METRICS}"

# ── Step 2: Re-fit HMM regime ────────────────────────────────────────
echo "  Fitting HMM regime..."
REGIME_OUTPUT=$("${ADVISOR}" ml regime --fit --output json 2>&1) || true

REGIME_STATE=$("${PYTHON}" -c "
import json
try:
    data = json.loads('''${REGIME_OUTPUT}''')
    print(json.dumps({
        'regime': data.get('regime_name', 'unknown'),
        'vix': data.get('vix', 0),
    }))
except:
    print(json.dumps({'regime': 'unknown', 'vix': 0}))
" 2>/dev/null) || REGIME_STATE='{"regime": "error"}'

echo "  Regime: ${REGIME_STATE}"

# ── Step 3: Version snapshot ────────────────────────────────────────
echo "  Creating model version snapshot..."
VERSION_ID=$("${PYTHON}" -c "
from advisor.ml.model_store import snapshot, prune
vid = snapshot(tag='cron-retrain')
removed = prune(keep=10)
print(vid)
if removed:
    import sys
    print(f'  Pruned {len(removed)} old version(s)', file=sys.stderr)
" 2>&1) || VERSION_ID="snapshot_failed"
echo "  Version: ${VERSION_ID}"

# ── Step 4: Log results ──────────────────────────────────────────────
FINISHED_AT=$(date -Iseconds)
DURATION=$(($(date +%s) - $(date -j -f "%Y-%m-%dT%H:%M:%S" "${STARTED_AT%+*}" +%s 2>/dev/null || echo 0)))

"${PYTHON}" -c "
import json
entry = {
    'started_at': '${STARTED_AT}',
    'finished_at': '${FINISHED_AT}',
    'duration_seconds': ${DURATION} if ${DURATION} > 0 else None,
    'version_id': '${VERSION_ID}',
    'train': json.loads('${TRAIN_METRICS}'),
    'regime': json.loads('${REGIME_STATE}'),
}
with open('${LOG_FILE}', 'a') as f:
    f.write(json.dumps(entry) + '\n')
print(json.dumps(entry, indent=2))
"

# ── Step 5: Rotate logs (keep last 30 days) ──────────────────────────
if [ -f "${LOG_FILE}" ]; then
    line_count=$(wc -l < "${LOG_FILE}")
    if [ "$line_count" -gt 365 ]; then
        tail -n 365 "${LOG_FILE}" > "${LOG_FILE}.tmp"
        mv "${LOG_FILE}.tmp" "${LOG_FILE}"
        echo "  Rotated log to last 365 entries"
    fi
fi

echo "[${FINISHED_AT}] Retrain complete."
