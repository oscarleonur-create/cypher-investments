"""Outcome tracker — log scanner signals and track 30-day outcomes."""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import yfinance as yf

SCANNERS = ["dip", "pead", "smart_money", "mispricing", "confluence", "knife_filter", "options"]

_DATA_DIR = Path(__file__).resolve().parents[3] / "data"
_SIGNALS_FILE = _DATA_DIR / "signal_outcomes.json"


def _load_signals() -> List[Dict[str, Any]]:
    if _SIGNALS_FILE.exists():
        return json.loads(_SIGNALS_FILE.read_text())
    return []


def _save_signals(signals: List[Dict[str, Any]]) -> None:
    _DATA_DIR.mkdir(parents=True, exist_ok=True)
    _SIGNALS_FILE.write_text(json.dumps(signals, indent=2, default=str))


def log_signal(
    ticker: str,
    scanner: str,
    score: float,
    verdict: str = "",
    price: float = 0.0,
    metadata: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    """Add a new signal record.

    Automatically snapshots ML features at signal time to prevent
    look-ahead bias when this data is used for training.
    """
    meta = dict(metadata) if metadata else {}

    # Snapshot features at signal time (prevents leakage in training merge)
    if "features" not in meta:
        try:
            from advisor.ml.features import FeatureEngine

            engine = FeatureEngine()
            feats = engine.compute_features(ticker)
            if feats:
                meta["features"] = feats
        except Exception:
            pass  # non-critical — training will fall back to recompute

    signals = _load_signals()
    record = {
        "id": str(uuid.uuid4())[:8],
        "ticker": ticker.upper(),
        "scanner": scanner,
        "score": score,
        "verdict": verdict,
        "signal_date": datetime.now().isoformat(),
        "entry_price": price,
        "metadata": meta,
        "resolved": False,
        "outcome_30d": None,
        "resolved_date": None,
    }
    signals.append(record)
    _save_signals(signals)
    return record


def resolve_signals() -> int:
    """Resolve all unresolved signals older than 30 days via batch yfinance download."""
    signals = _load_signals()
    cutoff = datetime.now() - timedelta(days=30)
    pending = [
        s
        for s in signals
        if not s["resolved"] and datetime.fromisoformat(s["signal_date"]) < cutoff
    ]
    if not pending:
        return 0

    tickers = list({s["ticker"] for s in pending})
    # Batch download — 35 days to ensure we cover the 30-day window
    data = yf.download(tickers, period="35d", group_by="ticker", progress=False)

    resolved_count = 0
    for sig in pending:
        tk = sig["ticker"]
        try:
            if len(tickers) == 1:
                prices = data["Close"].dropna()
            else:
                prices = data[tk]["Close"].dropna()

            sig_date = datetime.fromisoformat(sig["signal_date"]).date()
            # Get prices from signal date onward (up to 30 trading days)
            mask = prices.index.date >= sig_date
            window = prices[mask].iloc[:30]
            if len(window) < 5:
                continue

            entry = sig["entry_price"] if sig["entry_price"] > 0 else float(window.iloc[0])
            returns = (window / entry) - 1
            final_price = float(window.iloc[-1])

            sig["outcome_30d"] = {
                "price": round(final_price, 2),
                "return_pct": round(float(returns.iloc[-1]) * 100, 2),
                "max_drawdown": round(float(returns.min()) * 100, 2),
                "max_gain": round(float(returns.max()) * 100, 2),
            }
            sig["resolved"] = True
            sig["resolved_date"] = datetime.now().isoformat()
            resolved_count += 1
        except Exception:
            continue

    _save_signals(signals)
    return resolved_count


def get_stats(scanner: Optional[str] = None) -> Dict[str, Any]:
    """Win rate, avg return, avg drawdown by scanner or overall."""
    signals = _load_signals()
    resolved = [s for s in signals if s["resolved"] and s["outcome_30d"]]
    if scanner:
        resolved = [s for s in resolved if s["scanner"] == scanner]

    if not resolved:
        return {"count": 0, "win_rate": 0, "avg_return": 0, "avg_drawdown": 0}

    returns = [s["outcome_30d"]["return_pct"] for s in resolved]
    drawdowns = [s["outcome_30d"]["max_drawdown"] for s in resolved]
    wins = sum(1 for r in returns if r > 0)

    return {
        "count": len(resolved),
        "win_rate": round(wins / len(resolved) * 100, 1),
        "avg_return": round(sum(returns) / len(returns), 2),
        "avg_drawdown": round(sum(drawdowns) / len(drawdowns), 2),
    }


def export_training_data() -> pd.DataFrame:
    """Return resolved signals as a DataFrame for meta-ensemble training."""
    signals = _load_signals()
    resolved = [s for s in signals if s["resolved"] and s["outcome_30d"]]
    if not resolved:
        return pd.DataFrame()

    rows = []
    for s in resolved:
        row = {
            "ticker": s["ticker"],
            "scanner": s["scanner"],
            "score": s["score"],
            "verdict": s["verdict"],
            "entry_price": s["entry_price"],
            "return_pct": s["outcome_30d"]["return_pct"],
            "max_drawdown": s["outcome_30d"]["max_drawdown"],
            "max_gain": s["outcome_30d"]["max_gain"],
            "win": 1 if s["outcome_30d"]["return_pct"] > 5 else 0,
        }
        row.update(s.get("metadata", {}))
        rows.append(row)
    return pd.DataFrame(rows)


def status_summary() -> Dict[str, Any]:
    """Quick status of the signal database."""
    signals = _load_signals()
    return {
        "total": len(signals),
        "resolved": sum(1 for s in signals if s["resolved"]),
        "unresolved": sum(1 for s in signals if not s["resolved"]),
    }
