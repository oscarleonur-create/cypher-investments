"""Meta-ensemble — combine scanner scores into unified conviction score."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Dict

import numpy as np

from advisor.ml.outcome_tracker import export_training_data

_DATA_DIR = Path(__file__).resolve().parents[3] / "data"
_MODEL_FILE = _DATA_DIR / "meta_model.pkl"

HEURISTIC_WEIGHTS = {
    "dip_score": 0.25,
    "smart_money_score": 0.20,
    "mispricing_score": 0.20,
    "confluence_score": 0.15,
    "pead_score": 0.10,
    "knife_filter": 0.10,
}

FEATURE_COLS = list(HEURISTIC_WEIGHTS.keys()) + ["vix"]


def _normalize(value: float, lo: float = 0.0, hi: float = 100.0) -> float:
    if hi == lo:
        return 0.5
    return max(0.0, min(1.0, (value - lo) / (hi - lo)))


def _heuristic_score(scores: Dict[str, float]) -> float:
    total = 0.0
    for key, weight in HEURISTIC_WEIGHTS.items():
        raw = scores.get(key, 0.0)
        if key == "knife_filter":
            norm = 1.0 if raw else 0.0
        else:
            norm = _normalize(raw)
        total += norm * weight
    return round(total * 100, 1)


class MetaEnsemble:
    def __init__(self) -> None:
        self.model = None
        self.avg_win: float = 0.10
        self.avg_loss: float = 0.05
        self._load_model()

    def _load_model(self) -> None:
        if _MODEL_FILE.exists():
            with open(_MODEL_FILE, "rb") as f:
                data = pickle.load(f)
            self.model = data.get("model")
            self.avg_win = data.get("avg_win", 0.10)
            self.avg_loss = data.get("avg_loss", 0.05)

    def train(self, min_samples: int = 30) -> Dict[str, Any]:
        """Train logistic regression on outcome data."""
        from sklearn.linear_model import LogisticRegression

        df = export_training_data()
        if len(df) < min_samples:
            return {"trained": False, "reason": f"Only {len(df)} samples (need {min_samples})"}

        # Build feature matrix
        for col in FEATURE_COLS:
            if col not in df.columns:
                df[col] = 0.0
        X = df[FEATURE_COLS].fillna(0).values
        y = df["win"].values

        model = LogisticRegression(max_iter=1000)
        model.fit(X, y)

        # Compute avg win/loss for Kelly
        winners = df[df["win"] == 1]["return_pct"]
        losers = df[df["win"] == 0]["return_pct"]
        avg_win = float(winners.mean()) / 100 if len(winners) > 0 else 0.10
        avg_loss = abs(float(losers.mean())) / 100 if len(losers) > 0 else 0.05

        _DATA_DIR.mkdir(parents=True, exist_ok=True)
        with open(_MODEL_FILE, "wb") as f:
            pickle.dump({"model": model, "avg_win": avg_win, "avg_loss": avg_loss}, f)

        self.model = model
        self.avg_win = avg_win
        self.avg_loss = avg_loss

        return {"trained": True, "samples": len(df), "win_rate": round(y.mean() * 100, 1)}

    def predict(self, scores_dict: Dict[str, float]) -> Dict[str, Any]:
        """Produce unified score, win probability, and kelly fraction."""
        unified = _heuristic_score(scores_dict)

        if self.model is not None:
            features = np.array([[scores_dict.get(c, 0.0) for c in FEATURE_COLS]])
            win_prob = float(self.model.predict_proba(features)[0, 1])
            mode = "learned"
            confidence = "high" if len(export_training_data()) >= 100 else "medium"

            # Kelly criterion
            b = self.avg_win / self.avg_loss if self.avg_loss > 0 else 2.0
            p = win_prob
            q = 1 - p
            kelly = (b * p - q) / b if b > 0 else 0.0
            kelly = max(0.0, min(0.25, kelly))
        else:
            win_prob = unified / 100
            mode = "heuristic"
            confidence = "low"
            kelly = 0.0

        return {
            "unified_score": unified,
            "win_prob": round(win_prob, 3),
            "kelly_fraction": round(kelly, 4),
            "mode": mode,
            "confidence": confidence,
        }

    @property
    def model_status(self) -> str:
        return "trained" if self.model is not None else "heuristic-only"
