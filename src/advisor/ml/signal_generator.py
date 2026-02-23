"""ML signal generator — convert model predictions to StrategySignal objects."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any

import pandas as pd
import yfinance as yf

from advisor.engine.signals import SignalAction, StrategySignal
from advisor.ml.features import FeatureEngine
from advisor.ml.models import MLModelTrainer

logger = logging.getLogger(__name__)

# Prediction thresholds
_BUY_THRESHOLD = 0.65
_STRONG_BUY_THRESHOLD = 0.80
_SELL_THRESHOLD = 0.35


class MLSignalGenerator:
    """Convert model predictions into actionable trading signals."""

    def __init__(self) -> None:
        self._trainer: MLModelTrainer | None = None
        self._engine = FeatureEngine()
        self._loaded = False

    def _ensure_model(self) -> bool:
        """Lazy-load the trained model."""
        if self._loaded:
            return self._trainer is not None

        self._loaded = True
        if not MLModelTrainer.model_exists():
            logger.info("No trained ML model found")
            return False

        try:
            self._trainer = MLModelTrainer.load()
            return True
        except Exception as e:
            logger.error(f"Failed to load ML model: {e}")
            return False

    def is_available(self) -> bool:
        """Check if a trained model exists and is loadable."""
        return self._ensure_model()

    def generate_signal(self, symbol: str) -> StrategySignal | None:
        """Generate a trading signal for a single symbol.

        Returns None if the model is not available or feature computation fails.
        """
        if not self._ensure_model():
            return None

        symbol = symbol.upper()
        features = self._engine.compute_features(symbol)
        if not features:
            logger.warning(f"No features computed for {symbol}")
            return None

        # Get current price
        try:
            tk = yf.Ticker(symbol)
            price = tk.fast_info.get("lastPrice", 0.0) or 0.0
        except Exception:
            price = 0.0

        # Build feature row matching training feature order
        feature_names = self._trainer.feature_names or FeatureEngine.feature_names()
        row = {}
        for col in feature_names:
            # Sensible defaults for features not available in single-symbol mode
            if col.endswith("_cs_rank"):
                default = 0.5  # Cross-sectional ranks: median
            elif col == "hmm_regime":
                default = 1.0  # Normal regime
            elif col.startswith("hmm_") and col.endswith("_prob"):
                default = 1.0 / 3.0  # Equal regime probability
            else:
                default = 0.0
            row[col] = features.get(col, default)
        features_df = pd.DataFrame([row])

        prob = float(self._trainer.predict_proba(features_df)[0])
        action, reason = self._classify(prob, symbol)

        return StrategySignal(
            strategy_name="ml_signal",
            symbol=symbol,
            action=action,
            reason=reason,
            timestamp=datetime.now(),
            price=price,
        )

    @staticmethod
    def _default_for(col: str) -> float:
        """Return a sensible default for a feature not available in live mode."""
        if col.endswith("_cs_rank"):
            return 0.5
        if col == "hmm_regime":
            return 1.0
        if col.startswith("hmm_") and col.endswith("_prob"):
            return 1.0 / 3.0
        return 0.0

    def _classify(self, prob: float, symbol: str) -> tuple[SignalAction, str]:
        """Convert win probability to signal action and reason."""
        if prob >= _STRONG_BUY_THRESHOLD:
            return (
                SignalAction.BUY,
                f"ML STRONG BUY — {prob:.1%} win probability for {symbol}",
            )
        elif prob >= _BUY_THRESHOLD:
            return (
                SignalAction.BUY,
                f"ML BUY — {prob:.1%} win probability for {symbol}",
            )
        elif prob <= _SELL_THRESHOLD:
            return (
                SignalAction.SELL,
                f"ML SELL — {prob:.1%} win probability for {symbol}",
            )
        else:
            return (
                SignalAction.NEUTRAL,
                f"ML NEUTRAL — {prob:.1%} win probability for {symbol}",
            )

    def batch_scan(self, symbols: list[str]) -> list[tuple[str, StrategySignal, dict[str, Any]]]:
        """Generate signals for multiple symbols, sorted by conviction.

        Returns list of (symbol, signal, metadata) tuples.
        """
        if not self._ensure_model():
            return []

        results = []
        for symbol in symbols:
            signal = self.generate_signal(symbol)
            if signal is None:
                continue

            features = self._engine.compute_features(symbol)
            feature_names = self._trainer.feature_names or FeatureEngine.feature_names()
            row = {col: features.get(col, self._default_for(col)) for col in feature_names}
            features_df = pd.DataFrame([row])
            prob = float(self._trainer.predict_proba(features_df)[0])

            meta = {
                "win_probability": round(prob, 4),
                "model_type": str(self._trainer.model_type),
            }
            results.append((symbol, signal, meta))

        # Sort by win probability descending (BUYs first, then by conviction)
        results.sort(key=lambda x: x[2]["win_probability"], reverse=True)
        return results

    def explain_prediction(self, symbol: str) -> dict[str, Any]:
        """Get detailed explanation of a prediction for a symbol.

        Returns feature values, importances, and the prediction breakdown.
        """
        if not self._ensure_model():
            return {"error": "No trained model available"}

        symbol = symbol.upper()
        features = self._engine.compute_features(symbol)
        if not features:
            return {"error": f"Could not compute features for {symbol}"}

        feature_names = self._trainer.feature_names or FeatureEngine.feature_names()
        row = {col: features.get(col, self._default_for(col)) for col in feature_names}
        features_df = pd.DataFrame([row])

        prob = float(self._trainer.predict_proba(features_df)[0])
        importances = self._trainer.get_feature_importance()

        # Build ranked feature breakdown
        feature_breakdown = []
        for name in feature_names:
            feature_breakdown.append(
                {
                    "feature": name,
                    "value": round(features.get(name, 0.0), 6),
                    "importance": importances.get(name, 0.0),
                }
            )
        feature_breakdown.sort(key=lambda x: x["importance"], reverse=True)

        action, reason = self._classify(prob, symbol)

        return {
            "symbol": symbol,
            "win_probability": round(prob, 4),
            "signal": action.value,
            "reason": reason,
            "model_type": str(self._trainer.model_type),
            "features": feature_breakdown,
            "top_features": feature_breakdown[:5],
            "model_metrics": self._trainer.metrics,
        }
