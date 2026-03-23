"""Quantile regression — model full conditional return distributions.

Trains LightGBM quantile regressors at multiple quantile levels to predict
the full distribution of forward returns conditioned on the same 79 features
used by the classification model.

Usage::

    from advisor.ml.quantile import QuantileModel

    qm = QuantileModel.load()
    result = qm.predict_distribution(features_df)
    # result = {
    #     "quantiles": {0.05: -4.2, 0.10: -2.8, ..., 0.95: 5.1},
    #     "median_return": 0.8,
    #     "expected_range": [-2.8, 3.5],
    #     "var_5": -4.2,
    #     "var_10": -2.8,
    #     "cvar_5": -5.1,
    #     "upside_ratio": 1.3,
    #     "distribution_skew": "right",
    # }
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from advisor.ml.preprocessing import FeaturePreprocessor

logger = logging.getLogger(__name__)

# ── Trend filter (works on raw OHLCV, independent of ML features) ────


def compute_trend_score(ohlcv_df: pd.DataFrame) -> dict[str, Any]:
    """Compute a 0-3 trend-quality score from raw OHLCV data.

    Checks three conditions on the most recent bar:
    - SMA 20 > SMA 50  (price structure is bullish)
    - Price > SMA 50   (not in freefall)
    - SMA 50 slope > 0 (trend direction is up over last 10 bars)

    Returns:
        Dict with ``trend_score`` (0-3), component flags, and
        ``falling_knife_warning`` (True when score < 2).
    """
    close = ohlcv_df["Close"]
    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()

    latest = close.iloc[-1]
    sma20_val = sma20.iloc[-1]
    sma50_val = sma50.iloc[-1]

    # Slope: compare current SMA-50 to 10 bars ago
    sma50_prev = sma50.iloc[-11] if len(sma50) >= 11 else sma50.iloc[0]

    sma20_above_sma50 = (
        bool(sma20_val > sma50_val) if pd.notna(sma20_val) and pd.notna(sma50_val) else False
    )
    price_above_sma50 = bool(latest > sma50_val) if pd.notna(sma50_val) else False
    sma50_slope_positive = (
        bool(sma50_val > sma50_prev) if pd.notna(sma50_val) and pd.notna(sma50_prev) else False
    )

    score = int(sma20_above_sma50) + int(price_above_sma50) + int(sma50_slope_positive)

    return {
        "trend_score": score,
        "sma20_above_sma50": sma20_above_sma50,
        "price_above_sma50": price_above_sma50,
        "sma50_slope_positive": sma50_slope_positive,
        "falling_knife_warning": score < 2,
    }


_DATA_DIR = Path(__file__).resolve().parents[3] / "data"
_MODEL_DIR = _DATA_DIR / "ml_models"
_DEFAULT_PATH = _MODEL_DIR / "quantile_model.joblib"

_MODEL_VERSION = "1.0"

# Quantile levels covering tails, body, and median
DEFAULT_QUANTILES = [0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95]


def _create_lgbm_quantile_params(alpha: float) -> dict[str, Any]:
    """LightGBM params for quantile regression at a given alpha."""
    return {
        "objective": "quantile",
        "alpha": alpha,
        "n_estimators": 500,
        "learning_rate": 0.01,
        "max_depth": 5,
        "num_leaves": 20,
        "min_child_samples": 30,
        "reg_alpha": 0.5,
        "reg_lambda": 1.0,
        "colsample_bytree": 0.8,
        "subsample": 0.8,
        "verbose": -1,
        "n_jobs": -1,
    }


class QuantileModel:
    """Trains and stores LightGBM quantile regressors for return distribution."""

    def __init__(self, quantiles: list[float] | None = None) -> None:
        self.quantiles = quantiles or DEFAULT_QUANTILES
        self.models: dict[float, Any] = {}
        self.preprocessor: FeaturePreprocessor | None = None
        self.feature_names: list[str] | None = None
        self.metadata: dict[str, Any] = {}
        self.cv_metrics: dict[str, Any] = {}

    def train(
        self,
        features_df: pd.DataFrame,
        returns: pd.Series,
        dates: pd.Series,
        n_splits: int = 5,
        horizon: int = 10,
        decay: int = 365,
    ) -> dict[str, Any]:
        """Train quantile regressors with purged time-series CV.

        Args:
            features_df: Feature matrix (same as classifier).
            returns: Continuous forward returns (%).
            dates: Date index for time-series ordering.
            n_splits: Number of CV folds.
            horizon: Forward label horizon (for purging gap).
            decay: Sample weight half-life in days.

        Returns:
            Dict with training results and CV metrics.
        """
        import lightgbm as lgb

        from advisor.ml.models import _compute_sample_weights, _purged_time_series_split

        self.feature_names = list(features_df.columns)
        self.preprocessor = FeaturePreprocessor(scaling="none")

        # Purged CV for evaluation
        splits = _purged_time_series_split(dates, n_splits=n_splits, horizon=horizon)
        cv_pinball: dict[float, list[float]] = {q: [] for q in self.quantiles}
        cv_coverage: dict[str, list[float]] = {
            "50_pct": [],  # % returns in 25th-75th
            "80_pct": [],  # % returns in 10th-90th
            "90_pct": [],  # % returns in 5th-95th
        }

        for train_idx, test_idx in splits:
            X_train = features_df.iloc[train_idx]
            y_train = returns.iloc[train_idx]
            X_test = features_df.iloc[test_idx]
            y_test = returns.iloc[test_idx]

            pp = FeaturePreprocessor(scaling="none")
            X_train_t = pp.fit_transform(X_train)
            X_test_t = pp.transform(X_test)

            sample_weight = _compute_sample_weights(len(y_train), decay=decay)

            # Carve early-stopping validation (last 15%)
            val_size = max(5, int(len(y_train) * 0.15))
            X_es = X_train_t[-val_size:]
            y_es = y_train.iloc[-val_size:]
            X_fit = X_train_t[:-val_size]
            y_fit = y_train.iloc[:-val_size]
            sw_fit = sample_weight[:-val_size]

            fold_preds = {}
            for q in self.quantiles:
                model = lgb.LGBMRegressor(**_create_lgbm_quantile_params(q))

                X_fit_df = pd.DataFrame(X_fit, columns=self.feature_names)
                X_es_df = pd.DataFrame(X_es, columns=self.feature_names)

                model.fit(
                    X_fit_df,
                    y_fit,
                    sample_weight=sw_fit,
                    eval_set=[(X_es_df, y_es)],
                    callbacks=[lgb.early_stopping(50, verbose=False, min_delta=0.001)],
                )

                if model.best_iteration_ < 100:
                    model.set_params(n_estimators=max(200, model.n_estimators))
                    model.fit(X_fit_df, y_fit, sample_weight=sw_fit)

                X_test_df = pd.DataFrame(X_test_t, columns=self.feature_names)
                preds = model.predict(X_test_df)
                fold_preds[q] = preds

                # Pinball loss
                residual = y_test.values - preds
                loss = np.mean(np.where(residual >= 0, q * residual, (q - 1) * residual))
                cv_pinball[q].append(float(loss))

            # Coverage: what fraction of actuals fall within predicted intervals
            y_vals = y_test.values
            if 0.25 in fold_preds and 0.75 in fold_preds:
                in_50 = np.mean((y_vals >= fold_preds[0.25]) & (y_vals <= fold_preds[0.75]))
                cv_coverage["50_pct"].append(float(in_50))
            if 0.10 in fold_preds and 0.90 in fold_preds:
                in_80 = np.mean((y_vals >= fold_preds[0.10]) & (y_vals <= fold_preds[0.90]))
                cv_coverage["80_pct"].append(float(in_80))
            if 0.05 in fold_preds and 0.95 in fold_preds:
                in_90 = np.mean((y_vals >= fold_preds[0.05]) & (y_vals <= fold_preds[0.95]))
                cv_coverage["90_pct"].append(float(in_90))

        # Train final models on all data
        X_all = self.preprocessor.fit_transform(features_df)
        sample_weight_all = _compute_sample_weights(len(returns), decay=decay)

        for q in self.quantiles:
            model = lgb.LGBMRegressor(**_create_lgbm_quantile_params(q))
            X_all_df = pd.DataFrame(X_all, columns=self.feature_names)
            model.fit(X_all_df, returns, sample_weight=sample_weight_all)
            self.models[q] = model

        # Aggregate CV metrics
        self.cv_metrics = {
            "pinball_loss": {
                f"q{int(q*100):02d}": round(np.mean(losses), 4) for q, losses in cv_pinball.items()
            },
            "coverage": {
                k: round(np.mean(v) * 100, 1) if v else 0.0 for k, v in cv_coverage.items()
            },
        }

        self.metadata = {
            "n_samples": len(features_df),
            "n_features": len(self.feature_names),
            "n_splits": n_splits,
            "horizon": horizon,
            "decay": decay,
            "quantiles": self.quantiles,
            "trained_at": datetime.now().isoformat(),
            "version": _MODEL_VERSION,
        }

        return {
            "trained": True,
            "cv_metrics": self.cv_metrics,
            "metadata": self.metadata,
        }

    def predict_quantiles(self, features: pd.DataFrame) -> dict[float, np.ndarray]:
        """Predict return quantiles for feature rows.

        Returns:
            Dict mapping quantile level -> predicted values array.
        """
        if not self.models or self.preprocessor is None:
            raise RuntimeError("Model not trained or loaded.")

        if self.feature_names and isinstance(features, pd.DataFrame):
            features = features.reindex(columns=self.feature_names, fill_value=0.0)

        X = self.preprocessor.transform(features)
        X_df = pd.DataFrame(X, columns=self.feature_names)

        return {q: model.predict(X_df) for q, model in self.models.items()}

    def predict_distribution(
        self,
        features: pd.DataFrame,
        ohlcv_df: pd.DataFrame | None = None,
    ) -> dict[str, Any]:
        """Predict full return distribution summary for a single row.

        Args:
            features: Feature matrix (single row expected).
            ohlcv_df: Optional raw OHLCV data for trend-quality scoring.
                When provided, the output includes ``trend_score``,
                ``falling_knife_warning``, and ``risk_adjusted_score``.

        Returns dict with quantiles, VaR, CVaR, upside ratio, skew direction,
        and (when *ohlcv_df* is given) trend/risk-adjusted metrics.
        """
        quantile_preds = self.predict_quantiles(features)

        # Take first row
        q_values = {q: float(preds[0]) for q, preds in quantile_preds.items()}
        result = self._summarize_quantiles(q_values)

        # ── Trend filter & risk-adjusted score ───────────────────────
        if ohlcv_df is not None and len(ohlcv_df) >= 50:
            trend = compute_trend_score(ohlcv_df)
            result["trend_score"] = trend["trend_score"]
            result["sma20_above_sma50"] = trend["sma20_above_sma50"]
            result["price_above_sma50"] = trend["price_above_sma50"]
            result["sma50_slope_positive"] = trend["sma50_slope_positive"]
            result["falling_knife_warning"] = trend["falling_knife_warning"]

            # Risk-adjusted score: penalize wide downside tail when trend is broken
            var10_abs = abs(result["var_10"])
            penalty = 0.5 * var10_abs if trend["trend_score"] < 2 else 0.0
            result["risk_adjusted_score"] = round(result["median_return"] - penalty, 2)
        else:
            result["trend_score"] = None
            result["falling_knife_warning"] = None
            result["risk_adjusted_score"] = round(result["median_return"], 2)

        return result

    def predict_distribution_batch(
        self,
        features: pd.DataFrame,
        ohlcv_map: dict[str, pd.DataFrame] | None = None,
        tickers: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Predict distribution for multiple rows.

        Args:
            features: Feature matrix with one row per ticker.
            ohlcv_map: Optional mapping of ticker -> OHLCV DataFrame for trend scoring.
            tickers: Ticker list aligned with feature rows (required when *ohlcv_map* is given).
        """
        quantile_preds = self.predict_quantiles(features)
        results = []
        for i in range(len(features)):
            q_values = {q: float(preds[i]) for q, preds in quantile_preds.items()}
            summary = self._summarize_quantiles(q_values)

            # Attach trend/risk-adjusted score if OHLCV is available
            ticker = tickers[i] if tickers else None
            ohlcv = ohlcv_map.get(ticker) if ohlcv_map and ticker else None
            if ohlcv is not None and len(ohlcv) >= 50:
                trend = compute_trend_score(ohlcv)
                summary["trend_score"] = trend["trend_score"]
                summary["sma20_above_sma50"] = trend["sma20_above_sma50"]
                summary["price_above_sma50"] = trend["price_above_sma50"]
                summary["sma50_slope_positive"] = trend["sma50_slope_positive"]
                summary["falling_knife_warning"] = trend["falling_knife_warning"]

                var10_abs = abs(summary["var_10"])
                penalty = 0.5 * var10_abs if trend["trend_score"] < 2 else 0.0
                summary["risk_adjusted_score"] = round(summary["median_return"] - penalty, 2)
            else:
                summary["trend_score"] = None
                summary["falling_knife_warning"] = None
                summary["risk_adjusted_score"] = round(summary["median_return"], 2)

            results.append(summary)
        return results

    def _summarize_quantiles(self, q_values: dict[float, float]) -> dict[str, Any]:
        """Summarize a single set of quantile predictions."""
        result: dict[str, Any] = {"quantiles": {}}
        for q in sorted(q_values.keys()):
            result["quantiles"][q] = round(q_values[q], 2)

        median = q_values.get(0.50, 0.0)
        q05 = q_values.get(0.05, 0.0)
        q10 = q_values.get(0.10, 0.0)
        q25 = q_values.get(0.25, 0.0)
        q75 = q_values.get(0.75, 0.0)
        q90 = q_values.get(0.90, 0.0)
        q95 = q_values.get(0.95, 0.0)

        result["median_return"] = round(median, 2)
        result["expected_range"] = [round(q10, 2), round(q90, 2)]
        result["var_5"] = round(q05, 2)
        result["var_10"] = round(q10, 2)
        result["cvar_5"] = round(min(q05 * 1.2, q05), 2)

        upside = q90 - median
        downside = median - q10
        result["upside_ratio"] = (
            round(upside / downside, 2) if downside > 0 else (float("inf") if upside > 0 else 1.0)
        )

        upper_spread = q75 - median
        lower_spread = median - q25
        if upper_spread > lower_spread * 1.2:
            result["distribution_skew"] = "right"
        elif lower_spread > upper_spread * 1.2:
            result["distribution_skew"] = "left"
        else:
            result["distribution_skew"] = "symmetric"

        body_spread = q75 - q25
        tail_spread = q95 - q05
        result["tail_ratio"] = round(tail_spread / body_spread, 2) if body_spread > 0 else 0.0

        return result

    def get_feature_importance(self) -> dict[str, float]:
        """Aggregate feature importance across quantile models (median model weighted highest)."""
        if not self.models or self.feature_names is None:
            return {}

        # Weight median model most, tails less
        weights = {0.05: 0.5, 0.10: 0.7, 0.25: 0.9, 0.50: 1.0, 0.75: 0.9, 0.90: 0.7, 0.95: 0.5}
        total_imp = np.zeros(len(self.feature_names))
        total_weight = 0.0

        for q, model in self.models.items():
            if hasattr(model, "feature_importances_"):
                w = weights.get(q, 0.5)
                total_imp += model.feature_importances_ * w
                total_weight += w

        if total_weight > 0:
            total_imp /= total_weight

        pairs = sorted(
            zip(self.feature_names, total_imp),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        return {name: round(float(imp), 4) for name, imp in pairs}

    def save(self, path: Path | None = None) -> Path:
        """Persist quantile models to disk."""
        path = path or _DEFAULT_PATH
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "models": self.models,
            "preprocessor": self.preprocessor.to_dict() if self.preprocessor else None,
            "feature_names": self.feature_names,
            "cv_metrics": self.cv_metrics,
            "metadata": self.metadata,
            "quantiles": self.quantiles,
            "version": _MODEL_VERSION,
        }
        joblib.dump(payload, path)
        logger.info(f"Quantile model saved to {path}")
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "QuantileModel":
        """Load a saved quantile model."""
        path = path or _DEFAULT_PATH
        if not path.exists():
            raise FileNotFoundError(f"No quantile model found at {path}")

        payload = joblib.load(path)
        qm = cls(quantiles=payload.get("quantiles", DEFAULT_QUANTILES))
        qm.models = payload["models"]
        qm.feature_names = payload["feature_names"]
        qm.cv_metrics = payload.get("cv_metrics", {})
        qm.metadata = payload.get("metadata", {})

        pp_data = payload.get("preprocessor")
        if pp_data:
            qm.preprocessor = FeaturePreprocessor.from_dict(pp_data)

        return qm

    @staticmethod
    def model_exists(path: Path | None = None) -> bool:
        """Check if a saved quantile model exists."""
        return (path or _DEFAULT_PATH).exists()
