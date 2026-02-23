"""ML model training, evaluation, and persistence."""

from __future__ import annotations

import logging
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)

from advisor.ml.preprocessing import FeaturePreprocessor

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parents[3] / "data"
_MODEL_DIR = _DATA_DIR / "ml_models"
_DEFAULT_MODEL_PATH = _MODEL_DIR / "current_model.joblib"

_MODEL_VERSION = "2.0"


class ModelType(StrEnum):
    LOGISTIC = "logistic"
    LIGHTGBM = "lightgbm"
    ENSEMBLE = "ensemble"


def _purged_time_series_split(
    dates: pd.Series,
    n_splits: int = 5,
    horizon: int = 10,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Purged expanding-window time-series split.

    Prevents label-leakage by:
    - Gap: horizon + 5 bars between train end and test start
    - Purge: remove training samples whose forward window overlaps test start
    - Embargo: exclude bars after test end from future training folds
    """
    n = len(dates)
    gap_bars = horizon + 5
    fold_size = n // (n_splits + 1)
    splits = []

    for i in range(1, n_splits + 1):
        train_end = fold_size * i
        test_start = train_end + gap_bars
        test_end = min(train_end + fold_size + gap_bars, n)

        if test_start >= n or test_end <= test_start:
            continue

        # Purge: remove training indices whose label window overlaps test
        purge_start = max(0, test_start - horizon)
        train_idx = np.arange(0, min(train_end, purge_start))

        # Embargo: test fold excludes first `horizon` bars after test_end
        # (prevents contamination in future folds — handled implicitly by gap)
        test_idx = np.arange(test_start, test_end)

        if len(train_idx) < 20 or len(test_idx) < 5:
            continue

        splits.append((train_idx, test_idx))

    return splits


def _compute_sample_weights(n_samples: int, decay: int = 365) -> np.ndarray:
    """Exponential time-decay sample weights (recency bias).

    Args:
        n_samples: Number of training samples.
        decay: Half-life in trading days. 0 = uniform weights.

    Returns:
        Weight array of shape (n_samples,).
    """
    if decay <= 0:
        return np.ones(n_samples)
    decay_rate = np.log(2) / decay
    weights = np.exp(-decay_rate * (n_samples - np.arange(n_samples)))
    return weights


def _create_lgbm_params() -> dict[str, Any]:
    """Regularized LightGBM hyperparameters."""
    return {
        "n_estimators": 500,
        "learning_rate": 0.01,
        "max_depth": 5,
        "num_leaves": 20,
        "min_child_samples": 30,
        "reg_alpha": 0.5,
        "reg_lambda": 1.0,
        "colsample_bytree": 0.8,
        "subsample": 0.8,
        "class_weight": "balanced",
        "verbose": -1,
        "n_jobs": -1,
    }


class MLModelTrainer:
    """Train, evaluate, and persist ML models for signal prediction."""

    def __init__(self, model_type: ModelType = ModelType.LIGHTGBM) -> None:
        self.model_type = model_type
        self.model = None
        self.preprocessor: FeaturePreprocessor | None = None
        self.feature_names: list[str] | None = None
        self.metrics: dict[str, float] = {}
        self.metadata: dict[str, Any] = {}

    def train(
        self,
        features_df: pd.DataFrame,
        labels: pd.Series,
        dates: pd.Series,
        n_splits: int = 5,
        horizon: int = 10,
        decay: int = 365,
    ) -> dict[str, Any]:
        """Train model with purged time-series cross-validation.

        Args:
            features_df: Feature matrix (rows = samples, cols = features).
            labels: Binary target (1 = win, 0 = loss).
            dates: Date index for time-series ordering.
            n_splits: Number of CV folds.
            horizon: Forward label horizon (used for purging gap).
            decay: Sample weight half-life in days (0 = uniform).

        Returns:
            Dict with training results and CV metrics.
        """
        self.feature_names = list(features_df.columns)

        # Choose scaling based on model type
        scaling = "robust" if self.model_type == ModelType.LOGISTIC else "none"
        self.preprocessor = FeaturePreprocessor(scaling=scaling)

        # Purged time-series CV
        splits = _purged_time_series_split(
            dates,
            n_splits=n_splits,
            horizon=horizon,
        )
        cv_metrics = []

        for train_idx, test_idx in splits:
            X_train = features_df.iloc[train_idx]
            y_train = labels.iloc[train_idx]
            X_test = features_df.iloc[test_idx]
            y_test = labels.iloc[test_idx]

            pp = FeaturePreprocessor(scaling=scaling)
            X_train_t = pp.fit_transform(X_train)
            X_test_t = pp.transform(X_test)

            sample_weight = _compute_sample_weights(len(y_train), decay=decay)

            model = self._create_model()
            self._fit_model(
                model,
                X_train_t,
                y_train,
                sample_weight=sample_weight,
                X_val=X_test_t,
                y_val=y_test,
            )

            fold_metrics = self._compute_metrics(model, X_test_t, y_test)
            cv_metrics.append(fold_metrics)

        # Train final model on all data
        X_all = self.preprocessor.fit_transform(features_df)
        sample_weight_all = _compute_sample_weights(len(labels), decay=decay)
        self.model = self._create_model()
        self._fit_model(
            self.model,
            X_all,
            labels,
            sample_weight=sample_weight_all,
        )

        # Aggregate CV metrics
        avg_metrics = {}
        for key in cv_metrics[0]:
            values = [m[key] for m in cv_metrics]
            avg_metrics[f"cv_{key}_mean"] = round(np.mean(values), 4)
            avg_metrics[f"cv_{key}_std"] = round(np.std(values), 4)

        self.metrics = avg_metrics
        self.metadata = {
            "model_type": str(self.model_type),
            "n_samples": len(features_df),
            "n_features": len(self.feature_names),
            "n_splits": n_splits,
            "win_rate": round(float(labels.mean()) * 100, 1),
            "trained_at": datetime.now().isoformat(),
            "version": _MODEL_VERSION,
        }

        return {
            "trained": True,
            "cv_metrics": avg_metrics,
            "metadata": self.metadata,
        }

    def _create_model(self) -> Any:
        """Instantiate the appropriate model."""
        if self.model_type == ModelType.LOGISTIC:
            return LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
        elif self.model_type == ModelType.LIGHTGBM:
            import lightgbm as lgb

            return lgb.LGBMClassifier(**_create_lgbm_params())
        elif self.model_type == ModelType.ENSEMBLE:
            import lightgbm as lgb
            from sklearn.ensemble import StackingClassifier

            lr = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
            lgbm = lgb.LGBMClassifier(**_create_lgbm_params())
            return StackingClassifier(
                estimators=[("lr", lr), ("lgbm", lgbm)],
                final_estimator=LogisticRegression(max_iter=1000),
                cv=3,
                passthrough=False,
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")

    def _fit_model(
        self,
        model: Any,
        X_train: np.ndarray,
        y_train: pd.Series,
        sample_weight: np.ndarray | None = None,
        X_val: np.ndarray | None = None,
        y_val: pd.Series | None = None,
    ) -> None:
        """Fit model with early stopping for LightGBM-based models."""
        import lightgbm as lgb

        is_lgbm = isinstance(model, lgb.LGBMClassifier)

        # Wrap numpy arrays as DataFrames to preserve feature names for LightGBM
        feature_names = self.feature_names
        if is_lgbm and feature_names:
            X_train = pd.DataFrame(X_train, columns=feature_names)
            if X_val is not None:
                X_val = pd.DataFrame(X_val, columns=feature_names)

        if is_lgbm and X_val is not None and y_val is not None:
            model.fit(
                X_train,
                y_train,
                sample_weight=sample_weight,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    lgb.early_stopping(50, verbose=False, min_delta=0.001),
                ],
            )
            # Guard: if early stopping fired too early, retrain without it
            # to avoid collapsed probability distributions
            if model.best_iteration_ < 100:
                model.set_params(n_estimators=max(200, model.n_estimators))
                model.fit(X_train, y_train, sample_weight=sample_weight)
        elif is_lgbm:
            model.fit(X_train, y_train, sample_weight=sample_weight)
        elif sample_weight is not None and hasattr(model, "fit"):
            # StackingClassifier and LogisticRegression accept sample_weight
            try:
                model.fit(X_train, y_train, sample_weight=sample_weight)
            except TypeError:
                model.fit(X_train, y_train)
        else:
            model.fit(X_train, y_train)

    def _compute_metrics(self, model: Any, X: np.ndarray, y: pd.Series) -> dict[str, float]:
        """Compute evaluation metrics for a trained model."""
        y_pred = model.predict(X)
        y_prob = model.predict_proba(X)[:, 1]
        y_true = y.values

        metrics = {
            "accuracy": accuracy_score(y_true, y_pred),
            "precision": precision_score(y_true, y_pred, zero_division=0),
            "recall": recall_score(y_true, y_pred, zero_division=0),
            "f1": f1_score(y_true, y_pred, zero_division=0),
            "brier": brier_score_loss(y_true, y_prob),
        }

        # AUC requires both classes present
        if len(np.unique(y_true)) > 1:
            metrics["auc"] = roc_auc_score(y_true, y_prob)
        else:
            metrics["auc"] = 0.0

        return metrics

    def precision_at_thresholds(
        self,
        features_df: pd.DataFrame,
        labels: pd.Series,
        dates: pd.Series,
        thresholds: list[float] | None = None,
        n_splits: int = 5,
        horizon: int = 10,
        decay: int = 365,
    ) -> list[dict[str, Any]]:
        """Compute precision/recall at various probability thresholds using CV.

        Collects out-of-sample predictions from each CV fold, pools them,
        then computes precision, recall, and signal count at each threshold.
        This answers: "when the model is very confident, how often is it right?"
        """
        if thresholds is None:
            thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]

        scaling = "robust" if self.model_type == ModelType.LOGISTIC else "none"
        splits = _purged_time_series_split(dates, n_splits=n_splits, horizon=horizon)

        # Pool OOS predictions across all folds
        all_probs = []
        all_true = []

        for train_idx, test_idx in splits:
            X_train = features_df.iloc[train_idx]
            y_train = labels.iloc[train_idx]
            X_test = features_df.iloc[test_idx]
            y_test = labels.iloc[test_idx]

            pp = FeaturePreprocessor(scaling=scaling)
            X_train_t = pp.fit_transform(X_train)
            X_test_t = pp.transform(X_test)

            sample_weight = _compute_sample_weights(len(y_train), decay=decay)

            model = self._create_model()
            self._fit_model(
                model,
                X_train_t,
                y_train,
                sample_weight=sample_weight,
                X_val=X_test_t,
                y_val=y_test,
            )

            # Wrap as DataFrame for LightGBM feature name consistency
            import lightgbm as lgb

            if isinstance(model, lgb.LGBMClassifier) and self.feature_names:
                X_pred = pd.DataFrame(X_test_t, columns=self.feature_names)
            else:
                X_pred = X_test_t
            probs = model.predict_proba(X_pred)[:, 1]
            all_probs.extend(probs.tolist())
            all_true.extend(y_test.values.tolist())

        all_probs = np.array(all_probs)
        all_true = np.array(all_true)
        total_oos = len(all_true)
        base_rate = float(all_true.mean()) if total_oos > 0 else 0.0

        rows = []
        for t in thresholds:
            mask = all_probs >= t
            n_signals = int(mask.sum())
            if n_signals == 0:
                rows.append(
                    {
                        "threshold": t,
                        "n_signals": 0,
                        "pct_universe": 0.0,
                        "precision": 0.0,
                        "recall": 0.0,
                        "lift": 0.0,
                    }
                )
                continue

            true_pos = int(all_true[mask].sum())
            total_pos = int(all_true.sum())
            prec = true_pos / n_signals
            rec = true_pos / total_pos if total_pos > 0 else 0.0
            lift = prec / base_rate if base_rate > 0 else 0.0

            rows.append(
                {
                    "threshold": t,
                    "n_signals": n_signals,
                    "pct_universe": round(n_signals / total_oos * 100, 1),
                    "precision": round(prec, 4),
                    "recall": round(rec, 4),
                    "lift": round(lift, 2),
                }
            )

        return rows

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
        """Evaluate the trained model on held-out data."""
        if self.model is None or self.preprocessor is None:
            raise RuntimeError("Model not trained. Call train() first.")

        X_t = self.preprocessor.transform(X_test)
        return self._compute_metrics(self.model, X_t, y_test)

    def walk_forward_evaluate(
        self,
        features_df: pd.DataFrame,
        labels: pd.Series,
        dates: pd.Series,
        n_windows: int = 5,
        train_pct: float = 0.7,
        horizon: int = 10,
        decay: int = 365,
    ) -> dict[str, Any]:
        """Walk-forward validation: rolling IS/OOS windows.

        Returns in-sample vs out-of-sample metrics to detect overfitting.
        Uses purged gap and sample weighting consistent with training.
        """
        n = len(features_df)
        window_size = n // n_windows
        scaling = "robust" if self.model_type == ModelType.LOGISTIC else "none"
        gap = horizon + 5

        is_metrics_list = []
        oos_metrics_list = []

        for i in range(n_windows):
            start = i * window_size
            end = min(start + window_size, n)
            if end - start < 20:
                continue

            split = int((end - start) * train_pct) + start
            train_slice = slice(start, split)
            test_slice = slice(split + gap, end)

            X_train = features_df.iloc[train_slice]
            y_train = labels.iloc[train_slice]
            X_test = features_df.iloc[test_slice]
            y_test = labels.iloc[test_slice]

            if len(X_test) < 5 or len(np.unique(y_train)) < 2:
                continue

            pp = FeaturePreprocessor(scaling=scaling)
            X_train_t = pp.fit_transform(X_train)
            X_test_t = pp.transform(X_test)

            sample_weight = _compute_sample_weights(len(y_train), decay=decay)

            model = self._create_model()
            self._fit_model(
                model,
                X_train_t,
                y_train,
                sample_weight=sample_weight,
                X_val=X_test_t,
                y_val=y_test,
            )

            is_m = self._compute_metrics(model, X_train_t, y_train)
            oos_m = self._compute_metrics(model, X_test_t, y_test)
            is_metrics_list.append(is_m)
            oos_metrics_list.append(oos_m)

        if not oos_metrics_list:
            return {"error": "Not enough data for walk-forward evaluation"}

        # Aggregate
        def _avg(metrics_list: list[dict], key: str) -> float:
            vals = [m[key] for m in metrics_list if key in m]
            return round(np.mean(vals), 4) if vals else 0.0

        is_auc = _avg(is_metrics_list, "auc")
        oos_auc = _avg(oos_metrics_list, "auc")

        return {
            "n_windows": len(oos_metrics_list),
            "is_auc": is_auc,
            "oos_auc": oos_auc,
            "auc_gap": round(is_auc - oos_auc, 4),
            "is_accuracy": _avg(is_metrics_list, "accuracy"),
            "oos_accuracy": _avg(oos_metrics_list, "accuracy"),
            "is_f1": _avg(is_metrics_list, "f1"),
            "oos_f1": _avg(oos_metrics_list, "f1"),
        }

    def get_feature_importance(self) -> dict[str, float]:
        """Get feature importance from the trained model."""
        if self.model is None or self.feature_names is None:
            return {}

        model = self.model

        # Handle StackingClassifier / VotingClassifier
        if hasattr(model, "estimators_") and hasattr(model, "named_estimators_"):
            if "lgbm" in model.named_estimators_:
                model = model.named_estimators_["lgbm"]
            elif "lr" in model.named_estimators_:
                model = model.named_estimators_["lr"]

        if hasattr(model, "feature_importances_"):
            importances = model.feature_importances_
        elif hasattr(model, "coef_"):
            importances = np.abs(model.coef_[0])
        else:
            return {}

        pairs = sorted(
            zip(self.feature_names, importances),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        return {name: round(float(imp), 4) for name, imp in pairs}

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        """Predict win probability for feature rows."""
        if self.model is None or self.preprocessor is None:
            raise RuntimeError("Model not trained or loaded.")
        X = self.preprocessor.transform(features)
        return self.model.predict_proba(X)[:, 1]

    def save(self, path: Path | None = None) -> Path:
        """Persist model, preprocessor, and metadata to disk."""
        path = path or _DEFAULT_MODEL_PATH
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model": self.model,
            "preprocessor": self.preprocessor.to_dict() if self.preprocessor else None,
            "feature_names": self.feature_names,
            "metrics": self.metrics,
            "metadata": self.metadata,
            "version": _MODEL_VERSION,
        }
        joblib.dump(payload, path)
        logger.info(f"Model saved to {path}")
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "MLModelTrainer":
        """Load a saved model from disk."""
        path = path or _DEFAULT_MODEL_PATH
        if not path.exists():
            raise FileNotFoundError(f"No model found at {path}")

        payload = joblib.load(path)
        trainer = cls()
        trainer.model = payload["model"]
        trainer.feature_names = payload["feature_names"]
        trainer.metrics = payload.get("metrics", {})
        trainer.metadata = payload.get("metadata", {})

        pp_data = payload.get("preprocessor")
        if pp_data:
            trainer.preprocessor = FeaturePreprocessor.from_dict(pp_data)

        # Infer model type from metadata
        mt = payload.get("metadata", {}).get("model_type", "lightgbm")
        trainer.model_type = ModelType(mt)

        return trainer

    @staticmethod
    def model_exists(path: Path | None = None) -> bool:
        """Check if a saved model exists."""
        return (path or _DEFAULT_MODEL_PATH).exists()
