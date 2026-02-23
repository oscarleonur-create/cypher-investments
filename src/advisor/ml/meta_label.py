"""Meta-labeling — secondary model that predicts when the primary model is correct.

Implements Lopez de Prado's meta-labeling approach:
  Stage 1 (primary):  Existing LightGBM predicts direction (win/loss).
  Stage 2 (meta):     A secondary model predicts whether Stage 1 is
                       correct on each sample.  Meta-label = 1 if primary
                       was right, 0 if primary was wrong.

The meta-model's probability output answers: "given the primary model says
BUY, how likely is it to be correct?"  This decouples direction prediction
from trade filtering and dramatically improves precision at high thresholds.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from advisor.ml.models import (
    MLModelTrainer,
    ModelType,
    _compute_sample_weights,
    _purged_time_series_split,
)
from advisor.ml.preprocessing import FeaturePreprocessor

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parents[3] / "data"
_META_MODEL_PATH = _DATA_DIR / "ml_models" / "meta_model.joblib"


class MetaLabeler:
    """Train and apply a meta-labeling model on top of a primary classifier."""

    def __init__(self) -> None:
        self.meta_model: Any | None = None
        self.meta_preprocessor: FeaturePreprocessor | None = None
        self.meta_feature_names: list[str] | None = None
        self.metrics: dict[str, float] = {}

    def train(
        self,
        features_df: pd.DataFrame,
        labels: pd.Series,
        dates: pd.Series,
        primary_model_type: ModelType = ModelType.LIGHTGBM,
        n_splits: int = 5,
        horizon: int = 10,
        decay: int = 365,
    ) -> dict[str, Any]:
        """Train meta-labeling model using CV-generated primary predictions.

        For each CV fold:
          1. Train primary model on train split.
          2. Generate primary predictions on test split.
          3. Compute meta-labels: 1 if primary prediction was correct, 0 otherwise.
        Pool all (features + primary_prob, meta_label) pairs, then train
        the meta-model on the pooled data.
        """
        import lightgbm as lgb

        feature_names = list(features_df.columns)
        scaling = "robust" if primary_model_type == ModelType.LOGISTIC else "none"
        splits = _purged_time_series_split(dates, n_splits=n_splits, horizon=horizon)

        # Collect meta-training data from each CV fold
        meta_features_list: list[np.ndarray] = []
        meta_labels_list: list[np.ndarray] = []

        for train_idx, test_idx in splits:
            X_train = features_df.iloc[train_idx]
            y_train = labels.iloc[train_idx]
            X_test = features_df.iloc[test_idx]
            y_test = labels.iloc[test_idx]

            pp = FeaturePreprocessor(scaling=scaling)
            X_train_t = pp.fit_transform(X_train)
            X_test_t = pp.transform(X_test)
            sw = _compute_sample_weights(len(y_train), decay=decay)

            # Train primary model
            primary = MLModelTrainer(model_type=primary_model_type)
            primary.feature_names = feature_names
            primary_model = primary._create_model()
            primary._fit_model(
                primary_model,
                X_train_t,
                y_train,
                sample_weight=sw,
                X_val=X_test_t,
                y_val=y_test,
            )

            # Generate primary predictions on OOS
            if isinstance(primary_model, lgb.LGBMClassifier) and feature_names:
                X_pred_df = pd.DataFrame(X_test_t, columns=feature_names)
            else:
                X_pred_df = X_test_t
            primary_probs = primary_model.predict_proba(X_pred_df)[:, 1]
            primary_preds = (primary_probs >= 0.5).astype(int)

            # Meta-labels: 1 if primary prediction was correct
            y_true = y_test.values
            meta_labels = (primary_preds == y_true).astype(int)

            # Meta-features: original features + primary model probability
            meta_feats = np.column_stack([X_test_t, primary_probs])
            meta_features_list.append(meta_feats)
            meta_labels_list.append(meta_labels)

        # Pool all meta-training data
        X_meta = np.vstack(meta_features_list)
        y_meta = np.concatenate(meta_labels_list)

        self.meta_feature_names = feature_names + ["primary_prob"]

        logger.info(
            f"Meta-labeling: {len(y_meta)} samples, " f"primary correct rate: {y_meta.mean():.1%}"
        )

        # Train meta-model (LightGBM with lighter regularization)
        self.meta_preprocessor = FeaturePreprocessor(scaling="none")
        X_meta_t = self.meta_preprocessor.fit_transform(
            pd.DataFrame(X_meta, columns=self.meta_feature_names)
        )

        self.meta_model = lgb.LGBMClassifier(
            n_estimators=300,
            learning_rate=0.02,
            max_depth=4,
            num_leaves=15,
            min_child_samples=30,
            reg_alpha=0.3,
            reg_lambda=0.5,
            colsample_bytree=0.8,
            subsample=0.8,
            class_weight="balanced",
            verbose=-1,
        )

        meta_sw = _compute_sample_weights(len(y_meta), decay=decay)
        X_meta_df = pd.DataFrame(X_meta_t, columns=self.meta_feature_names)
        self.meta_model.fit(X_meta_df, y_meta, sample_weight=meta_sw)

        # Evaluate with a simple internal split on the pooled data
        # (proper evaluation is done in precision_at_thresholds)
        meta_probs = self.meta_model.predict_proba(X_meta_df)[:, 1]
        self.metrics = {
            "meta_auc": round(float(roc_auc_score(y_meta, meta_probs)), 4),
            "meta_samples": len(y_meta),
            "primary_correct_rate": round(float(y_meta.mean()), 4),
        }

        return {
            "trained": True,
            "metrics": self.metrics,
        }

    def predict_meta_prob(
        self,
        features: np.ndarray | pd.DataFrame,
        primary_prob: np.ndarray,
    ) -> np.ndarray:
        """Predict meta-probability (P(primary is correct) for each sample)."""
        if self.meta_model is None or self.meta_preprocessor is None:
            raise RuntimeError("Meta-model not trained. Call train() first.")

        if isinstance(features, pd.DataFrame):
            features = features.values

        X_meta = np.column_stack([features, primary_prob])
        X_meta_df = pd.DataFrame(X_meta, columns=self.meta_feature_names)
        X_meta_t = self.meta_preprocessor.transform(X_meta_df)
        X_meta_t_df = pd.DataFrame(X_meta_t, columns=self.meta_feature_names)
        return self.meta_model.predict_proba(X_meta_t_df)[:, 1]

    def combined_score(
        self,
        primary_prob: np.ndarray,
        meta_prob: np.ndarray,
    ) -> np.ndarray:
        """Compute final conviction score: primary_prob * meta_prob.

        This product answers: "P(win) * P(model is right about this call)".
        High scores require both high directional conviction AND high
        meta-confidence that this is a good trade setup.
        """
        return primary_prob * meta_prob

    def precision_at_thresholds(
        self,
        features_df: pd.DataFrame,
        labels: pd.Series,
        dates: pd.Series,
        primary_model_type: ModelType = ModelType.LIGHTGBM,
        thresholds: list[float] | None = None,
        n_splits: int = 5,
        horizon: int = 10,
        decay: int = 365,
    ) -> dict[str, list[dict[str, Any]]]:
        """Compare precision curves: primary-only vs meta-labeled.

        Uses a nested CV approach: outer loop generates OOS primary+meta
        predictions, inner CV trains both models.
        """
        import lightgbm as lgb

        if thresholds is None:
            thresholds = [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]

        feature_names = list(features_df.columns)
        scaling = "robust" if primary_model_type == ModelType.LOGISTIC else "none"
        splits = _purged_time_series_split(dates, n_splits=n_splits, horizon=horizon)

        # Pool OOS predictions from each fold
        all_primary_probs: list[float] = []
        all_meta_probs: list[float] = []
        all_combined: list[float] = []
        all_true: list[int] = []

        for fold_i, (train_idx, test_idx) in enumerate(splits):
            X_train = features_df.iloc[train_idx]
            y_train = labels.iloc[train_idx]
            X_test = features_df.iloc[test_idx]
            y_test = labels.iloc[test_idx]

            pp = FeaturePreprocessor(scaling=scaling)
            X_train_t = pp.fit_transform(X_train)
            X_test_t = pp.transform(X_test)
            sw = _compute_sample_weights(len(y_train), decay=decay)

            # --- Train primary model ---
            primary = MLModelTrainer(model_type=primary_model_type)
            primary.feature_names = feature_names
            primary_model = primary._create_model()
            primary._fit_model(
                primary_model,
                X_train_t,
                y_train,
                sample_weight=sw,
                X_val=X_test_t,
                y_val=y_test,
            )

            if isinstance(primary_model, lgb.LGBMClassifier) and feature_names:
                X_train_pred = pd.DataFrame(X_train_t, columns=feature_names)
                X_test_pred = pd.DataFrame(X_test_t, columns=feature_names)
            else:
                X_train_pred = X_train_t
                X_test_pred = X_test_t

            # Primary predictions on train (for meta-training) and test
            primary_probs_train = primary_model.predict_proba(X_train_pred)[:, 1]
            primary_probs_test = primary_model.predict_proba(X_test_pred)[:, 1]

            # --- Train meta-model on this fold's training data ---
            primary_preds_train = (primary_probs_train >= 0.5).astype(int)
            meta_labels_train = (primary_preds_train == y_train.values).astype(int)

            meta_feat_names = feature_names + ["primary_prob"]
            X_meta_train = np.column_stack([X_train_t, primary_probs_train])
            X_meta_train_df = pd.DataFrame(X_meta_train, columns=meta_feat_names)

            meta_model = lgb.LGBMClassifier(
                n_estimators=300,
                learning_rate=0.02,
                max_depth=4,
                num_leaves=15,
                min_child_samples=30,
                reg_alpha=0.3,
                reg_lambda=0.5,
                colsample_bytree=0.8,
                subsample=0.8,
                class_weight="balanced",
                verbose=-1,
            )
            meta_sw = _compute_sample_weights(len(meta_labels_train), decay=decay)
            meta_model.fit(X_meta_train_df, meta_labels_train, sample_weight=meta_sw)

            # --- Generate OOS meta predictions ---
            X_meta_test = np.column_stack([X_test_t, primary_probs_test])
            X_meta_test_df = pd.DataFrame(X_meta_test, columns=meta_feat_names)
            meta_probs_test = meta_model.predict_proba(X_meta_test_df)[:, 1]

            combined = primary_probs_test * meta_probs_test

            all_primary_probs.extend(primary_probs_test.tolist())
            all_meta_probs.extend(meta_probs_test.tolist())
            all_combined.extend(combined.tolist())
            all_true.extend(y_test.values.tolist())

        all_primary = np.array(all_primary_probs)
        all_meta = np.array(all_meta_probs)
        all_comb = np.array(all_combined)
        all_y = np.array(all_true)
        base_rate = float(all_y.mean())
        total = len(all_y)

        def _compute_curve(probs: np.ndarray, name: str) -> list[dict]:
            rows = []
            for t in thresholds:
                mask = probs >= t
                n = int(mask.sum())
                if n == 0:
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
                wins = int(all_y[mask].sum())
                total_pos = int(all_y.sum())
                prec = wins / n
                rec = wins / total_pos if total_pos > 0 else 0.0
                rows.append(
                    {
                        "threshold": t,
                        "n_signals": n,
                        "pct_universe": round(n / total * 100, 1),
                        "precision": round(prec, 4),
                        "recall": round(rec, 4),
                        "lift": round(prec / base_rate, 2) if base_rate > 0 else 0.0,
                    }
                )
            return rows

        return {
            "primary": _compute_curve(all_primary, "primary"),
            "meta": _compute_curve(all_meta, "meta"),
            "combined": _compute_curve(all_comb, "combined"),
            "base_rate": round(base_rate, 4),
            "total_oos": total,
        }

    def save(self, path: Path | None = None) -> Path:
        """Persist meta-model to disk."""
        path = path or _META_MODEL_PATH
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "meta_model": self.meta_model,
            "meta_preprocessor": (
                self.meta_preprocessor.to_dict() if self.meta_preprocessor else None
            ),
            "meta_feature_names": self.meta_feature_names,
            "metrics": self.metrics,
        }
        joblib.dump(payload, path)
        logger.info(f"Meta-model saved to {path}")
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "MetaLabeler":
        """Load a saved meta-model."""
        path = path or _META_MODEL_PATH
        if not path.exists():
            raise FileNotFoundError(f"No meta-model at {path}")
        payload = joblib.load(path)
        ml = cls()
        ml.meta_model = payload["meta_model"]
        ml.meta_feature_names = payload["meta_feature_names"]
        ml.metrics = payload.get("metrics", {})
        pp_data = payload.get("meta_preprocessor")
        if pp_data:
            ml.meta_preprocessor = FeaturePreprocessor.from_dict(pp_data)
        return ml

    @staticmethod
    def model_exists(path: Path | None = None) -> bool:
        return (path or _META_MODEL_PATH).exists()
