"""Feature preprocessing — scaling and NaN imputation for ML models."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


class FeaturePreprocessor:
    """Handles feature scaling and NaN imputation.

    - RobustScaler for logistic regression (handles financial outliers well)
    - Passthrough (NaN imputation only) for tree-based models
    """

    def __init__(self, scaling: str = "robust") -> None:
        """Initialize preprocessor.

        Args:
            scaling: "robust" for RobustScaler (logistic), "none" for trees.
        """
        self.scaling = scaling
        self.medians_: dict[str, float] | None = None
        self.centers_: dict[str, float] | None = None
        self.scales_: dict[str, float] | None = None
        self.feature_names_: list[str] | None = None
        self._fitted = False

    def fit(self, X: pd.DataFrame) -> "FeaturePreprocessor":
        """Fit the preprocessor on training data."""
        self.feature_names_ = list(X.columns)
        # Use 0.0 as fallback when an entire column is NaN
        medians = X.median()
        self.medians_ = {col: (float(v) if pd.notna(v) else 0.0) for col, v in medians.items()}

        if self.scaling == "robust":
            q25 = X.quantile(0.25)
            q75 = X.quantile(0.75)
            iqr = q75 - q25
            self.centers_ = dict(self.medians_)
            self.scales_ = {
                col: float(iqr[col]) if pd.notna(iqr[col]) and iqr[col] > 0 else 1.0
                for col in X.columns
            }

        self._fitted = True
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        """Transform features: impute NaNs and optionally scale."""
        if not self._fitted:
            raise RuntimeError("Preprocessor not fitted. Call fit() first.")

        df = X.copy()

        # Ensure all expected columns exist
        for col in self.feature_names_:
            if col not in df.columns:
                df[col] = 0.0

        # Reorder to match training columns
        df = df[self.feature_names_]

        # Impute NaN with training medians
        for col in self.feature_names_:
            df[col] = df[col].fillna(self.medians_.get(col, 0.0))

        # Replace any remaining inf values and NaNs
        df = df.replace([np.inf, -np.inf], 0.0).fillna(0.0)

        if self.scaling == "robust" and self.centers_ is not None:
            for col in self.feature_names_:
                center = self.centers_[col]
                scale = self.scales_[col]
                df[col] = (df[col] - center) / scale

        # Final safety: ensure no NaN reaches the model
        return np.nan_to_num(df.values.astype(np.float64), nan=0.0, posinf=0.0, neginf=0.0)

    def fit_transform(self, X: pd.DataFrame) -> np.ndarray:
        """Fit and transform in one step."""
        self.fit(X)
        return self.transform(X)

    def to_dict(self) -> dict[str, Any]:
        """Serialize preprocessor params for persistence."""
        return {
            "scaling": self.scaling,
            "medians": self.medians_,
            "centers": self.centers_,
            "scales": self.scales_,
            "feature_names": self.feature_names_,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "FeaturePreprocessor":
        """Restore preprocessor from saved params."""
        pp = cls(scaling=data["scaling"])
        pp.medians_ = data["medians"]
        pp.centers_ = data["centers"]
        pp.scales_ = data["scales"]
        pp.feature_names_ = data["feature_names"]
        pp._fitted = True
        return pp
