"""ML training pipeline — end-to-end orchestration."""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd

from advisor.ml.features import FeatureEngine
from advisor.ml.models import MLModelTrainer, ModelType

logger = logging.getLogger(__name__)

# Broad multi-sector universe (~40 names)
_DEFAULT_SYMBOLS = [
    # Tech
    "AAPL",
    "MSFT",
    "GOOGL",
    "AMZN",
    "NVDA",
    "META",
    "TSLA",
    "ADBE",
    "CRM",
    "AMD",
    # Financials
    "JPM",
    "BAC",
    "GS",
    "V",
    "MA",
    # Healthcare
    "UNH",
    "JNJ",
    "LLY",
    "PFE",
    "ABBV",
    # Consumer
    "HD",
    "MCD",
    "NKE",
    "COST",
    "PG",
    # Energy / Industrials
    "XOM",
    "CVX",
    "CAT",
    "HON",
    "UPS",
    # Communication / Media
    "NFLX",
    "DIS",
    "CMCSA",
    # Diverse
    "BRK-B",
    "WMT",
    "KO",
    "PEP",
    "INTC",
    "CSCO",
    "T",
]

_DEFAULT_THRESHOLD = 3.0  # 3% return = "win" (lowered for shorter horizons)
_DEFAULT_HORIZON = 10  # 10-day forward return


class MLPipeline:
    """End-to-end ML training and evaluation pipeline."""

    def __init__(
        self,
        symbols: list[str] | None = None,
        lookback: str = "5y",
        threshold: float = _DEFAULT_THRESHOLD,
        horizon: int = _DEFAULT_HORIZON,
        label_mode: str = "barrier",
        decay: int = 365,
    ) -> None:
        self.symbols = [s.upper() for s in (symbols or _DEFAULT_SYMBOLS)]
        self.lookback = lookback
        self.threshold = threshold
        self.horizon = horizon
        self.label_mode = label_mode
        self.decay = decay
        self._engine = FeatureEngine()

    def build_training_data(
        self,
        cutoff_date: str | None = None,
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Build features + labels from historical OHLCV data.

        Args:
            cutoff_date: If set, only include data up to this date (YYYY-MM-DD)
                for strict OOS separation. The cutoff is stored in model metadata.

        Returns:
            (features_df, labels, dates) — aligned DataFrames/Series.
        """
        import yfinance as yf

        all_features = []
        all_labels = []
        all_dates = []

        for symbol in self.symbols:
            logger.info(f"Building features for {symbol}...")
            try:
                features = self._engine.compute_features_df(
                    symbol,
                    period=self.lookback,
                )
                if features.empty or len(features) < 60:
                    logger.warning(f"Skipping {symbol}: insufficient data")
                    continue

                # Download prices to compute forward returns
                df = yf.download(symbol, period=self.lookback, progress=False)
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)

                close = df["Close"]

                # Compute labels based on label_mode
                if self.label_mode == "barrier":
                    labels, common_idx = self._compute_triple_barrier_labels(
                        close,
                        features,
                    )
                else:
                    # Fixed threshold (legacy)
                    forward_ret = close.shift(-self.horizon) / close - 1
                    forward_ret = forward_ret * 100  # Convert to percentage
                    common_idx = features.index.intersection(forward_ret.dropna().index)
                    ret_aligned = forward_ret.loc[common_idx]
                    labels = (ret_aligned > self.threshold).astype(int)

                # Apply temporal cutoff if specified
                if cutoff_date:
                    cutoff_ts = pd.Timestamp(cutoff_date)
                    mask = common_idx <= cutoff_ts
                    common_idx = common_idx[mask]
                    labels = labels.loc[common_idx]

                if len(common_idx) < 30:
                    logger.warning(f"Skipping {symbol}: insufficient aligned data")
                    continue

                feat_aligned = features.loc[common_idx]

                all_features.append(feat_aligned)
                all_labels.append(labels)
                all_dates.append(pd.Series(common_idx, index=common_idx))

            except Exception as e:
                logger.error(f"Error building data for {symbol}: {e}")
                continue

        if not all_features:
            raise ValueError("No training data could be built from any symbol")

        # ── HMM Regime features ───────────────────────────────────────────
        # Compute regime features for the full date range across all symbols
        try:
            from advisor.ml.regime import RegimeDetector

            detector = RegimeDetector()
            if RegimeDetector.model_exists():
                detector = RegimeDetector.load()
                logger.info("Loaded existing HMM regime model")
            else:
                logger.info("Fitting HMM regime model...")
                detector.fit(lookback=self.lookback)
                detector.save()

            # Collect all dates across symbols
            all_dates_set = set()
            for feat in all_features:
                all_dates_set.update(feat.index)
            all_dates_idx = pd.DatetimeIndex(sorted(all_dates_set))

            regime_features = detector.compute_regime_features(all_dates_idx)

            # Merge regime features into each symbol's feature DataFrame
            for i, feat in enumerate(all_features):
                regime_slice = regime_features.reindex(feat.index, method="ffill")
                for col in regime_slice.columns:
                    feat[col] = regime_slice[col].fillna(0.0)
                all_features[i] = feat

            logger.info("Added %d HMM regime features", len(regime_features.columns))
        except Exception as e:
            logger.warning(f"HMM regime features unavailable: {e}")

        # ── Cross-sectional features ─────────────────────────────────────
        # Rank key features within universe on each date.  Each symbol's
        # feature df is indexed by date, so we tag with symbol, concat into
        # a panel, group by date, and compute percentile ranks.
        _CS_SOURCE_COLS = [
            "ret_5d",
            "ret_20d",
            "ret_60d",
            "volume_zscore",
            "realized_vol_20",
            "momentum_rank",
        ]

        all_symbols_tagged: list[pd.DataFrame] = []
        all_labels_tagged: list[pd.Series] = []
        all_dates_tagged: list[pd.Series] = []

        for feat, lab, dt in zip(all_features, all_labels, all_dates):
            tagged = feat.copy()
            tagged["__date__"] = feat.index
            all_symbols_tagged.append(tagged)
            all_labels_tagged.append(lab)
            all_dates_tagged.append(dt)

        panel = pd.concat(all_symbols_tagged)

        # Compute cross-sectional percentile ranks per date
        cs_cols_present = [c for c in _CS_SOURCE_COLS if c in panel.columns]
        if cs_cols_present:
            grouped = panel.groupby("__date__")
            for col in cs_cols_present:
                cs_name = f"{col}_cs_rank"
                panel[cs_name] = grouped[col].rank(pct=True)

        panel.drop(columns=["__date__"], inplace=True)

        features_df = panel
        labels_series = pd.concat(all_labels_tagged)
        dates_series = pd.concat(all_dates_tagged)

        # Sort by date
        sort_idx = dates_series.argsort()
        features_df = features_df.iloc[sort_idx].reset_index(drop=True)
        labels_series = labels_series.iloc[sort_idx].reset_index(drop=True)
        dates_series = dates_series.iloc[sort_idx].reset_index(drop=True)

        # Fill NaN in cross-sectional ranks (dates with only 1 symbol)
        cs_rank_cols = [c for c in features_df.columns if c.endswith("_cs_rank")]
        features_df[cs_rank_cols] = features_df[cs_rank_cols].fillna(0.5)

        logger.info(
            f"Built training data: {len(features_df)} samples, "
            f"{features_df.shape[1]} features "
            f"({len(cs_rank_cols)} cross-sectional), "
            f"win rate: {labels_series.mean():.1%}"
        )

        # Incorporate outcome tracker data if available
        features_df, labels_series, dates_series = self._merge_outcome_data(
            features_df, labels_series, dates_series
        )

        return features_df, labels_series, dates_series

    def _merge_outcome_data(
        self,
        features_df: pd.DataFrame,
        labels: pd.Series,
        dates: pd.Series,
    ) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
        """Merge outcome tracker data as additional training samples.

        Uses features cached at signal time (via log_signal) to prevent
        look-ahead bias. Falls back to recomputing only if no cache exists.
        """
        try:
            from advisor.ml.outcome_tracker import export_training_data

            outcome_df = export_training_data()
            if outcome_df.empty or len(outcome_df) < 5:
                return features_df, labels, dates

            logger.info(f"Adding {len(outcome_df)} samples from outcome tracker")

            for _, row in outcome_df.iterrows():
                ticker = row.get("ticker", "")
                if not ticker:
                    continue

                # Prefer cached features snapshot from signal time
                cached_feats = row.get("features")
                if isinstance(cached_feats, dict) and cached_feats:
                    feats = cached_feats
                else:
                    # Fallback: recompute (may have look-ahead bias)
                    feats = self._engine.compute_features(ticker)
                    if not feats:
                        continue

                feat_row = {col: feats.get(col, 0.0) for col in features_df.columns}
                win = 1 if row.get("return_pct", 0) > self.threshold else 0

                signal_date = row.get("signal_date")
                ts = pd.Timestamp(signal_date) if signal_date else pd.Timestamp.now()

                features_df = pd.concat([features_df, pd.DataFrame([feat_row])], ignore_index=True)
                labels = pd.concat([labels, pd.Series([win])], ignore_index=True)
                dates = pd.concat([dates, pd.Series([ts])], ignore_index=True)

        except Exception as e:
            logger.warning(f"Could not merge outcome data: {e}")

        return features_df, labels, dates

    def _compute_triple_barrier_labels(
        self,
        close: pd.Series,
        features: pd.DataFrame,
    ) -> tuple[pd.Series, pd.DatetimeIndex]:
        """Compute volatility-adjusted triple-barrier labels.

        For each sample, sets upper/lower barriers based on realized volatility.
        Scans forward bars to determine which barrier is hit first.

        Returns:
            (labels, valid_index) — aligned Series and DatetimeIndex.
        """
        vol_col = "realized_vol_20"
        has_vol = vol_col in features.columns
        close_arr = close.values
        close_idx = close.index
        feat_idx = features.index

        labels = {}
        for dt in feat_idx:
            pos = close_idx.get_loc(dt)
            if pos + self.horizon >= len(close_arr):
                continue  # not enough forward data

            entry_price = close_arr[pos]
            if has_vol:
                vol = features.loc[dt, vol_col]
                if pd.isna(vol) or vol <= 0:
                    vol = 0.15  # fallback ~15% annualized
            else:
                vol = 0.15

            # Volatility-scaled barriers (annualized vol -> horizon-scaled %)
            barrier_pct = max(
                self.threshold,
                1.5 * vol * np.sqrt(self.horizon / 252) * 100,
            )
            upper = entry_price * (1 + barrier_pct / 100)
            lower = entry_price * (1 - barrier_pct / 100)

            # Scan forward bars
            label = None
            for j in range(1, self.horizon + 1):
                p = close_arr[pos + j]
                if p >= upper:
                    label = 1
                    break
                if p <= lower:
                    label = 0
                    break

            if label is None:
                # Neither barrier hit — use sign of final return
                final_ret = close_arr[pos + self.horizon] / entry_price - 1
                label = 1 if final_ret > 0 else 0

            labels[dt] = label

        if not labels:
            return pd.Series(dtype=int), pd.DatetimeIndex([])

        label_series = pd.Series(labels, dtype=int)
        valid_idx = label_series.index
        return label_series, valid_idx

    def train_and_evaluate(
        self,
        model_type: ModelType = ModelType.LIGHTGBM,
        n_cv_splits: int = 5,
        train_cutoff: str | None = None,
        _snapshot: bool = True,
    ) -> dict[str, Any]:
        """Full pipeline: build data -> train -> evaluate -> save.

        Args:
            model_type: Which model to train.
            n_cv_splits: Number of time-series CV folds.
            train_cutoff: If set, only train on data up to this date.
                Stored in model metadata for OOS backtest enforcement.
            _snapshot: Create versioned snapshot (set False when called from train_with_meta).

        Returns training results with CV metrics.
        """
        features_df, labels, dates = self.build_training_data(cutoff_date=train_cutoff)

        trainer = MLModelTrainer(model_type=model_type)
        result = trainer.train(
            features_df,
            labels,
            dates,
            n_splits=n_cv_splits,
            horizon=self.horizon,
            decay=self.decay,
        )

        # Store training config in metadata for backtest enforcement
        result["metadata"]["horizon"] = self.horizon
        result["metadata"]["threshold"] = self.threshold
        result["metadata"]["label_mode"] = self.label_mode
        result["metadata"]["decay"] = self.decay
        result["metadata"]["train_cutoff"] = train_cutoff or str(dates.max().date())
        result["metadata"]["n_symbols"] = len(self.symbols)
        result["metadata"]["symbols"] = self.symbols
        trainer.metadata = result["metadata"]

        model_path = trainer.save()
        result["model_path"] = str(model_path)
        result["feature_importance"] = trainer.get_feature_importance()

        # Version snapshot
        if _snapshot:
            try:
                from advisor.ml.model_store import prune, snapshot

                vid = snapshot(metrics=result.get("cv_metrics", {}))
                prune(keep=10)
                result["version_id"] = vid
            except Exception as e:
                logger.warning("Model versioning failed: %s", e)

        return result

    def train_with_meta(
        self,
        model_type: ModelType = ModelType.LIGHTGBM,
        n_cv_splits: int = 5,
        train_cutoff: str | None = None,
    ) -> dict[str, Any]:
        """Train primary + meta-labeling model.

        Returns training results for both models and a precision comparison.
        """
        from advisor.ml.meta_label import MetaLabeler

        # Train primary model first (skip snapshot — we'll snapshot after meta is saved)
        result = self.train_and_evaluate(
            model_type=model_type,
            n_cv_splits=n_cv_splits,
            train_cutoff=train_cutoff,
            _snapshot=False,
        )

        # Build data again for meta-training (same data, same features)
        features_df, labels, dates = self.build_training_data(cutoff_date=train_cutoff)

        # Train meta-labeler
        meta = MetaLabeler()
        meta_result = meta.train(
            features_df,
            labels,
            dates,
            primary_model_type=model_type,
            n_splits=n_cv_splits,
            horizon=self.horizon,
            decay=self.decay,
        )
        meta_path = meta.save()
        result["meta_labeling"] = meta_result
        result["meta_model_path"] = str(meta_path)

        # Precision comparison: primary-only vs meta-labeled
        precision_comparison = meta.precision_at_thresholds(
            features_df,
            labels,
            dates,
            primary_model_type=model_type,
            n_splits=n_cv_splits,
            horizon=self.horizon,
            decay=self.decay,
        )
        result["precision_comparison"] = precision_comparison

        # Version snapshot (captures primary + meta + HMM)
        try:
            from advisor.ml.model_store import prune, snapshot

            metrics = {
                **result.get("cv_metrics", {}),
                "meta_auc": meta_result.get("metrics", {}).get("meta_auc"),
            }
            vid = snapshot(metrics=metrics)
            prune(keep=10)
            result["version_id"] = vid
        except Exception as e:
            logger.warning("Model versioning failed: %s", e)

        return result

    def compare_models(
        self,
        n_cv_splits: int = 5,
        train_cutoff: str | None = None,
    ) -> pd.DataFrame:
        """Train and compare all model types."""
        features_df, labels, dates = self.build_training_data(cutoff_date=train_cutoff)

        rows = []
        for mt in [ModelType.LOGISTIC, ModelType.LIGHTGBM, ModelType.ENSEMBLE]:
            logger.info(f"Training {mt}...")
            trainer = MLModelTrainer(model_type=mt)
            result = trainer.train(
                features_df,
                labels,
                dates,
                n_splits=n_cv_splits,
                horizon=self.horizon,
                decay=self.decay,
            )
            cv = result["cv_metrics"]

            row = {
                "model": str(mt),
                "cv_auc_mean": cv.get("cv_auc_mean", 0),
                "cv_auc_std": cv.get("cv_auc_std", 0),
                "cv_f1_mean": cv.get("cv_f1_mean", 0),
                "cv_accuracy_mean": cv.get("cv_accuracy_mean", 0),
                "cv_precision_mean": cv.get("cv_precision_mean", 0),
                "cv_recall_mean": cv.get("cv_recall_mean", 0),
                "cv_brier_mean": cv.get("cv_brier_mean", 0),
            }
            rows.append(row)

        return pd.DataFrame(rows).set_index("model")

    def precision_curve(
        self,
        model_type: ModelType = ModelType.LIGHTGBM,
        thresholds: list[float] | None = None,
        n_cv_splits: int = 5,
    ) -> dict[str, Any]:
        """Run precision-at-threshold analysis on pooled CV predictions."""
        features_df, labels, dates = self.build_training_data()

        trainer = MLModelTrainer(model_type=model_type)
        trainer.feature_names = list(features_df.columns)
        rows = trainer.precision_at_thresholds(
            features_df,
            labels,
            dates,
            thresholds=thresholds,
            n_splits=n_cv_splits,
            horizon=self.horizon,
            decay=self.decay,
        )

        base_rate = float(labels.mean())
        return {
            "thresholds": rows,
            "total_oos_samples": rows[0]["n_signals"] if rows else 0,
            "base_win_rate": round(base_rate * 100, 1),
            "model_type": str(model_type),
            "horizon": self.horizon,
            "n_symbols": len(self.symbols),
        }

    def walk_forward(
        self,
        model_type: ModelType = ModelType.LIGHTGBM,
        n_windows: int = 5,
    ) -> dict[str, Any]:
        """Run walk-forward validation to detect overfitting."""
        features_df, labels, dates = self.build_training_data()

        trainer = MLModelTrainer(model_type=model_type)
        return trainer.walk_forward_evaluate(
            features_df,
            labels,
            dates,
            n_windows=n_windows,
            horizon=self.horizon,
            decay=self.decay,
        )

    def backtest_signals(
        self,
        symbol: str,
        start: str | None = None,
        buy_threshold: float = 0.65,
    ) -> dict[str, Any]:
        """Backtest ML signals on strictly out-of-sample data.

        Loads the trained model and enforces that only data AFTER the
        model's train_cutoff date is used for evaluation. This prevents
        in-sample leakage from inflating backtest results.
        """
        import yfinance as yf

        symbol = symbol.upper()

        if not MLModelTrainer.model_exists():
            return {"error": "No trained model. Run 'advisor ml train' first."}

        trainer = MLModelTrainer.load()
        meta = trainer.metadata
        horizon = meta.get("horizon", self.horizon)
        threshold = meta.get("threshold", self.threshold)
        train_cutoff = meta.get("train_cutoff")
        train_symbols = meta.get("symbols", [])

        in_training_set = symbol in [s.upper() for s in train_symbols]

        # Download with enough warmup for features that need long history
        # (e.g. alpha_mom_12_1 needs 252 bars).  We'll filter to OOS dates below.
        features_df = self._engine.compute_features_df(symbol, period="2y")
        if features_df.empty:
            return {"error": f"Could not compute features for {symbol}"}

        # Enforce OOS: filter out any dates <= train_cutoff
        if train_cutoff:
            cutoff_ts = pd.Timestamp(train_cutoff)
            features_df = features_df[features_df.index > cutoff_ts]
            if features_df.empty:
                return {
                    "error": (
                        f"No out-of-sample data for {symbol} after " f"train cutoff {train_cutoff}"
                    )
                }

        # Get prices for forward return calculation
        df = yf.download(symbol, period="2y", progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        close = df["Close"]
        forward_ret = (close.shift(-horizon) / close - 1) * 100

        common_idx = features_df.index.intersection(forward_ret.dropna().index)
        if len(common_idx) < 10:
            return {"error": "Not enough out-of-sample data for backtesting"}

        feat_aligned = features_df.loc[common_idx]
        ret_aligned = forward_ret.loc[common_idx]

        # Generate predictions
        probs = trainer.predict_proba(feat_aligned)
        buy_mask = probs >= buy_threshold
        sell_threshold = 1.0 - buy_threshold
        sell_mask = probs <= sell_threshold

        buy_returns = ret_aligned.values[buy_mask]
        sell_returns = ret_aligned.values[sell_mask]
        all_returns = ret_aligned.values

        # Buy-and-hold baseline for comparison
        bh_return = float(np.mean(all_returns))

        results: dict[str, Any] = {
            "symbol": symbol,
            "horizon": horizon,
            "threshold": threshold,
            "oos_start": str(common_idx.min().date()),
            "oos_end": str(common_idx.max().date()),
            "train_cutoff": train_cutoff,
            "in_training_set": in_training_set,
            "total_bars": len(common_idx),
            "buy_signals": int(buy_mask.sum()),
            "sell_signals": int(sell_mask.sum()),
            "neutral_signals": int((~buy_mask & ~sell_mask).sum()),
            "baseline_avg_return": round(bh_return, 2),
        }

        if len(buy_returns) > 0:
            results["buy_avg_return"] = round(float(np.mean(buy_returns)), 2)
            results["buy_win_rate"] = round(float(np.mean(buy_returns > 0)) * 100, 1)
            results["buy_avg_win"] = (
                round(float(np.mean(buy_returns[buy_returns > 0])), 2)
                if np.any(buy_returns > 0)
                else 0.0
            )
            results["buy_avg_loss"] = (
                round(float(np.mean(buy_returns[buy_returns <= 0])), 2)
                if np.any(buy_returns <= 0)
                else 0.0
            )
            # Edge vs baseline
            results["buy_edge"] = round(results["buy_avg_return"] - bh_return, 2)
        else:
            results["buy_avg_return"] = 0.0
            results["buy_win_rate"] = 0.0
            results["buy_edge"] = 0.0

        if len(sell_returns) > 0:
            results["sell_avg_return"] = round(float(np.mean(sell_returns)), 2)
            results["sell_accuracy"] = round(float(np.mean(sell_returns < 0)) * 100, 1)

        return results
