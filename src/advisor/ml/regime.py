"""HMM-based market regime detection.

Fits a 3-state Gaussian Hidden Markov Model on SPY daily returns and VIX
levels to identify low-volatility (calm), normal, and high-volatility
(stressed) market regimes.  The fitted model can produce regime labels
and transition probabilities for use as ML features.
"""

from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parents[3] / "data"
_MODEL_DIR = _DATA_DIR / "ml_models"
_MODEL_PATH = _MODEL_DIR / "hmm_regime.joblib"

_N_STATES = 3
_REGIME_NAMES = {0: "low_vol", 1: "normal", 2: "high_vol"}


class RegimeDetector:
    """3-state Gaussian HMM for market regime detection.

    States are labeled by ascending volatility after fitting:
        0 = low_vol (calm)
        1 = normal
        2 = high_vol (stressed)
    """

    def __init__(self) -> None:
        self.model = None
        self._state_map: dict[int, int] | None = None
        self._n_features: int = 2
        self._fitted_at: str | None = None

    # ── Fitting ──────────────────────────────────────────────────────────

    def fit(self, lookback: str = "5y") -> dict:
        """Download SPY + VIX data and fit the HMM.

        Args:
            lookback: yfinance period string for historical data.

        Returns:
            Dict with fit summary (n_obs, log_likelihood, regime counts).
        """
        from hmmlearn.hmm import GaussianHMM

        obs, dates, n_features = self._build_observations(lookback=lookback)
        self._n_features = n_features

        logger.info(
            "Fitting %d-state GaussianHMM on %d observations (%d features)",
            _N_STATES,
            len(obs),
            n_features,
        )

        hmm = GaussianHMM(
            n_components=_N_STATES,
            covariance_type="full",
            n_iter=100,
            random_state=42,
        )
        hmm.fit(obs)
        self.model = hmm

        # Label states by ascending volatility of SPY returns (feature 0)
        self._state_map = self._label_states_by_volatility(hmm, obs)
        self._fitted_at = datetime.now().isoformat()

        # Decode for summary stats
        raw_states = hmm.predict(obs)
        mapped = np.array([self._state_map[s] for s in raw_states])

        summary = {
            "n_observations": len(obs),
            "n_features": n_features,
            "log_likelihood": round(float(hmm.score(obs)), 2),
            "fitted_at": self._fitted_at,
            "regime_counts": {_REGIME_NAMES[i]: int((mapped == i).sum()) for i in range(_N_STATES)},
        }
        logger.info("HMM fit complete: %s", summary)
        return summary

    # ── Regime detection ─────────────────────────────────────────────────

    def detect_regime(self, date: str | None = None) -> dict:
        """Return the regime for a given date (or the latest available).

        Args:
            date: ISO date string (YYYY-MM-DD). If None, uses most recent
                trading day.

        Returns:
            Dict with keys: regime, regime_name, regime_prob, spy_vol, vix.
        """
        self._ensure_fitted()

        obs, dates, _ = self._build_observations(lookback="6mo")

        if date is not None:
            target = pd.Timestamp(date)
            # Find the closest date on or before the target
            mask = dates <= target
            if not mask.any():
                raise ValueError(f"No data available on or before {date}")
            idx = mask.values.nonzero()[0][-1]
        else:
            idx = len(dates) - 1

        # Get state probabilities for the entire sequence up to idx
        seq = obs[: idx + 1]
        log_prob, raw_states = self.model.decode(seq)
        posteriors = self.model.predict_proba(seq)

        raw_state = raw_states[-1]
        mapped_state = self._state_map[raw_state]

        # Remap posterior probabilities to match our labeling
        raw_posteriors = posteriors[-1]
        mapped_probs = np.zeros(_N_STATES)
        for raw_s, mapped_s in self._state_map.items():
            mapped_probs[mapped_s] = raw_posteriors[raw_s]

        # SPY realized vol (20-day annualized) at this point
        spy_returns = obs[: idx + 1, 0]
        spy_vol = float(np.std(spy_returns[-20:]) * np.sqrt(252)) if len(spy_returns) >= 20 else 0.0

        # VIX level (feature 1 if available, else 0)
        vix_val = float(obs[idx, 1]) if obs.shape[1] > 1 else 0.0

        return {
            "date": str(dates[idx].date()),
            "regime": int(mapped_state),
            "regime_name": _REGIME_NAMES[mapped_state],
            "regime_prob": mapped_probs.tolist(),
            "spy_vol": round(spy_vol, 4),
            "vix": round(vix_val, 2),
        }

    # ── Feature computation ──────────────────────────────────────────────

    def compute_regime_features(self, dates: pd.DatetimeIndex) -> pd.DataFrame:
        """Compute regime features for a set of dates.

        Args:
            dates: DatetimeIndex of dates to compute features for.

        Returns:
            DataFrame indexed by ``dates`` with columns:
            - hmm_regime (int 0/1/2)
            - hmm_low_vol_prob (float)
            - hmm_normal_prob (float)
            - hmm_high_vol_prob (float)
            - hmm_transition_prob (float) — probability of transitioning
              to a different regime on the next step
        """
        self._ensure_fitted()

        # Build observations covering the requested date range plus warm-up
        earliest = dates.min()
        latest = dates.max()
        warmup_start = earliest - pd.DateOffset(months=6)

        obs_all, dates_all, _ = self._build_observations(
            start=str(warmup_start.date()),
            end=str((latest + pd.DateOffset(days=5)).date()),
        )

        if len(obs_all) == 0:
            logger.warning("No observation data available for regime features")
            return pd.DataFrame(
                index=dates,
                columns=[
                    "hmm_regime",
                    "hmm_low_vol_prob",
                    "hmm_normal_prob",
                    "hmm_high_vol_prob",
                    "hmm_transition_prob",
                ],
                data=0.0,
            )

        # Decode full sequence
        raw_states = self.model.predict(obs_all)
        posteriors = self.model.predict_proba(obs_all)

        # Map raw states to labeled states
        mapped_states = np.array([self._state_map[s] for s in raw_states])

        # Remap posterior columns
        mapped_posteriors = np.zeros_like(posteriors)
        for raw_s, mapped_s in self._state_map.items():
            mapped_posteriors[:, mapped_s] = posteriors[:, raw_s]

        # Transition probability: P(next state != current state)
        # From the model's transition matrix, sum off-diagonal for current state
        transmat = self.model.transmat_
        transition_probs = np.zeros(len(raw_states))
        for i, rs in enumerate(raw_states):
            transition_probs[i] = 1.0 - transmat[rs, rs]

        # Build a full-range Series for each feature, then reindex to requested dates
        regime_series = pd.Series(mapped_states, index=dates_all, dtype=int)
        low_vol_prob = pd.Series(mapped_posteriors[:, 0], index=dates_all)
        normal_prob = pd.Series(mapped_posteriors[:, 1], index=dates_all)
        high_vol_prob = pd.Series(mapped_posteriors[:, 2], index=dates_all)
        trans_prob = pd.Series(transition_probs, index=dates_all)

        result = pd.DataFrame(
            {
                "hmm_regime": regime_series.reindex(dates, method="ffill"),
                "hmm_low_vol_prob": low_vol_prob.reindex(dates, method="ffill"),
                "hmm_normal_prob": normal_prob.reindex(dates, method="ffill"),
                "hmm_high_vol_prob": high_vol_prob.reindex(dates, method="ffill"),
                "hmm_transition_prob": trans_prob.reindex(dates, method="ffill"),
            },
            index=dates,
        )

        # Fill any remaining NaN (dates before observation window)
        result["hmm_regime"] = result["hmm_regime"].fillna(1).astype(int)
        result = result.fillna(0.0)

        return result

    # ── Persistence ──────────────────────────────────────────────────────

    def save(self, path: Path | None = None) -> Path:
        """Persist the fitted HMM to disk."""
        self._ensure_fitted()
        path = path or _MODEL_PATH
        path.parent.mkdir(parents=True, exist_ok=True)

        payload = {
            "model": self.model,
            "state_map": self._state_map,
            "n_features": self._n_features,
            "fitted_at": self._fitted_at,
        }
        joblib.dump(payload, path)
        logger.info("HMM regime model saved to %s", path)
        return path

    @classmethod
    def load(cls, path: Path | None = None) -> "RegimeDetector":
        """Load a previously fitted HMM from disk."""
        path = path or _MODEL_PATH
        if not path.exists():
            raise FileNotFoundError(f"No HMM regime model found at {path}")

        payload = joblib.load(path)
        detector = cls()
        detector.model = payload["model"]
        detector._state_map = payload["state_map"]
        detector._n_features = payload.get("n_features", 2)
        detector._fitted_at = payload.get("fitted_at")

        logger.info("HMM regime model loaded from %s (fitted %s)", path, detector._fitted_at)
        return detector

    @staticmethod
    def model_exists(path: Path | None = None) -> bool:
        """Check whether a saved HMM model exists on disk."""
        return (path or _MODEL_PATH).exists()

    # ── Internal helpers ─────────────────────────────────────────────────

    def _ensure_fitted(self) -> None:
        """Raise if the model has not been fitted or loaded."""
        if self.model is None or self._state_map is None:
            raise RuntimeError("RegimeDetector not fitted. Call fit() or load() first.")

    def _build_observations(
        self,
        lookback: str | None = None,
        start: str | None = None,
        end: str | None = None,
    ) -> tuple[np.ndarray, pd.DatetimeIndex, int]:
        """Download SPY + VIX and return (observations, dates, n_features).

        Falls back to SPY-only (1 feature) if VIX data is unavailable.

        Args:
            lookback: yfinance period string (used when start is None).
            start: Start date (YYYY-MM-DD).
            end: End date (YYYY-MM-DD).

        Returns:
            (obs, dates, n_features) where obs is shape (T, n_features).
        """
        dl_kwargs: dict = {"progress": False}
        if start:
            dl_kwargs["start"] = start
            if end:
                dl_kwargs["end"] = end
        else:
            dl_kwargs["period"] = lookback or "5y"

        # Download SPY
        spy = yf.download("SPY", **dl_kwargs)
        if spy.empty:
            raise ValueError("Failed to download SPY data")

        if isinstance(spy.columns, pd.MultiIndex):
            spy.columns = spy.columns.get_level_values(0)

        spy_returns = spy["Close"].pct_change().dropna()
        dates = spy_returns.index

        # Attempt to download VIX
        vix_close = None
        try:
            vix = yf.download("^VIX", **dl_kwargs)
            if not vix.empty:
                if isinstance(vix.columns, pd.MultiIndex):
                    vix.columns = vix.columns.get_level_values(0)
                vix_close = vix["Close"].reindex(dates, method="ffill").dropna()
        except Exception as e:
            logger.warning("VIX download failed, falling back to SPY-only: %s", e)

        # Build observation matrix
        if vix_close is not None and len(vix_close) > 0:
            common = dates.intersection(vix_close.index)
            if len(common) > 50:
                spy_ret_aligned = spy_returns.loc[common].values.reshape(-1, 1)
                vix_aligned = vix_close.loc[common].values.reshape(-1, 1)
                obs = np.hstack([spy_ret_aligned, vix_aligned])
                logger.debug(
                    "Built 2-feature observations: %d rows (SPY returns + VIX)",
                    len(obs),
                )
                return obs, common, 2

        # Fallback: SPY returns only
        logger.warning("Using SPY returns only (1 feature) — VIX data unavailable or insufficient")
        obs = spy_returns.values.reshape(-1, 1)
        return obs, dates, 1

    @staticmethod
    def _label_states_by_volatility(
        hmm,
        obs: np.ndarray,
    ) -> dict[int, int]:
        """Sort HMM states by the volatility of SPY returns within each state.

        The state whose observations have the lowest standard deviation of
        SPY returns gets label 0 (calm), middle gets 1 (normal), highest
        gets 2 (stressed).

        Returns:
            Mapping from raw HMM state index -> labeled regime index.
        """
        raw_states = hmm.predict(obs)
        spy_returns = obs[:, 0]

        state_vols = {}
        for s in range(_N_STATES):
            mask = raw_states == s
            if mask.sum() > 1:
                state_vols[s] = float(np.std(spy_returns[mask]))
            else:
                # Degenerate state — assign large vol so it sorts last
                state_vols[s] = float("inf")

        # Sort by volatility ascending: lowest vol -> regime 0, etc.
        sorted_states = sorted(state_vols, key=lambda s: state_vols[s])

        state_map = {}
        for label, raw_state in enumerate(sorted_states):
            state_map[raw_state] = label

        logger.info(
            "State labeling (raw -> labeled): %s | vols: %s",
            state_map,
            {s: round(v, 6) for s, v in state_vols.items()},
        )
        return state_map
