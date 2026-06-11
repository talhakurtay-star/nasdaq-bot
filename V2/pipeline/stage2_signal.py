"""
V2/pipeline/stage2_signal.py
-----------------------------
Stage 2 — Ensemble Signal Generator.

Combines V1 rule engine (weight 0.65) with LightGBM model (weight 0.35) to
produce a final LONG / SHORT / HOLD signal.

Logic
-----
- If rule_signal == HOLD → return HOLD (ML cannot override)
- If CHOPPY regime      → skip ML, return rule_signal directly (which may itself be HOLD)
- If rule_signal == LONG:
    lgbm_long_prob >= 0.52  → LONG
    lgbm_long_prob <  0.45  → HOLD (veto)
    0.45–0.52               → LONG (rule engine wins alone)
- Same logic mirrored for SHORT (using lgbm_short_prob)
- LightGBM gracefully degrades: if model not found → prob = 0.5 (neutral, rule wins)
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Literal, Tuple

import numpy as np
import pandas as pd

# ── V1 path ──────────────────────────────────────────────────────────────────
_V1_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'V1'))
if _V1_ROOT not in sys.path:
    sys.path.insert(0, _V1_ROOT)

# ── V2 config path ────────────────────────────────────────────────────────────
_V2_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
if _V2_ROOT not in sys.path:
    sys.path.insert(0, _V2_ROOT)

try:
    from config.settings import (
        ENSEMBLE_LGBM_WEIGHT,
        ENSEMBLE_RULE_WEIGHT,
        ENSEMBLE_MIN_SCORE,
        LGBM_MODEL_DIR,
        LGBM_LONG_CONFIRM,
        LGBM_LONG_VETO,
    )
except ImportError:
    ENSEMBLE_LGBM_WEIGHT = 0.35
    ENSEMBLE_RULE_WEIGHT = 0.65
    ENSEMBLE_MIN_SCORE   = 0.52
    LGBM_MODEL_DIR       = os.path.join(os.path.dirname(__file__), '..', 'models')
    LGBM_LONG_CONFIRM    = 0.52
    LGBM_LONG_VETO       = 0.45

try:
    from strategy.engine import StrategyEngine
except ImportError:
    StrategyEngine = None  # type: ignore

try:
    import lightgbm as lgb
    _LGB_AVAILABLE = True
except ImportError:
    _LGB_AVAILABLE = False

logger = logging.getLogger(__name__)

SignalResult = Literal["STRONG_LONG", "STRONG_SHORT", "HOLD"]


class EnsembleSignal:
    """
    Ensemble Signal Generator: V1 Rule Engine + LightGBM.

    Usage
    -----
    gen = EnsembleSignal()
    signal, score = gen.generate(df, current_index, regime)
    """

    def __init__(self) -> None:
        # V1 rule engine
        if StrategyEngine is not None:
            self._rule_engine = StrategyEngine()
        else:
            self._rule_engine = None
            logger.warning("StrategyEngine not available — rule signal will be HOLD.")

        # LightGBM models (lazy-loaded)
        self._lgbm_long:  object | None = None
        self._lgbm_short: object | None = None
        self._models_tried = False  # attempt load only once

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def generate(
        self,
        df: pd.DataFrame,
        current_index: int,
        regime: str,
        feature_engine=None,
    ) -> Tuple[SignalResult, float]:
        """
        Generate an ensemble signal for the current bar.

        Parameters
        ----------
        df            : DataFrame with calculated indicators
        current_index : integer position of current bar
        regime        : RegimeType string from Stage 1
        feature_engine: optional FeatureEngineV2 instance for ML features

        Returns
        -------
        (signal, score)
            signal — "STRONG_LONG" | "STRONG_SHORT" | "HOLD"
            score  — ensemble confidence score 0.0–1.0
        """
        # ── Step 1: Rule engine signal ────────────────────────────────────
        if self._rule_engine is not None:
            rule_signal = self._rule_engine.generate_base_signal(df, current_index)
        else:
            rule_signal = "HOLD"

        # Map V1 signal names to canonical V2 names
        if rule_signal == "STRONG_LONG":
            rule_dir = "LONG"
        elif rule_signal == "STRONG_SHORT":
            rule_dir = "SHORT"
        else:
            rule_dir = "HOLD"

        # ── If rule says HOLD, done — ML cannot override ──────────────────
        if rule_dir == "HOLD":
            return "HOLD", 0.0

        # ── If CHOPPY regime, skip ML and return rule signal directly ─────
        if regime == "CHOPPY":
            logger.debug("CHOPPY regime: skip ML, return rule signal %s", rule_signal)
            return rule_signal, ENSEMBLE_RULE_WEIGHT  # type: ignore[return-value]

        # ── Step 2: LightGBM probability ─────────────────────────────────
        lgbm_long_prob, lgbm_short_prob = self._get_lgbm_probs(
            df, current_index, feature_engine
        )

        # ── Step 3: Apply ensemble logic ──────────────────────────────────
        if rule_dir == "LONG":
            prob = lgbm_long_prob
            if prob >= LGBM_LONG_CONFIRM:
                # Both agree
                score = (
                    ENSEMBLE_RULE_WEIGHT * 1.0
                    + ENSEMBLE_LGBM_WEIGHT * prob
                )
                return "STRONG_LONG", min(score, 1.0)
            elif prob < LGBM_LONG_VETO:
                # ML vetoes rule signal
                logger.debug(
                    "LGBM veto LONG | bar=%d | lgbm_prob=%.3f < veto=%.2f",
                    current_index, prob, LGBM_LONG_VETO,
                )
                return "HOLD", 0.0
            else:
                # 0.45–0.52 range: rule engine wins alone
                score = ENSEMBLE_RULE_WEIGHT * 1.0
                return "STRONG_LONG", score

        else:  # rule_dir == "SHORT"
            prob = lgbm_short_prob
            if prob >= LGBM_LONG_CONFIRM:
                score = (
                    ENSEMBLE_RULE_WEIGHT * 1.0
                    + ENSEMBLE_LGBM_WEIGHT * prob
                )
                return "STRONG_SHORT", min(score, 1.0)
            elif prob < LGBM_LONG_VETO:
                logger.debug(
                    "LGBM veto SHORT | bar=%d | lgbm_prob=%.3f < veto=%.2f",
                    current_index, prob, LGBM_LONG_VETO,
                )
                return "HOLD", 0.0
            else:
                score = ENSEMBLE_RULE_WEIGHT * 1.0
                return "STRONG_SHORT", score

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _load_models(self) -> None:
        """Attempt to load LightGBM models from disk (once)."""
        if self._models_tried:
            return
        self._models_tried = True

        if not _LGB_AVAILABLE:
            logger.info("lightgbm not installed — ensemble falls back to rule engine.")
            return

        long_path  = os.path.join(LGBM_MODEL_DIR, "lgbm_long.txt")
        short_path = os.path.join(LGBM_MODEL_DIR, "lgbm_short.txt")

        if os.path.exists(long_path):
            try:
                self._lgbm_long = lgb.Booster(model_file=long_path)
                logger.info("Loaded LGBM long model from %s", long_path)
            except Exception as exc:
                logger.warning("Failed to load LGBM long model: %s", exc)
        else:
            logger.info("LGBM long model not found at %s — pass-through (prob=0.5)", long_path)

        if os.path.exists(short_path):
            try:
                self._lgbm_short = lgb.Booster(model_file=short_path)
                logger.info("Loaded LGBM short model from %s", short_path)
            except Exception as exc:
                logger.warning("Failed to load LGBM short model: %s", exc)
        else:
            logger.info("LGBM short model not found at %s — pass-through (prob=0.5)", short_path)

    def _get_lgbm_probs(
        self,
        df: pd.DataFrame,
        current_index: int,
        feature_engine=None,
    ) -> Tuple[float, float]:
        """
        Return (lgbm_long_prob, lgbm_short_prob).
        Falls back to 0.5 (neutral) if model not loaded or inference fails.
        """
        self._load_models()

        if self._lgbm_long is None and self._lgbm_short is None:
            return 0.5, 0.5

        # Build feature row
        feat_row = self._build_feature_row(df, current_index, feature_engine)
        if feat_row is None:
            return 0.5, 0.5

        long_prob  = self._predict_single(self._lgbm_long,  feat_row)
        short_prob = self._predict_single(self._lgbm_short, feat_row)
        return long_prob, short_prob

    @staticmethod
    def _predict_single(model, feat_row: pd.DataFrame) -> float:
        """Run a single-row prediction; return 0.5 on any error."""
        if model is None:
            return 0.5
        try:
            pred = model.predict(feat_row)
            prob = float(pred[0]) if hasattr(pred, '__len__') else float(pred)
            return max(0.0, min(1.0, prob))
        except Exception as exc:
            logger.debug("LGBM prediction error: %s", exc)
            return 0.5

    def _build_feature_row(
        self,
        df: pd.DataFrame,
        current_index: int,
        feature_engine=None,
    ) -> pd.DataFrame | None:
        """Build a single-row DataFrame of ML features for current bar.
        Columns are aligned to the model's expected SIGNAL_FEATURES order."""
        # Get the feature column list from the model (stored at training time)
        feature_names = self._get_model_feature_names()

        try:
            if feature_engine is not None:
                raw = feature_engine.generate_live_features(df, current_index)
            else:
                row = df.iloc[current_index]
                raw = pd.DataFrame([{
                    col: float(val)
                    for col, val in row.items()
                    if isinstance(val, (int, float, np.floating, np.integer))
                    and not (isinstance(val, float) and np.isnan(val))
                }])

            if raw is None or raw.empty:
                return None

            if feature_names:
                # Reindex to match exact training column order; fill missing with 0
                raw = raw.reindex(columns=feature_names, fill_value=0.0)
            return raw
        except Exception as exc:
            logger.debug("Feature build error: %s", exc)
            return None

    def _get_model_feature_names(self) -> list:
        """Return the feature names the long model was trained on (if available)."""
        if self._lgbm_long is not None:
            try:
                return self._lgbm_long.feature_name()
            except Exception:
                pass
        if self._lgbm_short is not None:
            try:
                return self._lgbm_short.feature_name()
            except Exception:
                pass
        return []
