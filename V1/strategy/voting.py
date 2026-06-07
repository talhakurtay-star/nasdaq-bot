"""
strategy/voting.py
------------------
Direction-aware XGBoost veto layer.

Long and short probabilities are produced by separate models. This deliberately
removes the invalid assumption that P(short TP) == 1 - P(long TP).
"""

from __future__ import annotations

import logging
import os
from typing import Optional

import pandas as pd

try:
    import xgboost as xgb
except ImportError as exc:
    raise ImportError("xgboost package is missing. Install with: pip install xgboost") from exc

try:
    from config.settings import XGB_PROBABILITY_THRESHOLD
    from config import settings as _cfg

    MODEL_DIR: str = _cfg.MODEL_DIR
except ImportError:
    XGB_PROBABILITY_THRESHOLD = 0.54
    MODEL_DIR = "models"

from strategy.engine import SignalType

logger = logging.getLogger(__name__)

MODEL_FILES = {
    "STRONG_LONG": "xgb_model_long.json",
    "STRONG_SHORT": "xgb_model_short.json",
}


class VotingMechanism:
    """XGBoost probability filter with independent long and short models."""

    def __init__(self) -> None:
        self.models: dict[str, Optional[xgb.Booster]] = {
            "STRONG_LONG": None,
            "STRONG_SHORT": None,
        }
        self.threshold: float = XGB_PROBABILITY_THRESHOLD
        self._load_models()

    def _load_one(self, signal: SignalType, filename: str) -> Optional[xgb.Booster]:
        model_path = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(model_path):
            logger.warning(
                "%s model not found: %s. This direction will fail closed to HOLD.",
                signal,
                model_path,
            )
            return None

        try:
            booster = xgb.Booster()
            booster.load_model(model_path)
            logger.info("%s XGBoost model loaded: %s", signal, model_path)
            return booster
        except xgb.core.XGBoostError as exc:
            logger.error("%s model could not be loaded (%s): %s", signal, model_path, exc)
            return None

    def _load_models(self) -> None:
        for signal, filename in MODEL_FILES.items():
            self.models[signal] = self._load_one(signal, filename)

    def _extract_probability(self, features_df: pd.DataFrame, signal: SignalType) -> float:
        model = self.models.get(signal)
        if model is None:
            return 0.0

        raw_pred = model.predict(xgb.DMatrix(features_df))
        if raw_pred.ndim == 1:
            return float(raw_pred[0])
        return float(raw_pred[0, 1])

    def decide_trade(
        self,
        df: pd.DataFrame,
        current_index: int,
        base_signal: SignalType,
        feature_engine,
    ) -> SignalType:
        if base_signal == "HOLD":
            return "HOLD"

        if base_signal not in self.models:
            logger.warning("Unknown base signal %s. Returning HOLD.", base_signal)
            return "HOLD"

        if self.models.get(base_signal) is None:
            logger.warning(
                "Bar %d: %s model is unavailable. Fail-closed HOLD.",
                current_index,
                base_signal,
            )
            return "HOLD"

        try:
            features_df = feature_engine.generate_live_features(df, current_index)
        except Exception as exc:
            logger.error("Bar %d: feature generation failed: %s. HOLD.", current_index, exc)
            return "HOLD"

        if features_df is None or features_df.empty:
            logger.debug("Bar %d: insufficient feature history. HOLD.", current_index)
            return "HOLD"

        try:
            probability = self._extract_probability(features_df, base_signal)
        except Exception as exc:
            logger.error("Bar %d: XGBoost prediction failed: %s. HOLD.", current_index, exc)
            return "HOLD"

        if probability >= self.threshold:
            logger.info(
                "Bar %d: %s approved | probability=%.4f >= threshold=%.2f",
                current_index,
                base_signal,
                probability,
                self.threshold,
            )
            return base_signal

        logger.info(
            "Bar %d: %s vetoed | probability=%.4f < threshold=%.2f",
            current_index,
            base_signal,
            probability,
            self.threshold,
        )
        return "HOLD"
