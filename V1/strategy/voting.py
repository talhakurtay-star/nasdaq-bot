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
import pickle
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


CALIBRATED_FILES = {
    "STRONG_LONG":  "xgb_model_long_calibrated.pkl",
    "STRONG_SHORT": "xgb_model_short_calibrated.pkl",
}


class VotingMechanism:
    """XGBoost probability filter with independent long and short models."""

    def __init__(self) -> None:
        self.models: dict[str, Optional[xgb.Booster]] = {
            "STRONG_LONG": None,
            "STRONG_SHORT": None,
        }
        self.calibrated: dict[str, Optional[object]] = {
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

    def _load_calibrated(self, signal: SignalType, filename: str) -> Optional[object]:
        cal_path = os.path.join(MODEL_DIR, filename)
        if not os.path.exists(cal_path):
            return None
        try:
            with open(cal_path, "rb") as f:
                cal = pickle.load(f)
            logger.info("%s kalibre model yüklendi: %s", signal, cal_path)
            return cal
        except Exception as exc:
            logger.warning("%s kalibrasyon yüklenemedi: %s", signal, exc)
            return None

    def _load_models(self) -> None:
        for signal, filename in MODEL_FILES.items():
            self.models[signal] = self._load_one(signal, filename)
        for signal, filename in CALIBRATED_FILES.items():
            self.calibrated[signal] = self._load_calibrated(signal, filename)

    def _extract_probability(self, features_df: pd.DataFrame, signal: SignalType) -> float:
        model = self.models.get(signal)
        if model is None:
            return 0.0

        raw_pred = model.predict(xgb.DMatrix(features_df))
        raw_prob = float(raw_pred[0]) if raw_pred.ndim == 1 else float(raw_pred[0, 1])

        # Kalibre iso model varsa ham prob'u düzelt
        cal = self.calibrated.get(signal)
        if cal is not None:
            try:
                iso = cal.get("iso")
                if iso is not None:
                    return float(iso.predict([raw_prob])[0])
            except Exception:
                pass

        return raw_prob

    def get_signal_with_prob(
        self,
        df: pd.DataFrame,
        current_index: int,
        base_signal: SignalType,
        feature_engine,
    ) -> tuple[SignalType, float]:
        """
        decide_trade ile aynı mantık ama (signal, probability) tuple döndürür.
        Multi-symbol sıralama için kullanılır.
        """
        if base_signal == "HOLD":
            return "HOLD", 0.0
        if base_signal not in self.models or self.models.get(base_signal) is None:
            return "HOLD", 0.0
        try:
            features_df = feature_engine.generate_live_features(df, current_index)
        except Exception:
            return "HOLD", 0.0
        if features_df is None or features_df.empty:
            return "HOLD", 0.0
        try:
            probability = self._extract_probability(features_df, base_signal)
        except Exception:
            return "HOLD", 0.0
        if self.threshold <= 0.0 or probability >= self.threshold:
            return base_signal, probability
        return "HOLD", probability

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

        # XGB veto devre dışı: model AUC ~0.46 (rastgele altı) olduğunda
        # strateji motorunun filtrelerine güvenmek daha iyi sonuç verir.
        # Threshold=0.0 ile tüm strateji sinyalleri geçer.
        if self.threshold <= 0.0:
            logger.debug("Bar %d: %s — XGB veto disabled, signal passed.", current_index, base_signal)
            return base_signal

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
