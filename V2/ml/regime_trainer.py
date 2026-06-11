"""
V2/ml/regime_trainer.py
------------------------
Train a LightGBM classifier to predict market regime.

Input
-----
A CSV of NAS100 15m bars with all V2 indicators already calculated.
The trainer labels each bar with the rule-based regime (RegimeDetector) and
then trains LightGBM to replicate / generalise that classification.

Output
------
V2/models/lgbm_regime.txt — trained LightGBM booster

Usage
-----
    python V2/ml/regime_trainer.py --csv V1/cache/cache_15m_360d.csv

    or from Python:
        from ml.regime_trainer import train_regime_model
        train_regime_model(df)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ── Path bootstrap ────────────────────────────────────────────────────────────
_V2_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
_V1_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'V1'))
for _p in (_V2_ROOT, _V1_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    import lightgbm as lgb
    _LGB_OK = True
except ImportError:
    _LGB_OK = False

from ml.features_v2 import FeatureEngineV2
from pipeline.stage1_regime import RegimeDetector

try:
    from config.settings import LGBM_MODEL_DIR
except ImportError:
    LGBM_MODEL_DIR = os.path.join(_V2_ROOT, "models")

logger = logging.getLogger(__name__)

# Regime → integer label map
REGIME_LABELS = {
    "TREND_BULL": 0,
    "TREND_BEAR": 1,
    "BREAKOUT":   2,
    "CHOPPY":     3,
    "TRANSITION": 4,
}
WARMUP = 250  # bars to skip at start for indicator warm-up


# ── Feature columns used for regime prediction ───────────────────────────────
REGIME_FEATURES = [
    "ADX", "DI_Diff", "ATR_Ratio", "BB_Width_Ratio",
    "MACD_Hist_ATR", "MACD_Hist_Slope",
    "RSI", "RSI_Slope", "EMA_Align_Bars",
    "Close_vs_EMA50", "EMA_Gap_Pct",
]


def load_and_prepare(csv_path: str) -> pd.DataFrame:
    """Load CSV, compute V2 indicators, label regimes."""
    logger.info("Loading CSV: %s", csv_path)
    raw = pd.read_csv(csv_path, parse_dates=["Datetime"], index_col="Datetime")
    raw.sort_index(inplace=True)

    fe = FeatureEngineV2()
    df = fe.calculate_indicators(raw)

    detector = RegimeDetector()
    labels   = []
    for i in range(len(df)):
        if i < WARMUP:
            labels.append(-1)
        else:
            regime = detector.detect(df, i)
            labels.append(REGIME_LABELS.get(regime, 4))

    df["regime_label"] = labels
    return df


def build_feature_matrix(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Extract feature matrix X and label y from labelled DataFrame."""
    fe = FeatureEngineV2()
    rows = []
    ys   = []

    for i in range(WARMUP, len(df)):
        label = df["regime_label"].iloc[i]
        if label < 0:
            continue
        feat = fe.generate_live_features(df, i)
        if feat is None or feat.empty:
            continue
        rows.append(feat.iloc[0])
        ys.append(label)

    if not rows:
        raise ValueError("No valid training rows generated.")

    X = pd.DataFrame(rows).reindex(columns=REGIME_FEATURES, fill_value=0.0)
    y = pd.Series(ys, dtype=int)
    return X, y


def train_regime_model(df: pd.DataFrame, model_dir: str = LGBM_MODEL_DIR) -> None:
    """Train and save a LightGBM regime classifier."""
    if not _LGB_OK:
        raise ImportError("lightgbm is required. Install with: pip install lightgbm")

    logger.info("Building feature matrix for regime training...")
    X, y = build_feature_matrix(df)
    logger.info("Training set: %d rows, %d features, %d classes", len(X), X.shape[1], y.nunique())

    # Fill NaN with column median
    X.fillna(X.median(numeric_only=True), inplace=True)

    n_classes = len(REGIME_LABELS)
    dataset   = lgb.Dataset(X, label=y)

    params = {
        "objective":        "multiclass",
        "num_class":        n_classes,
        "metric":           "multi_logloss",
        "num_leaves":       31,
        "learning_rate":    0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq":     5,
        "verbose":         -1,
        "n_jobs":           -1,
    }

    logger.info("Training LightGBM regime classifier...")
    booster = lgb.train(
        params,
        dataset,
        num_boost_round=200,
        valid_sets=[dataset],
        callbacks=[lgb.log_evaluation(50)],
    )

    os.makedirs(model_dir, exist_ok=True)
    out_path = os.path.join(model_dir, "lgbm_regime.txt")
    booster.save_model(out_path)
    logger.info("Regime model saved → %s", out_path)


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")

    parser = argparse.ArgumentParser(description="Train V2 regime classifier")
    parser.add_argument(
        "--csv",
        default=os.path.join(_V1_ROOT, "cache", "cache_15m_360d.csv"),
        help="Path to NAS100 15m CSV file",
    )
    parser.add_argument("--model-dir", default=LGBM_MODEL_DIR)
    args = parser.parse_args()

    df = load_and_prepare(args.csv)
    train_regime_model(df, model_dir=args.model_dir)
