"""
V2/ml/signal_trainer.py
------------------------
Train LightGBM binary classifiers for LONG and SHORT trade signals.

For each trade direction the model predicts P(trade profitable).

Label construction
------------------
A bar is a positive LONG example if:
  - The V1 rule engine emitted STRONG_LONG
  - The subsequent trade closed with net_pnl > 0  (TP hit, not SL)

A bar is a positive SHORT example similarly.

Output
------
V2/models/lgbm_long.txt
V2/models/lgbm_short.txt

Usage
-----
    python V2/ml/signal_trainer.py --csv V1/cache/cache_15m_360d.csv

    or from Python:
        from ml.signal_trainer import train_signal_models
        train_signal_models(df)
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from typing import Tuple

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
from strategy.engine import StrategyEngine

try:
    from config.settings import (
        ATR_MULTIPLIER,
        REWARD_RISK_RATIO,
        LGBM_MODEL_DIR,
    )
except ImportError:
    ATR_MULTIPLIER    = 2.5
    REWARD_RISK_RATIO = 2.5
    LGBM_MODEL_DIR    = os.path.join(_V2_ROOT, "models")

logger = logging.getLogger(__name__)

WARMUP = 250  # bars to skip at start

# Features to use for signal prediction (all V1 + key V2 additions)
SIGNAL_FEATURES = [
    # V1 core
    "RSI", "RSI_7", "RSI_Change3",
    "ADX", "DMP", "DMN", "DI_Diff",
    "MACD_Hist_ATR", "MACD_Hist_Change", "MACD_Signal_Gap",
    "BB_Pct", "BB_Width",
    "StochRSI_K", "StochRSI_D", "StochRSI_KD_Diff",
    "ROC",
    "EMA_Gap_Pct", "Close_vs_EMA50", "Close_vs_EMA200",
    "EMA_Fast_Slope", "EMA_Slow_Slope",
    "EMA_Align_Bars",
    "Body_Ratio", "Upper_Wick_Ratio", "Lower_Wick_Ratio", "Candle_Direction",
    "Price_Position_20",
    "ATR_Pct", "ATR_Ratio",
    "Volume_Z_Score", "Volume_Trend",
    "ADX_Persistence", "ADX_Slope",
    "H1_EMA_Trend", "H4_EMA_Trend", "D1_Close_vs_EMA200",
    "Sin_Hour", "Cos_Hour", "Sin_DayOfWeek", "Cos_DayOfWeek",
    # V2 additions
    "BB_Width_Ratio", "MACD_Hist_Slope", "RSI_Slope", "EMA9_Slope_Pct",
]


def _simulate_outcome(df: pd.DataFrame, entry_idx: int, direction: str) -> int:
    """
    Simulate whether a trade opened at entry_idx would win (TP) or lose (SL).

    Returns 1 for TP (win), 0 for SL (loss), -1 if undetermined (trade still open
    at end of data).
    """
    try:
        bar       = df.iloc[entry_idx]
        entry_px  = float(bar["Close"])
        atr       = float(bar.get("ATR", 0.0) or 0.0)
        if atr <= 0:
            return -1

        sl_dist   = atr * ATR_MULTIPLIER
        tp_dist   = sl_dist * REWARD_RISK_RATIO

        if direction == "LONG":
            sl_px = entry_px - sl_dist
            tp_px = entry_px + tp_dist
        else:
            sl_px = entry_px + sl_dist
            tp_px = entry_px - tp_dist

        for i in range(entry_idx + 1, min(entry_idx + 200, len(df))):
            h = float(df.iloc[i]["High"])
            l = float(df.iloc[i]["Low"])

            if direction == "LONG":
                if l <= sl_px:
                    return 0
                if h >= tp_px:
                    return 1
            else:
                if h >= sl_px:
                    return 0
                if l <= tp_px:
                    return 1

        return -1  # timeout — exclude

    except Exception:
        return -1


def build_datasets(df: pd.DataFrame) -> Tuple[
    pd.DataFrame, pd.Series, pd.DataFrame, pd.Series
]:
    """
    Build (X_long, y_long, X_short, y_short) training datasets.
    Only bars where the rule engine fires are labelled.
    """
    fe      = FeatureEngineV2()
    engine  = StrategyEngine()

    X_long,  y_long  = [], []
    X_short, y_short = [], []

    for i in range(WARMUP, len(df)):
        signal = engine.generate_base_signal(df, i)
        if signal not in ("STRONG_LONG", "STRONG_SHORT"):
            continue

        direction = "LONG" if signal == "STRONG_LONG" else "SHORT"
        outcome   = _simulate_outcome(df, i, direction)
        if outcome < 0:
            continue

        feat = fe.generate_live_features(df, i)
        if feat is None or feat.empty:
            continue

        row = feat.iloc[0]

        if direction == "LONG":
            X_long.append(row)
            y_long.append(outcome)
        else:
            X_short.append(row)
            y_short.append(outcome)

    def _to_df(rows, ys):
        X = pd.DataFrame(rows).reindex(columns=SIGNAL_FEATURES, fill_value=0.0)
        X.fillna(X.median(numeric_only=True), inplace=True)
        return X, pd.Series(ys, dtype=int)

    logger.info("LONG  samples: %d (win=%d)", len(y_long),  sum(y_long))
    logger.info("SHORT samples: %d (win=%d)", len(y_short), sum(y_short))

    return _to_df(X_long, y_long) + _to_df(X_short, y_short)


def _train_one(X: pd.DataFrame, y: pd.Series, label: str, model_dir: str) -> None:
    """Train and save one binary LightGBM classifier."""
    if len(y) < 50:
        logger.warning("%s: only %d samples — skipping training.", label, len(y))
        return

    dataset = lgb.Dataset(X, label=y)
    params  = {
        "objective":        "binary",
        "metric":           "binary_logloss",
        "num_leaves":       31,
        "learning_rate":    0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq":     5,
        "verbose":         -1,
        "n_jobs":           -1,
    }
    booster = lgb.train(
        params,
        dataset,
        num_boost_round=300,
        valid_sets=[dataset],
        callbacks=[lgb.log_evaluation(100)],
    )

    os.makedirs(model_dir, exist_ok=True)
    out_path = os.path.join(model_dir, f"lgbm_{label.lower()}.txt")
    booster.save_model(out_path)
    logger.info("%s model saved → %s", label, out_path)


def train_signal_models(df: pd.DataFrame, model_dir: str = LGBM_MODEL_DIR) -> None:
    """Train and save LGBM long and short signal classifiers."""
    if not _LGB_OK:
        raise ImportError("lightgbm is required. Install with: pip install lightgbm")

    logger.info("Building signal training datasets...")
    X_long, y_long, X_short, y_short = build_datasets(df)

    logger.info("Training LONG model...")
    _train_one(X_long, y_long, "long", model_dir)

    logger.info("Training SHORT model...")
    _train_one(X_short, y_short, "short", model_dir)


# ── CLI entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")

    parser = argparse.ArgumentParser(description="Train V2 signal classifiers")
    parser.add_argument(
        "--csv",
        default=os.path.join(_V1_ROOT, "cache", "cache_15m_360d.csv"),
        help="Path to NAS100 15m CSV file",
    )
    parser.add_argument("--model-dir", default=LGBM_MODEL_DIR)
    args = parser.parse_args()

    from ml.features_v2 import FeatureEngineV2
    fe  = FeatureEngineV2()
    raw = pd.read_csv(args.csv, parse_dates=["Datetime"], index_col="Datetime")
    raw.sort_index(inplace=True)
    df  = fe.calculate_indicators(raw)

    train_signal_models(df, model_dir=args.model_dir)
