"""
ml/trainer.py
-------------
Dual-direction XGBoost training pipeline.

The old model answered only this question:
    "If I open LONG here, does TP arrive before SL?"

That is not enough for a short signal. This trainer now creates two independent
targets and two independent models:
    target_long  -> P(long trade reaches TP before SL)
    target_short -> P(short trade reaches TP before SL)
"""

from __future__ import annotations

import logging
import os
import sys
import time
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import xgboost as xgb

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from config.settings import ATR_MULTIPLIER, MODEL_DIR, REWARD_RISK_RATIO
from data.data_feed import DataFeed
from ml.features import FeatureEngine

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("trainer")

TRAIN_RATIO = 0.80
LOOKAHEAD = int(os.getenv("STRESS_LABEL_LOOKAHEAD", "20"))
RANDOM_STATE = 42
MODEL_FILES = {
    "long": "xgb_model_long.json",
    "short": "xgb_model_short.json",
}


def _apply_training_window(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optional split support for walk-forward optimization.

    STRESS_TRAIN_WINDOW:
        FULL -> use all bars
        IS   -> use everything up to the last STRESS_OOS_DAYS days
        OOS  -> use only the last STRESS_OOS_DAYS days
    """
    window = os.getenv("STRESS_TRAIN_WINDOW", "FULL").upper()
    if window == "FULL" or df.empty:
        return df

    oos_days = int(os.getenv("STRESS_OOS_DAYS", "90"))
    cutoff = df.index.max() - pd.Timedelta(days=oos_days)

    if window == "IS":
        sliced = df.loc[df.index <= cutoff].copy()
    elif window == "OOS":
        sliced = df.loc[df.index > cutoff].copy()
    else:
        logger.warning("Unknown STRESS_TRAIN_WINDOW=%s, using FULL.", window)
        return df

    logger.info(
        "Training window=%s | cutoff=%s | bars=%d/%d",
        window,
        cutoff,
        len(sliced),
        len(df),
    )
    return sliced


def _first_touch_label(
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    start_idx: int,
    sl_price: float,
    tp_price: float,
    direction: str,
) -> float:
    """
    Return 1.0 if TP is touched before SL, 0.0 if SL is touched first, NaN if
    neither is touched within LOOKAHEAD bars. If both touch in the same bar, SL wins.
    """
    n = len(high_arr)
    for j in range(start_idx + 1, min(start_idx + 1 + LOOKAHEAD, n)):
        if direction == "long":
            sl_touched = low_arr[j] <= sl_price
            tp_touched = high_arr[j] >= tp_price
        else:
            sl_touched = high_arr[j] >= sl_price
            tp_touched = low_arr[j] <= tp_price

        if sl_touched and tp_touched:
            return 0.0
        if sl_touched:
            return 0.0
        if tp_touched:
            return 1.0

    return np.nan


def generate_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generate independent long and short labels for each bar.

    target_long:
        1 if a long entry at Close reaches long TP before long SL.

    target_short:
        1 if a short entry at Close reaches short TP before short SL.
    """
    labels = pd.DataFrame(index=df.index, columns=["target_long", "target_short"], dtype=float)

    close_arr = df["Close"].values
    high_arr = df["High"].values
    low_arr = df["Low"].values
    atr_arr = df["ATR"].values

    for i in range(len(df) - 1):
        atr = atr_arr[i]
        if np.isnan(atr) or atr <= 0:
            continue

        entry = close_arr[i]
        sl_dist = atr * ATR_MULTIPLIER
        tp_dist = sl_dist * REWARD_RISK_RATIO

        labels.iat[i, labels.columns.get_loc("target_long")] = _first_touch_label(
            high_arr=high_arr,
            low_arr=low_arr,
            start_idx=i,
            sl_price=entry - sl_dist,
            tp_price=entry + tp_dist,
            direction="long",
        )
        labels.iat[i, labels.columns.get_loc("target_short")] = _first_touch_label(
            high_arr=high_arr,
            low_arr=low_arr,
            start_idx=i,
            sl_price=entry + sl_dist,
            tp_price=entry - tp_dist,
            direction="short",
        )

    for col in labels.columns:
        valid = labels[col].dropna()
        pos = int((valid == 1).sum())
        neg = int((valid == 0).sum())
        total = pos + neg
        pos_pct = (pos / total * 100.0) if total else 0.0
        logger.info(
            "%s labels | valid=%d | TP(1)=%d (%.1f%%) | SL(0)=%d",
            col,
            total,
            pos,
            pos_pct,
            neg,
        )

    return labels


def build_feature_matrix(
    df: pd.DataFrame, feature_engine: FeatureEngine, valid_indices: list[int]
) -> pd.DataFrame:
    rows = []
    skipped = 0
    logger.info("Building feature matrix for %d bars...", len(valid_indices))
    t0 = time.time()

    for idx in valid_indices:
        feat = feature_engine.generate_live_features(df, idx)
        if feat is None or feat.empty:
            skipped += 1
            continue
        rows.append(feat)

    if not rows:
        raise ValueError("Feature matrix is empty; data or warm-up period is insufficient.")

    logger.info(
        "Feature matrix ready | rows=%d | skipped=%d | elapsed=%.1fs",
        len(rows),
        skipped,
        time.time() - t0,
    )
    return pd.concat(rows, axis=0)


def chronological_split(
    X: pd.DataFrame, y: pd.Series,
    train_ratio: float = 0.70,
    val_ratio: float   = 0.15,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    """Kronolojik 3'lü split: train / val / test (sızıntısız)."""
    n = len(X)
    t = int(n * train_ratio)
    v = int(n * (train_ratio + val_ratio))

    X_train, y_train = X.iloc[:t],    y.iloc[:t]
    X_val,   y_val   = X.iloc[t:v],   y.iloc[t:v]
    X_test,  y_test  = X.iloc[v:],    y.iloc[v:]

    logger.info(
        "3-way split | train=%d (%s→%s) | val=%d (%s→%s) | test=%d (%s→%s)",
        len(X_train), X_train.index[0], X_train.index[-1],
        len(X_val),   X_val.index[0],   X_val.index[-1],
        len(X_test),  X_test.index[0],  X_test.index[-1],
    )
    return X_train, X_val, X_test, y_train, y_val, y_test


def train_xgboost(
    name: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_val: pd.DataFrame,
    y_val: pd.Series,
) -> xgb.XGBClassifier:
    """
    Üç parçalı split: train (70%) → val (15%) → test (15%).
    Early stopping yalnızca val setine bakar; test seti hiç görülmez.
    Bu yapı validation sızıntısını tamamen engeller.
    """
    if y_train.nunique() < 2:
        raise ValueError(f"{name}: training target has only one class.")

    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    scale_pos_weight = neg / pos if pos > 0 else 1.0
    logger.info(
        "%s | SL(0)=%d | TP(1)=%d | scale_pos_weight=%.3f",
        name, neg, pos, scale_pos_weight,
    )

    # Basit model: max_depth=3, güçlü regularizasyon → overfitting engellenir
    model = xgb.XGBClassifier(
        n_estimators=500,
        max_depth=3,
        learning_rate=0.03,
        subsample=0.70,
        colsample_bytree=0.70,
        colsample_bylevel=0.70,
        min_child_weight=20,
        gamma=0.5,
        reg_alpha=0.5,
        reg_lambda=3.0,
        scale_pos_weight=scale_pos_weight,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        early_stopping_rounds=50,
        verbosity=1,
    )

    logger.info("Training %s model...", name)
    t0 = time.time()
    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],   # sadece val seti — test sızdırmaz
        verbose=100,
    )
    logger.info(
        "%s done | %.1fs | best_iter=%s",
        name, time.time() - t0, getattr(model, "best_iteration", "?"),
    )
    return model


def print_metrics(name: str, model: xgb.XGBClassifier, X_test: pd.DataFrame, y_test: pd.Series) -> None:
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred_proba) if y_test.nunique() > 1 else 0.0

    logger.info(
        "\n%s model test metrics\nsamples=%d | accuracy=%.4f | roc_auc=%.4f\n%s",
        name,
        len(y_test),
        acc,
        roc_auc,
        classification_report(y_test, y_pred, labels=[0, 1], target_names=["SL(0)", "TP(1)"], zero_division=0),
    )

    fi = pd.Series(model.feature_importances_, index=X_test.columns).sort_values(ascending=False).head(10)
    logger.info("%s top features:", name)
    for feat, score in fi.items():
        logger.info("  %-28s %.4f", feat, score)


def save_models(models: Dict[str, xgb.XGBClassifier]) -> Dict[str, str]:
    os.makedirs(MODEL_DIR, exist_ok=True)
    saved: Dict[str, str] = {}
    for side, model in models.items():
        save_path = os.path.join(MODEL_DIR, MODEL_FILES[side])
        model.save_model(save_path)
        saved[side] = save_path
        logger.info("Saved %s model -> %s (%.1f KB)", side, save_path, os.path.getsize(save_path) / 1024)
    return saved


def run_training() -> None:
    logger.info("=" * 64)
    logger.info("Dual-direction model training started")
    logger.info("=" * 64)

    data_feed = DataFeed()
    feature_engine = FeatureEngine()

    raw_df = data_feed.download_historical_data()
    df = feature_engine.calculate_indicators(raw_df)
    df = _apply_training_window(df)
    logger.info("Bars available for training: %d", len(df))

    labels = generate_labels(df)
    valid_mask = labels["target_long"].notna() | labels["target_short"].notna()
    valid_indices = [df.index.get_loc(ts) for ts in labels.index[valid_mask]]

    X_all = build_feature_matrix(df, feature_engine, valid_indices)
    X_all.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_all = X_all.dropna(axis=0)
    labels = labels.loc[X_all.index]

    logger.info("Clean shared feature matrix: %d samples x %d features", len(X_all), X_all.shape[1])

    models: Dict[str, xgb.XGBClassifier] = {}
    target_map = {"long": "target_long", "short": "target_short"}
    for side, target_col in target_map.items():
        y_all = labels[target_col].dropna()
        common_idx = X_all.index.intersection(y_all.index)
        X_side = X_all.loc[common_idx]
        y_side = y_all.loc[common_idx].astype(int)

        X_train, X_val, X_test, y_train, y_val, y_test = chronological_split(X_side, y_side)
        model = train_xgboost(side, X_train, y_train, X_val, y_val)
        print_metrics(side, model, X_test, y_test)   # test hiç görülmemiş veridir
        models[side] = model

    save_models(models)
    logger.info("Dual-direction training finished successfully.")


if __name__ == "__main__":
    run_training()
