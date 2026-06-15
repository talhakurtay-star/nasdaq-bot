"""
train_now.py — MT5 olmadan CSV'den eğitim çalıştırıcı.
IS/OOS ayrımı otomatik: backtest son BACKTEST_DAYS günü kullanır,
eğitim ondan önceki tüm veriyi kullanır → veri karışmaz.
Çalıştır: python train_now.py
"""
import logging
import os
import sys
import time
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
import xgboost as xgb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import ATR_MULTIPLIER, MODEL_DIR, REWARD_RISK_RATIO
from ml.features import FeatureEngine
from ml.labeling import generate_labels, LOOKAHEAD

BACKTEST_DAYS     = int(os.getenv("BACKTEST_DAYS", "360"))
TRAIN_WINDOW_DAYS = int(os.getenv("TRAIN_WINDOW_DAYS", "180"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("train_now")

TRAIN_RATIO   = 0.80
RANDOM_STATE  = 42
MODEL_FILES   = {"long": "xgb_model_long.json", "short": "xgb_model_short.json"}


def load_training_data() -> pd.DataFrame:
    """
    csv/nasdaq_15m.csv (MT5 formatı) yükler ve IS/OOS ayrımı uygular.
    Backtest: son BACKTEST_DAYS gün  → model bu veriyi HİÇ görmez.
    Eğitim  : backtest başlangıcından önceki TÜM veri.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(base_dir, "csv", "nasdaq_15m.csv")
    if not os.path.exists(csv_file):
        raise FileNotFoundError(f"CSV bulunamadı: {csv_file}")

    df = pd.read_csv(
        csv_file,
        sep="\t",
        names=["Date", "Time", "Open", "High", "Low", "Close", "TickVol", "Vol", "Spread"],
        skiprows=1,
        dtype=str,
    )
    df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%Y.%m.%d %H:%M:%S")
    df = df.set_index("Datetime").drop(columns=["Date", "Time", "Vol", "Spread"])
    df = df.rename(columns={"Open": "Open", "High": "High", "Low": "Low",
                             "Close": "Close", "TickVol": "Volume"})
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df.sort_index()

    max_date = df.index.max()
    backtest_start = max_date - timedelta(days=BACKTEST_DAYS)
    train_start    = backtest_start - timedelta(days=TRAIN_WINDOW_DAYS)
    df_train = df[(df.index >= train_start) & (df.index < backtest_start)].copy()

    logger.info("Tam CSV      : %d bar | %s → %s", len(df), df.index[0], max_date)
    logger.info("Backtest OOS : %s → %s (%d gün) — eğitimden DIŞLANDI",
                backtest_start.date(), max_date.date(), BACKTEST_DAYS)
    logger.info("Eğitim IS    : %d bar | %s → %s  (son %d gün)",
                len(df_train), df_train.index[0], df_train.index[-1], TRAIN_WINDOW_DAYS)
    return df_train


def _log_label_stats(labels):
    for col in labels.columns:
        valid = labels[col].dropna()
        pos   = int((valid == 1).sum())
        neg   = int((valid == 0).sum())
        total = pos + neg
        logger.info("%s | toplam=%d | TP(1)=%d (%0.1f%%) | SL(0)=%d",
                    col, total, pos, pos / total * 100 if total else 0, neg)


def build_feature_matrix(df, feature_engine, valid_indices):
    rows = []
    skipped = 0
    logger.info("Feature matrix oluşturuluyor (%d bar)...", len(valid_indices))
    t0 = time.time()
    for idx in valid_indices:
        feat = feature_engine.generate_live_features(df, idx)
        if feat is None or feat.empty:
            skipped += 1
            continue
        rows.append(feat)
    if not rows:
        raise ValueError("Feature matrix boş — veri yetersiz.")
    logger.info("Feature matrix hazır | rows=%d | skipped=%d | %.1fs",
                len(rows), skipped, time.time() - t0)
    return pd.concat(rows, axis=0)


def train_model(name, X_train, y_train, X_val, y_val):
    neg = int((y_train == 0).sum())
    pos = int((y_train == 1).sum())
    spw = neg / pos if pos > 0 else 1.0
    logger.info("%s | SL(0)=%d | TP(1)=%d | scale_pos_weight=%.3f", name, neg, pos, spw)

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
        scale_pos_weight=spw,
        objective="binary:logistic",
        eval_metric="auc",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        early_stopping_rounds=50,
        verbosity=1,
    )
    model.fit(X_train, y_train,
              eval_set=[(X_val, y_val)],
              verbose=100)
    return model


def print_metrics(name, model, X_test, y_test):
    proba = model.predict_proba(X_test)[:, 1]
    pred  = (proba >= 0.5).astype(int)
    acc   = accuracy_score(y_test, pred)
    auc   = roc_auc_score(y_test, proba) if y_test.nunique() > 1 else 0.0
    logger.info("\n%s | samples=%d | accuracy=%.4f | roc_auc=%.4f\n%s",
                name, len(y_test), acc, auc,
                classification_report(y_test, pred, labels=[0,1],
                                      target_names=["SL(0)","TP(1)"], zero_division=0))
    fi = pd.Series(model.feature_importances_, index=X_test.columns).sort_values(ascending=False).head(10)
    logger.info("%s top features:", name)
    for feat, score in fi.items():
        logger.info("  %-30s %.4f", feat, score)


def main():
    logger.info("=" * 64)
    logger.info("Eğitim başlıyor (IS/OOS otomatik ayrım, MT5 gereksiz)")
    logger.info("Eğitim penceresi: backtest'ten önceki son %d gün", TRAIN_WINDOW_DAYS)
    logger.info("=" * 64)

    feature_engine = FeatureEngine()
    raw_df = load_training_data()
    df = feature_engine.calculate_indicators(raw_df)
    logger.info("İndikatörler hesaplandı: %d bar x %d sütun", len(df), len(df.columns))

    labels = generate_labels(df, ATR_MULTIPLIER, REWARD_RISK_RATIO)
    _log_label_stats(labels)
    valid_mask    = labels["target_long"].notna() | labels["target_short"].notna()
    valid_indices = [df.index.get_loc(ts) for ts in labels.index[valid_mask]]

    X_all = build_feature_matrix(df, feature_engine, valid_indices)
    X_all.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_all.dropna(axis=0, inplace=True)
    labels = labels.loc[X_all.index]
    logger.info("Temiz feature matrix: %d sample x %d feature", len(X_all), X_all.shape[1])

    os.makedirs(MODEL_DIR, exist_ok=True)
    target_map = {"long": "target_long", "short": "target_short"}

    for side, col in target_map.items():
        y_all      = labels[col].dropna()
        common_idx = X_all.index.intersection(y_all.index)
        X_side     = X_all.loc[common_idx]
        y_side     = y_all.loc[common_idx].astype(int)

        n = len(X_side)
        t, v = int(n * 0.70), int(n * 0.85)
        X_train, y_train = X_side.iloc[:t],  y_side.iloc[:t]
        X_val,   y_val   = X_side.iloc[t:v], y_side.iloc[t:v]
        X_test,  y_test  = X_side.iloc[v:],  y_side.iloc[v:]
        logger.info("%s | train=%d | val=%d | test=%d", side, len(X_train), len(X_val), len(X_test))

        model = train_model(side, X_train, y_train, X_val, y_val)
        print_metrics(side, model, X_test, y_test)  # test hiç görülmemiş veri

        # Probability calibration: val seti üzerinde isotonic regression
        # XGB ham prob → gerçek frekans eşleştirmesi → dinamik risk güvenilir olur
        import pickle
        val_proba = model.predict_proba(X_val)[:, 1]
        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(val_proba, y_val.values)

        # Test kalitesi
        test_raw  = model.predict_proba(X_test)[:, 1]
        test_cal  = iso.predict(test_raw)
        raw_auc = roc_auc_score(y_test, test_raw) if y_test.nunique() > 1 else 0.0
        cal_auc = roc_auc_score(y_test, test_cal) if y_test.nunique() > 1 else 0.0
        logger.info("%s | Ham AUC=%.4f → Kalibre AUC=%.4f", side, raw_auc, cal_auc)

        path = os.path.join(MODEL_DIR, MODEL_FILES[side])
        model.save_model(path)

        cal_path = path.replace(".json", "_calibrated.pkl")
        with open(cal_path, "wb") as f:
            pickle.dump({"iso": iso, "side": side}, f)
        logger.info("Model kaydedildi → %s | Kalibrasyon → %s", path, cal_path)

    logger.info("=" * 64)
    logger.info("Eğitim tamamlandı!")
    logger.info("=" * 64)


if __name__ == "__main__":
    main()
