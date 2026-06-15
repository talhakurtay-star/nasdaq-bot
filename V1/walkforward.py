"""
walkforward.py — Walk-Forward Backtest
Her çeyrek için model yeniden eğitilir, OOS performansı birleştirilir.
Çalıştır: python walkforward.py

Pencere şeması (BACKTEST_DAYS=360):
  Çeyrek 1: eğitim=T-360→T-270, test=T-270→T-180 arası
  Çeyrek 2: eğitim=T-270→T-180, test=T-180→T-90 arası
  Çeyrek 3: eğitim=T-180→T-90,  test=T-90→T-0 arası
  Not: Her çeyrek 6 aylık eğitim verisi kullanır (180 gün)
"""
import logging
import os
import sys
import subprocess
import time
from datetime import timedelta

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, roc_auc_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import ATR_MULTIPLIER, MODEL_DIR, REWARD_RISK_RATIO
from ml.features import FeatureEngine
from ml.labeling import generate_labels, LOOKAHEAD

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("walkforward")

BACKTEST_DAYS     = int(os.getenv("BACKTEST_DAYS", "360"))
TRAIN_WINDOW_DAYS = int(os.getenv("TRAIN_WINDOW_DAYS", "180"))
RANDOM_STATE      = 42
MODEL_FILES       = {"long": "xgb_model_long.json", "short": "xgb_model_short.json"}
N_QUARTERS        = 3  # 360 gün → 3 x 120 gün çeyrek


def load_full_csv() -> pd.DataFrame:
    from data.data_feed import _load_mt5_csv, _ensure_spy_vix
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(base_dir, "csv", "nasdaq_15m.csv")
    df = _load_mt5_csv(csv_file)
    df = _ensure_spy_vix(df)
    return df


def train_quarter_model(df_train_raw, feature_engine):
    df = feature_engine.calculate_indicators(df_train_raw)
    labels = generate_labels(df, ATR_MULTIPLIER, REWARD_RISK_RATIO)
    valid_mask    = labels["target_long"].notna() | labels["target_short"].notna()
    valid_indices = [df.index.get_loc(ts) for ts in labels.index[valid_mask]]

    rows = []
    for idx in valid_indices:
        feat = feature_engine.generate_live_features(df, idx)
        if feat is not None and not feat.empty:
            rows.append(feat)
    if not rows:
        raise ValueError("Feature matrix boş.")

    X_all = pd.concat(rows, axis=0)
    X_all.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_all.dropna(axis=0, inplace=True)
    labels = labels.loc[X_all.index]

    os.makedirs(MODEL_DIR, exist_ok=True)
    target_map = {"long": "target_long", "short": "target_short"}

    for side, col in target_map.items():
        y_all      = labels[col].dropna()
        common_idx = X_all.index.intersection(y_all.index)
        X_side     = X_all.loc[common_idx]
        y_side     = y_all.loc[common_idx].astype(int)

        n = len(X_side)
        t = int(n * 0.80)
        X_train, y_train = X_side.iloc[:t], y_side.iloc[:t]
        X_val,   y_val   = X_side.iloc[t:], y_side.iloc[t:]

        neg = int((y_train == 0).sum())
        pos = int((y_train == 1).sum())
        spw = neg / pos if pos > 0 else 1.0

        model = xgb.XGBClassifier(
            n_estimators=500, max_depth=3, learning_rate=0.03,
            subsample=0.70, colsample_bytree=0.70, colsample_bylevel=0.70,
            min_child_weight=20, gamma=0.5, reg_alpha=0.5, reg_lambda=3.0,
            scale_pos_weight=spw, objective="binary:logistic", eval_metric="auc",
            random_state=RANDOM_STATE, n_jobs=-1, early_stopping_rounds=50, verbosity=0,
        )
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
        model.save_model(os.path.join(MODEL_DIR, MODEL_FILES[side]))

        if len(y_val) > 1 and y_val.nunique() > 1:
            proba = model.predict_proba(X_val)[:, 1]
            auc   = roc_auc_score(y_val, proba)
            logger.info("  %s | train=%d | val=%d | AUC=%.4f", side, len(X_train), len(X_val), auc)


def run_quarter_backtest(date_from: str, date_to: str, xgb_threshold: float) -> dict:
    """Backtest çalıştırır ve temp_metrics.json üzerinden sonuçları okur."""
    import json

    base_dir  = os.path.dirname(os.path.abspath(__file__))
    json_path = os.path.join(base_dir, "temp_metrics.json")

    # Eski dosyayı sil — yeni çalıştırma yazana kadar karışmasın
    try:
        os.remove(json_path)
    except FileNotFoundError:
        pass

    env = os.environ.copy()
    env["SIM_DATE_FROM"]        = date_from
    env["SIM_DATE_TO"]          = date_to
    env["STRESS_XGB_THRESHOLD"] = str(xgb_threshold)
    env["WALKFORWARD_QUIET"]    = "1"

    subprocess.run(
        [sys.executable, os.path.join(base_dir, "backtest.py")],
        capture_output=True, text=True, env=env,
        cwd=base_dir,
    )

    # temp_metrics.json'dan oku
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            raw = json.load(f)
        return {
            "pnl_pct":    raw.get("net_pnl_pct", 0.0),
            "end_balance": 100_000 * (1 + raw.get("net_pnl_pct", 0.0) / 100),
            "win_rate":   raw.get("win_rate_pct", 0.0),
            "trades":     raw.get("toplam_islem", 0),
            "max_dd":     abs(raw.get("maks_drawdown_pct", 0.0)),
        }
    except Exception as exc:
        logger.error("temp_metrics.json okunamadı: %s", exc)
        return {}


def main():
    logger.info("=" * 64)
    logger.info("WALK-FORWARD BACKTEST BAŞLIYOR")
    logger.info("Çeyrek sayısı: %d | Eğitim penceresi: %d gün", N_QUARTERS, TRAIN_WINDOW_DAYS)
    logger.info("=" * 64)

    feature_engine = FeatureEngine()
    df_full = load_full_csv()

    max_date       = df_full.index.max()
    backtest_start = max_date - timedelta(days=BACKTEST_DAYS)
    quarter_days   = BACKTEST_DAYS // N_QUARTERS

    logger.info("Backtest dönemi: %s → %s", backtest_start.date(), max_date.date())

    compound_balance = 100_000.0  # zincirleme bakiye takibi
    total_trades     = 0
    quarter_results  = []

    for q in range(N_QUARTERS):
        q_start = backtest_start + timedelta(days=q * quarter_days)
        q_end   = backtest_start + timedelta(days=(q + 1) * quarter_days)
        if q == N_QUARTERS - 1:
            q_end = max_date + timedelta(days=1)

        train_end   = q_start
        train_start = train_end - timedelta(days=TRAIN_WINDOW_DAYS)

        logger.info("")
        logger.info("── Çeyrek %d/%d ──────────────────────────────────", q + 1, N_QUARTERS)
        logger.info("  Eğitim : %s → %s", train_start.date(), train_end.date())
        logger.info("  Test   : %s → %s", q_start.date(), q_end.date())

        # Eğitim verisi hazırla ve modeli eğit
        df_train = df_full[(df_full.index >= train_start) & (df_full.index < train_end)].copy()
        if len(df_train) < 1000:
            logger.warning("  Eğitim verisi yetersiz (%d bar), XGB devre dışı", len(df_train))
            xgb_threshold = 0.0
        else:
            logger.info("  Model eğitiliyor (%d bar)...", len(df_train))
            t0 = time.time()
            train_quarter_model(df_train, feature_engine)
            logger.info("  Model hazır (%.1fs)", time.time() - t0)
            xgb_threshold = 0.45  # Dinamik risk alt sınırı (0.45-0.60=0.5x, 0.60-0.70=1x, >0.70=1.2x)

        # Çeyreği backtest et
        metrics = run_quarter_backtest(
            str(q_start.date()), str(q_end.date()), xgb_threshold
        )

        pnl    = metrics.get("pnl_pct", 0.0)
        trades = metrics.get("trades", 0)
        wr     = metrics.get("win_rate", 0.0)
        dd     = metrics.get("max_dd", 0.0)
        compound_balance *= (1 + pnl / 100.0)
        total_trades     += trades
        quarter_results.append(metrics)

        logger.info("  Sonuç  : PnL=%+.2f%% | İşlem=%d | WR=%.1f%% | MaxDD=%.2f%% | Bakiye=%.0f",
                    pnl, trades, wr, dd, compound_balance)

    logger.info("")
    logger.info("=" * 64)
    total_return_pct = (compound_balance / 100_000.0 - 1) * 100
    logger.info("WALK-FORWARD ÖZET")
    logger.info("=" * 64)
    logger.info("  Compound PnL  : %+.2f%% (zincirleme | basit toplam değil)", total_return_pct)
    logger.info("  Bitiş Bakiye  : %.0f (başlangıç 100,000)", compound_balance)
    logger.info("  Toplam İşlem  : %d", total_trades)
    for i, r in enumerate(quarter_results):
        logger.info("  Q%d: %+.2f%% | %d işlem | WR=%.1f%% | MaxDD=%.2f%%",
                    i + 1, r.get("pnl_pct", 0), r.get("trades", 0),
                    r.get("win_rate", 0), r.get("max_dd", 0))
    logger.info("=" * 64)


if __name__ == "__main__":
    main()
