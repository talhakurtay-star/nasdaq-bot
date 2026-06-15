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

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("walkforward")

BACKTEST_DAYS     = int(os.getenv("BACKTEST_DAYS", "360"))
TRAIN_WINDOW_DAYS = int(os.getenv("TRAIN_WINDOW_DAYS", "180"))
LOOKAHEAD         = 25
RANDOM_STATE      = 42
MODEL_FILES       = {"long": "xgb_model_long.json", "short": "xgb_model_short.json"}
N_QUARTERS        = 3  # 360 gün → 3 x 120 gün çeyrek


def load_full_csv() -> pd.DataFrame:
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_file = os.path.join(base_dir, "csv", "nasdaq_15m.csv")
    df = pd.read_csv(
        csv_file, sep="\t",
        names=["Date", "Time", "Open", "High", "Low", "Close", "TickVol", "Vol", "Spread"],
        skiprows=1, dtype=str,
    )
    df["Datetime"] = pd.to_datetime(df["Date"] + " " + df["Time"], format="%Y.%m.%d %H:%M:%S")
    df = df.set_index("Datetime").drop(columns=["Date", "Time", "Vol", "Spread"])
    df = df.rename(columns={"TickVol": "Volume"})
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df.dropna(subset=["Open", "High", "Low", "Close"]).sort_index()


def _first_touch_label(high_arr, low_arr, start_idx, sl_price, tp_price, direction):
    n = len(high_arr)
    for j in range(start_idx + 1, min(start_idx + 1 + LOOKAHEAD, n)):
        if direction == "long":
            sl_hit = low_arr[j] <= sl_price
            tp_hit = high_arr[j] >= tp_price
        else:
            sl_hit = high_arr[j] >= sl_price
            tp_hit = low_arr[j] <= tp_price
        if sl_hit and tp_hit:
            return 0.0
        if sl_hit:
            return 0.0
        if tp_hit:
            return 1.0
    return np.nan


def generate_labels(df):
    labels = pd.DataFrame(index=df.index, columns=["target_long", "target_short"], dtype=float)
    close_arr, high_arr, low_arr, atr_arr = (
        df["Close"].values, df["High"].values, df["Low"].values, df["ATR"].values
    )
    for i in range(len(df) - 1):
        atr = atr_arr[i]
        if np.isnan(atr) or atr <= 0:
            continue
        entry   = close_arr[i]
        sl_dist = atr * ATR_MULTIPLIER
        tp_dist = sl_dist * REWARD_RISK_RATIO
        labels.iat[i, 0] = _first_touch_label(high_arr, low_arr, i, entry - sl_dist, entry + tp_dist, "long")
        labels.iat[i, 1] = _first_touch_label(high_arr, low_arr, i, entry + sl_dist, entry - tp_dist, "short")
    return labels


def train_quarter_model(df_train_raw, feature_engine):
    df = feature_engine.calculate_indicators(df_train_raw)
    labels = generate_labels(df)
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
    env = os.environ.copy()
    env["SIM_DATE_FROM"]      = date_from
    env["SIM_DATE_TO"]        = date_to
    env["STRESS_XGB_THRESHOLD"] = str(xgb_threshold)
    env["WALKFORWARD_QUIET"]  = "1"

    result = subprocess.run(
        [sys.executable, os.path.join(os.path.dirname(__file__), "backtest.py")],
        capture_output=True, text=True, env=env,
        cwd=os.path.dirname(__file__),
    )
    output = result.stdout + result.stderr

    # Parse key metrics from output
    metrics = {}
    for line in output.split("\n"):
        if "Net PnL" in line and "%" in line:
            try:
                pct = float(line.split("(")[1].split("%")[0].replace("+", ""))
                metrics["pnl_pct"] = pct
            except Exception:
                pass
        if "Bitiş Bakiyesi" in line:
            try:
                bal = float(line.split("$")[1].strip().replace(",", "").split()[0])
                metrics["end_balance"] = bal
            except Exception:
                pass
        if "Win Rate" in line:
            try:
                wr = float(line.split(":")[1].strip().replace("%", ""))
                metrics["win_rate"] = wr
            except Exception:
                pass
        if "Toplam İşlem" in line:
            try:
                n = int(line.split(":")[1].strip())
                metrics["trades"] = n
            except Exception:
                pass
        if "Maks. Drawdown" in line and "%" in line:
            try:
                dd = float(line.split("(")[1].split("%")[0])
                metrics["max_dd"] = dd
            except Exception:
                pass

    return metrics


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

    total_pnl_pct = 0.0
    total_trades  = 0
    quarter_results = []

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

        pnl   = metrics.get("pnl_pct", 0.0)
        trades = metrics.get("trades", 0)
        wr    = metrics.get("win_rate", 0.0)
        dd    = metrics.get("max_dd", 0.0)
        total_pnl_pct += pnl
        total_trades  += trades
        quarter_results.append(metrics)

        logger.info("  Sonuç  : PnL=%+.2f%% | İşlem=%d | WR=%.1f%% | MaxDD=%.2f%%",
                    pnl, trades, wr, dd)

    logger.info("")
    logger.info("=" * 64)
    logger.info("WALK-FORWARD ÖZET")
    logger.info("=" * 64)
    logger.info("  Toplam PnL    : %+.2f%% / yıl (~%+.2f%% / ay)",
                total_pnl_pct, total_pnl_pct / 12)
    logger.info("  Toplam İşlem  : %d", total_trades)
    for i, r in enumerate(quarter_results):
        logger.info("  Q%d: %+.2f%% | %d işlem | WR=%.1f%%",
                    i + 1, r.get("pnl_pct", 0), r.get("trades", 0), r.get("win_rate", 0))
    logger.info("=" * 64)


if __name__ == "__main__":
    main()
