"""
ml/labeling.py — Ortak etiketleme fonksiyonları.
train_now.py ve walkforward.py bu modülü paylaşır.
"""
import numpy as np

LOOKAHEAD = 25


def first_touch_label(
    high_arr: np.ndarray,
    low_arr: np.ndarray,
    start_idx: int,
    sl_price: float,
    tp_price: float,
    direction: str,
) -> float:
    """
    Bar dizisinde SL veya TP'ye ilk temas eden tarafı döndürür.

    Returns
    -------
    1.0  → TP ilk temas (kazanç)
    0.0  → SL ilk temas veya aynı barda çelişki (kayıp, muhafazakâr)
    nan  → LOOKAHEAD içinde hiçbiri tetiklenmedi
    """
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


def generate_labels(df, atr_multiplier: float, reward_risk_ratio: float):
    """
    Her bar için long ve short etiketleri üretir.
    Son LOOKAHEAD bar dahil edilmez (label sızıntısını önler).
    """
    import pandas as pd
    labels = pd.DataFrame(index=df.index, columns=["target_long", "target_short"], dtype=float)
    close_arr = df["Close"].values
    high_arr  = df["High"].values
    low_arr   = df["Low"].values
    atr_arr   = df["ATR"].values

    # Son LOOKAHEAD barı atla — bu barların label'ları bir sonraki pencereye uzanır
    safe_end = len(df) - LOOKAHEAD - 1
    for i in range(safe_end):
        atr = atr_arr[i]
        if np.isnan(atr) or atr <= 0:
            continue
        entry   = close_arr[i]
        sl_dist = atr * atr_multiplier
        tp_dist = sl_dist * reward_risk_ratio

        labels.iat[i, 0] = first_touch_label(high_arr, low_arr, i, entry - sl_dist, entry + tp_dist, "long")
        labels.iat[i, 1] = first_touch_label(high_arr, low_arr, i, entry + sl_dist, entry - tp_dist, "short")

    return labels
