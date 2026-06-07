"""
data/data_feed.py
-----------------
Cache tabanlı veri besleyici (MT5 gerektirmez).

- cache_15m_360d.csv master kaynak olarak kullanılır.
- BACKTEST_DAYS ayarına göre son N günü kırpar.
- TIMEFRAME farklıysa (1h, 5m vb.) 15m verisini resample eder.
"""

import os
import pandas as pd
from datetime import timedelta

try:
    from config.settings import (
        TIMEFRAME, BACKTEST_DAYS, CACHE_DIR
    )
except ImportError:
    TIMEFRAME     = "15m"
    BACKTEST_DAYS = 360
    CACHE_DIR     = "cache"

_BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MASTER_CACHE = os.path.join(_BASE_DIR, "cache", "cache_15m_360d.csv")

_TF_MAP = {
    "1m":  "1min",
    "5m":  "5min",
    "15m": "15min",
    "30m": "30min",
    "1h":  "1h",
    "4h":  "4h",
    "1d":  "1D",
}


def _resample(df: pd.DataFrame, tf: str) -> pd.DataFrame:
    """15m OHLCV DataFrame'ini istenen periyoda resample eder."""
    rule = _TF_MAP.get(tf, "15min")
    if rule == "15min":
        return df  # zaten doğru periyot

    agg = {
        "Open":      "first",
        "High":      "max",
        "Low":       "min",
        "Close":     "last",
        "Volume":    "sum",
    }
    for col in ("SPY_Close", "VIX_Close"):
        if col in df.columns:
            agg[col] = "last"

    resampled = df.resample(rule).agg(agg).dropna(subset=["Close"])
    print(f"[DataFeed] Resample: 15m → {tf} | {len(df)} → {len(resampled)} bar")
    return resampled


class DataFeed:
    def __init__(self):
        os.makedirs(CACHE_DIR, exist_ok=True)

    def download_historical_data(self, force_refresh: bool = False) -> pd.DataFrame:
        if not os.path.exists(_MASTER_CACHE):
            raise FileNotFoundError(
                f"Master cache bulunamadı: {_MASTER_CACHE}\n"
                "cache/cache_15m_360d.csv dosyasının mevcut olduğunu kontrol edin."
            )

        df = pd.read_csv(_MASTER_CACHE, index_col=0, parse_dates=True)
        print(f"[DataFeed] Master cache: {len(df)} bar "
              f"({df.index[0]} → {df.index[-1]})")

        # Zaman dilimine göre resample
        df = _resample(df, TIMEFRAME)

        # BACKTEST_DAYS kadar son dilimi al
        cutoff = df.index.max() - timedelta(days=BACKTEST_DAYS)
        df_out = df.loc[df.index > cutoff].copy()

        if len(df_out) == 0:
            print(f"[DataFeed] UYARI: Dilim boş, tüm veri kullanılıyor.")
            df_out = df.copy()
        elif BACKTEST_DAYS > 360:
            print(f"[DataFeed] UYARI: BACKTEST_DAYS={BACKTEST_DAYS} > mevcut 360 gün "
                  f"— {len(df_out)} bar kullanılıyor.")
        else:
            print(f"[DataFeed] Son {BACKTEST_DAYS} gün: {len(df_out)} bar")

        return df_out
