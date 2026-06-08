"""
data/data_feed.py
-----------------
Cache tabanlı ve CSV tabanlı veri besleyici.

USE_CSV_DATA = True  → csv/nasdaq_15m.csv okunur (MT5 export formatı desteklenir).
USE_CSV_DATA = False → cache/cache_15m_360d.csv okunur (eski davranış).

MT5 manuel CSV formatı:
  <DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<TICKVOL>,<VOL>,<SPREAD>
  veya
  Date,Time,Open,High,Low,Close,Volume
"""

import os
import pandas as pd
from datetime import timedelta

try:
    from config.settings import (
        TIMEFRAME, BACKTEST_DAYS, CACHE_DIR,
        USE_CSV_DATA, CSV_DIR, CSV_FILE_NAME,
    )
except ImportError:
    TIMEFRAME     = "15m"
    BACKTEST_DAYS = 360
    CACHE_DIR     = "cache"
    USE_CSV_DATA  = True
    CSV_DIR       = "csv"
    CSV_FILE_NAME = "nasdaq_15m.csv"

_BASE_DIR     = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
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
        return df

    agg = {"Open": "first", "High": "max", "Low": "min", "Close": "last", "Volume": "sum"}
    for col in ("SPY_Close", "VIX_Close"):
        if col in df.columns:
            agg[col] = "last"

    resampled = df.resample(rule).agg(agg).dropna(subset=["Close"])
    print(f"[DataFeed] Resample: 15m → {tf} | {len(df)} → {len(resampled)} bar")
    return resampled


def _ensure_spy_vix(df: pd.DataFrame) -> pd.DataFrame:
    """
    XGBoost modeli SPY_Close ve VIX_Close sütunlarını bekler.
    CSV'de yoksa proxy değerlerle doldurur:
      SPY_Close → Close (fiyat hareketi korelasyonu için yeterli)
      VIX_Close → 15.0  (nötr/ortalama volatilite varsayımı)
    """
    if "SPY_Close" not in df.columns:
        df = df.copy()
        df["SPY_Close"] = df["Close"]
        print("[DataFeed] SPY_Close sütunu yok → Close kopyalandı.")
    if "VIX_Close" not in df.columns:
        df = df.copy()
        df["VIX_Close"] = 15.0
        print("[DataFeed] VIX_Close sütunu yok → sabit 15.0 atandı.")
    return df


def _load_mt5_csv(path: str) -> pd.DataFrame:
    """
    MT5 manual-export CSV'sini okur.

    Desteklenen başlık formatları:
      1) <DATE>,<TIME>,<OPEN>,<HIGH>,<LOW>,<CLOSE>,<TICKVOL>,...
      2) Date,Time,Open,High,Low,Close,Volume,...
      3) Datetime tek sütun halinde (ISO 8601)
      4) Noktalı virgül (;) ayraçlı MT5 formatı
    """
    # Ayraç otomatik tespiti
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        first_line = f.readline()
    sep = ";" if first_line.count(";") > first_line.count(",") else ","

    df = pd.read_csv(path, header=0, sep=sep)

    # Sütun isimlerini temizle: boşluk + <> kaldır, büyük harf yap
    df.columns = [c.strip().strip("<>").upper() for c in df.columns]

    print(f"[DataFeed] CSV sütunları: {list(df.columns)}")

    # DATE + TIME sütunlarını birleştir
    if "DATE" in df.columns and "TIME" in df.columns:
        df["Datetime"] = pd.to_datetime(
            df["DATE"].astype(str) + " " + df["TIME"].astype(str),
            dayfirst=False,
            errors="coerce",
        )
        df = df.drop(columns=["DATE", "TIME"])
    elif "DATETIME" in df.columns:
        df["Datetime"] = pd.to_datetime(df["DATETIME"], errors="coerce")
        df = df.drop(columns=["DATETIME"])
    else:
        # İlk sütunu datetime olarak dene
        first_col = df.columns[0]
        df["Datetime"] = pd.to_datetime(df[first_col], errors="coerce")
        df = df.drop(columns=[first_col])

    df = df.dropna(subset=["Datetime"])
    df = df.set_index("Datetime")
    df.index = pd.DatetimeIndex(df.index)

    # Sütunları standart OHLCV isimlerine map'le (her olası varyant dahil)
    rename_map = {
        "OPEN":    "Open",
        "HIGH":    "High",
        "LOW":     "Low",
        "CLOSE":   "Close",
        "TICKVOL": "Volume",
        "VOL":     "Volume",
        "VOLUME":  "Volume",
        "REAL_VOLUME": "Volume",
        # yfinance-style
        "ADJ CLOSE": "Close",
    }
    df = df.rename(columns={c: rename_map[c] for c in list(df.columns) if c in rename_map})

    # Sütun bulunamadıysa pozisyon bazlı atama (son çare)
    ohlc = ["Open", "High", "Low", "Close"]
    missing = [c for c in ohlc if c not in df.columns]
    if missing:
        numeric_cols = [c for c in df.columns if pd.api.types.is_numeric_dtype(df[c]) or
                        pd.to_numeric(df[c], errors="coerce").notna().mean() > 0.8]
        print(f"[DataFeed] UYARI: {missing} sütunları bulunamadı, "
              f"numerik sütunlar pozisyon bazlı atanıyor: {numeric_cols[:4]}")
        for i, col in enumerate(ohlc):
            if col not in df.columns and i < len(numeric_cols):
                df = df.rename(columns={numeric_cols[i]: col})

    # Volume sütunu yoksa sıfırla oluştur
    if "Volume" not in df.columns:
        df["Volume"] = 0

    # OHLCV sütunlarını sayısala çevir
    for col in ("Open", "High", "Low", "Close", "Volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Hâlâ eksik sütun varsa açıklayıcı hata ver
    missing_final = [c for c in ohlc if c not in df.columns]
    if missing_final:
        raise KeyError(
            f"OHLC sütunları bulunamadı: {missing_final}\n"
            f"CSV'deki sütunlar: {list(df.columns)}\n"
            f"CSV dosyasının ilk satırını kontrol edin: {path}"
        )

    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    df = df.sort_index()  # kronolojik sıra

    return df


class DataFeed:
    def __init__(self, cache_file: str | None = None):
        os.makedirs(CACHE_DIR, exist_ok=True)
        os.makedirs(CSV_DIR,   exist_ok=True)
        self._cache_file = cache_file  # None → USE_CSV_DATA flag'ine göre karar verilir

    def download_historical_data(self, force_refresh: bool = False) -> pd.DataFrame:
        if USE_CSV_DATA and self._cache_file is None:
            return self._load_from_csv()
        else:
            return self._load_from_cache()

    # ------------------------------------------------------------------

    def _load_from_csv(self) -> pd.DataFrame:
        csv_path = os.path.join(CSV_DIR, CSV_FILE_NAME)

        if not os.path.exists(csv_path):
            raise FileNotFoundError(
                f"\n{'='*60}\n"
                f"CSV dosyası bulunamadı: {csv_path}\n"
                f"Lütfen MT5'ten export ettiğiniz '{CSV_FILE_NAME}' dosyasını\n"
                f"şu klasöre koyun:\n  {CSV_DIR}\n"
                f"{'='*60}\n"
                "MT5'ten export: Grafik → Sağ tık → 'Veriyi Kaydet' → CSV\n"
                "Sembol: NAS100 / USTEC / US100  |  Periyot: M15"
            )

        df = _load_mt5_csv(csv_path)
        print(
            f"[DataFeed] CSV yüklendi: {os.path.basename(csv_path)} | "
            f"{len(df)} bar | {df.index[0]} → {df.index[-1]}"
        )

        df = _ensure_spy_vix(df)
        df = _resample(df, TIMEFRAME)

        cutoff = df.index.max() - timedelta(days=BACKTEST_DAYS)
        df_out = df.loc[df.index > cutoff].copy()
        if len(df_out) == 0:
            print("[DataFeed] UYARI: Dilim boş, tüm veri kullanılıyor.")
            df_out = df.copy()
        else:
            print(f"[DataFeed] Son {BACKTEST_DAYS} gün: {len(df_out)} bar")

        return df_out

    def _load_from_cache(self) -> pd.DataFrame:
        cache_path = self._cache_file if self._cache_file else _MASTER_CACHE

        if not os.path.exists(cache_path):
            raise FileNotFoundError(
                f"Cache dosyası bulunamadı: {cache_path}\n"
                "Dosyanın mevcut olduğunu kontrol edin."
            )

        df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        print(
            f"[DataFeed] Cache ({os.path.basename(cache_path)}): "
            f"{len(df)} bar ({df.index[0]} → {df.index[-1]})"
        )

        df = _ensure_spy_vix(df)
        df = _resample(df, TIMEFRAME)

        cutoff = df.index.max() - timedelta(days=BACKTEST_DAYS)
        df_out = df.loc[df.index > cutoff].copy()

        if len(df_out) == 0:
            print("[DataFeed] UYARI: Dilim boş, tüm veri kullanılıyor.")
            df_out = df.copy()
        elif BACKTEST_DAYS > 360:
            print(
                f"[DataFeed] UYARI: BACKTEST_DAYS={BACKTEST_DAYS} > mevcut 360 gün "
                f"— {len(df_out)} bar kullanılıyor."
            )
        else:
            print(f"[DataFeed] Son {BACKTEST_DAYS} gün: {len(df_out)} bar")

        return df_out
