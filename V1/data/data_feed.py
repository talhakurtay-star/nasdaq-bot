"""
data/data_feed.py
-----------------
Cache tabanlı, CSV tabanlı ve multi-symbol veri besleyici.

Öncelik sırası:
  1. cache/symbols/{symbol}_15m.csv varsa → multi-symbol mod (yfinance pipeline)
  2. USE_CSV_DATA=True → csv/nasdaq_15m.csv (MT5 export)
  3. cache/cache_15m_360d.csv → eski tekli cache modu
"""

import os
import pandas as pd
from datetime import timedelta

try:
    from config.settings import (
        TIMEFRAME, BACKTEST_DAYS, CACHE_DIR,
        USE_CSV_DATA, CSV_DIR, CSV_FILE_NAME,
        UNIVERSE,
    )
except ImportError:
    TIMEFRAME     = "15m"
    BACKTEST_DAYS = 360
    CACHE_DIR     = "cache"
    USE_CSV_DATA  = True
    CSV_DIR       = "csv"
    CSV_FILE_NAME = "nasdaq_15m.csv"
    UNIVERSE      = []

_BASE_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_MASTER_CACHE  = os.path.join(_BASE_DIR, "cache", "cache_15m_360d.csv")
_SYMBOLS_DIR   = os.path.join(_BASE_DIR, "cache", "symbols")
# MT5 multi-symbol export klasörü (kullanıcının export ettiği CSV'ler burada)
_MT5_MULTI_DIR = os.path.join(_BASE_DIR, "csv", "symbols")

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
    # Ayraç otomatik tespiti (tab, noktalı virgül veya virgül)
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        first_line = f.readline()
    if first_line.count("\t") >= 3:
        sep = "\t"
    elif first_line.count(";") > first_line.count(","):
        sep = ";"
    else:
        sep = ","

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
    # Çift sütun (duplicate) oluşmasını önlemek için seçici davranıyoruz.
    vol_candidates = ["REAL_VOLUME", "VOLUME", "VOL", "TICKVOL"]
    chosen_vol_col = None
    for c in vol_candidates:
        if c in df.columns:
            try:
                col_series = pd.to_numeric(df[c], errors="coerce").fillna(0)
                if (col_series != 0).any():
                    chosen_vol_col = c
                    break
            except Exception:
                pass
    if not chosen_vol_col:
        for c in vol_candidates:
            if c in df.columns:
                chosen_vol_col = c
                break

    close_candidates = ["ADJ CLOSE", "CLOSE"]
    chosen_close_col = None
    for c in close_candidates:
        if c in df.columns:
            chosen_close_col = c
            break

    rename_map = {
        "OPEN": "Open",
        "HIGH": "High",
        "LOW":  "Low",
    }
    if chosen_close_col:
        rename_map[chosen_close_col] = "Close"
    if chosen_vol_col:
        rename_map[chosen_vol_col] = "Volume"

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

    # ── Veri kalite kontrolleri ───────────────────────────────────────────
    # Duplikate timestamp tespiti
    dup_count = df.index.duplicated().sum()
    if dup_count > 0:
        print(f"[DataFeed] UYARI: {dup_count} duplikate timestamp tespit edildi → kaldırılıyor.")
        df = df[~df.index.duplicated(keep="first")]

    # OHLC mantık kontrolü: High < Low veya Close dışarıda
    bad_hl = (df["High"] < df["Low"]).sum()
    if bad_hl > 0:
        print(f"[DataFeed] UYARI: {bad_hl} barda High < Low — bu barlar çıkarılıyor.")
        df = df[df["High"] >= df["Low"]]

    # Sıfır veya negatif fiyat
    bad_price = (df[["Open", "High", "Low", "Close"]] <= 0).any(axis=1).sum()
    if bad_price > 0:
        print(f"[DataFeed] UYARI: {bad_price} barda sıfır/negatif fiyat — çıkarılıyor.")
        df = df[(df[["Open", "High", "Low", "Close"]] > 0).all(axis=1)]

    return df


def load_all_symbols() -> dict[str, pd.DataFrame]:
    """
    Multi-symbol veri yükleyici — öncelik sırası:

    1. csv/symbols/   — MT5'ten export edilmiş GERÇEK bağımsız semboller
    2. cache/symbols/ — sentetik semboller (sadece REAL_SYMBOLS_ONLY=0 ise)

    STRESS_SINGLE_SYMBOL=1 → multi-symbol tamamen devre dışı
    REAL_SYMBOLS_ONLY=1    → sadece gerçek MT5 verisi (sentetik yok, varsayılan)
    REAL_SYMBOLS_ONLY=0    → sentetik veriye de izin ver
    """
    # STRESS_SINGLE_SYMBOL=1 → tek sembol modunu zorla
    if os.getenv("STRESS_SINGLE_SYMBOL", "0") == "1":
        return {}

    # 1) MT5 gerçek veri klasörü
    mt5_real_dir = _MT5_MULTI_DIR
    has_real_data = (
        os.path.isdir(mt5_real_dir)
        and any(f.endswith("_15m.csv") for f in os.listdir(mt5_real_dir))
    )

    if has_real_data:
        result = _load_symbols_from_dir(mt5_real_dir, loader="mt5")
        if result:
            print(f"[DataFeed] MT5 gerçek veri: {len(result)} sembol ({mt5_real_dir})")
            return result

    # 2) Sentetik cache — varsayılan olarak devre dışı (sadece açıkça izin verilirse)
    real_only = os.getenv("REAL_SYMBOLS_ONLY", "1")
    if real_only != "0":
        print("[DataFeed] Sentetik semboller devre dışı (REAL_SYMBOLS_ONLY=1). "
              "Gerçek veri için: csv/symbols/ klasörüne MT5 CSV'lerini kopyalayın.")
        return {}

    if not os.path.isdir(_SYMBOLS_DIR):
        return {}

    return _load_symbols_from_dir(_SYMBOLS_DIR, loader="csv")


def _load_symbols_from_dir(directory: str, loader: str = "csv") -> dict[str, pd.DataFrame]:
    """Belirtilen klasördeki tüm *_15m.csv dosyalarını yükler."""
    result: dict[str, pd.DataFrame] = {}
    for fname in sorted(os.listdir(directory)):
        if not fname.endswith("_15m.csv"):
            continue
        symbol = fname.replace("_15m.csv", "")
        path   = os.path.join(directory, fname)
        try:
            if loader == "mt5":
                df = _load_mt5_csv(path)
                df = _ensure_spy_vix(df)
            else:
                df = pd.read_csv(path, index_col=0, parse_dates=True)
            df = df.dropna(subset=["Open", "High", "Low", "Close"])
            df = df.sort_index()
            result[symbol] = df
            print(f"[DataFeed] {symbol}: {len(df)} bar ({df.index[0].date()} → {df.index[-1].date()})")
        except Exception as e:
            print(f"[DataFeed] UYARI: {fname} yüklenemedi: {e}")

    print(f"[DataFeed] Toplam {len(result)} sembol yüklendi.")
    return result


def download_yf_symbols(
    symbols: list[str] | None = None,
    period: str = "2y",
    interval: str = "15m",
    output_dir: str | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Download 15m OHLCV data for given symbols via yfinance and save as
    MT5-compatible tab-separated CSV files in V1/csv/symbols/.

    Format: DATE\tTIME\tOPEN\tHIGH\tLOW\tCLOSE\tTICKVOL\tVOL\tSPREAD

    Parameters
    ----------
    symbols   : List of yfinance tickers (default: ["GLD", "TLT", "GC=F"])
    period    : yfinance period string (default: "2y")
    interval  : yfinance interval string (default: "15m")
    output_dir: Directory to save CSVs (default: V1/csv/symbols/)

    Returns
    -------
    dict mapping symbol → DataFrame (OHLCV with DatetimeIndex)
    """
    try:
        import yfinance as yf
    except ImportError:
        raise ImportError("yfinance is required: pip install yfinance")

    if symbols is None:
        symbols = ["GLD", "TLT", "GC=F"]

    if output_dir is None:
        output_dir = _MT5_MULTI_DIR
    os.makedirs(output_dir, exist_ok=True)

    result: dict[str, pd.DataFrame] = {}

    for sym in symbols:
        print(f"[DataFeed] Downloading {sym} ({interval}, {period}) via yfinance...")
        try:
            ticker = yf.Ticker(sym)
            df_raw = ticker.history(period=period, interval=interval, auto_adjust=True)

            if df_raw is None or df_raw.empty:
                print(f"[DataFeed] WARN: No data for {sym}, skipping.")
                continue

            # Standardise columns
            df_raw = df_raw.rename(columns={
                "Open":   "Open",
                "High":   "High",
                "Low":    "Low",
                "Close":  "Close",
                "Volume": "Volume",
            })
            df_raw = df_raw[["Open", "High", "Low", "Close", "Volume"]].copy()
            df_raw = df_raw.dropna(subset=["Open", "High", "Low", "Close"])
            df_raw.index = pd.DatetimeIndex(df_raw.index)
            df_raw = df_raw.sort_index()

            # Build safe filename: replace "=" with "_" for GC=F → GC_F
            safe_sym = sym.replace("=", "_")
            out_path = os.path.join(output_dir, f"{safe_sym}_15m.csv")

            # Write MT5-compatible tab-separated CSV
            with open(out_path, "w", encoding="utf-8") as fh:
                fh.write("DATE\tTIME\tOPEN\tHIGH\tLOW\tCLOSE\tTICKVOL\tVOL\tSPREAD\n")
                for ts, row in df_raw.iterrows():
                    date_str = ts.strftime("%Y.%m.%d")
                    time_str = ts.strftime("%H:%M")
                    vol      = int(row["Volume"]) if not pd.isna(row["Volume"]) else 0
                    fh.write(
                        f"{date_str}\t{time_str}\t"
                        f"{row['Open']:.6f}\t{row['High']:.6f}\t"
                        f"{row['Low']:.6f}\t{row['Close']:.6f}\t"
                        f"{vol}\t{vol}\t0\n"
                    )

            result[safe_sym] = df_raw
            print(
                f"[DataFeed] {sym} → {out_path} | "
                f"{len(df_raw)} bars ({df_raw.index[0].date()} → {df_raw.index[-1].date()})"
            )

        except Exception as exc:
            print(f"[DataFeed] ERROR downloading {sym}: {exc}")

    print(f"[DataFeed] download_yf_symbols complete: {len(result)} symbols saved to {output_dir}")
    return result


class DataFeed:
    def __init__(self, cache_file: str | None = None, symbol: str | None = None):
        os.makedirs(CACHE_DIR, exist_ok=True)
        os.makedirs(CSV_DIR,   exist_ok=True)
        self._cache_file = cache_file
        self._symbol     = symbol  # Multi-symbol modda hangi sembol

    def download_historical_data(self, force_refresh: bool = False) -> pd.DataFrame:
        # 1) Belirli sembol isteniyorsa direkt cache'ten al
        if self._symbol:
            return self._load_symbol_cache(self._symbol)

        # 2) MT5 CSV varsa her zaman önce onu kullan (csv/nasdaq_15m.csv)
        #    Bu, cache/symbols/ içindeki sentetik verinin önüne geçer.
        if USE_CSV_DATA and self._cache_file is None:
            csv_path = os.path.join(CSV_DIR, CSV_FILE_NAME)
            if os.path.exists(csv_path):
                return self._load_from_csv()

        # 3) cache/symbols/ — yalnızca tek-sembol modu kapalıysa kullan
        single_sym_mode = os.getenv("STRESS_SINGLE_SYMBOL", "0") == "1"
        if os.path.isdir(_SYMBOLS_DIR) and not self._cache_file and not single_sym_mode:
            csvs = [f for f in os.listdir(_SYMBOLS_DIR) if f.endswith("_15m.csv")]
            if csvs:
                sym = sorted(csvs)[0].replace("_15m.csv", "")
                return self._load_symbol_cache(sym)

        # 4) Eski cache modu
        print(f"[DataFeed] {CSV_FILE_NAME} bulunamadı, cache moduna geçiliyor.")
        return self._load_from_cache()

    def _load_symbol_cache(self, symbol: str) -> pd.DataFrame:
        path = os.path.join(_SYMBOLS_DIR, f"{symbol}_15m.csv")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"{symbol} için cache bulunamadı: {path}\n"
                "Önce 'python data/data_pipeline.py' çalıştırın."
            )
        df = pd.read_csv(path, index_col=0, parse_dates=True)
        df = _ensure_spy_vix(df)
        df = _resample(df, TIMEFRAME)
        df = df.dropna(subset=["Open", "High", "Low", "Close"]).sort_index()
        print(f"[DataFeed] {symbol}: {len(df)} bar")
        return df

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
