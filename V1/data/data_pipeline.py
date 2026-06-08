"""
data/data_pipeline.py
---------------------
yfinance tabanlı otomatik veri indirme modülü.

UNIVERSE listesindeki 15 NASDAQ hissesini + SPY + VIX'i indirir,
cache/symbols/ klasörüne {symbol}_15m.csv olarak kaydeder.

Kullanım:
    python data/data_pipeline.py          # tüm sembolleri indir
    python data/data_pipeline.py AAPL     # sadece tek sembol
"""

from __future__ import annotations

import os
import sys
import time
import logging

import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

try:
    import yfinance as yf
except ImportError:
    raise ImportError("yfinance kurulu değil. Çalıştır: pip install yfinance")

try:
    from config.settings import UNIVERSE, SPY_SYMBOL, VIX_SYMBOL, CACHE_DIR
except ImportError:
    UNIVERSE   = ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA",
                  "AVGO", "COST", "NFLX", "AMD",  "ADBE", "QCOM", "TXN", "AMAT"]
    SPY_SYMBOL = "SPY"
    VIX_SYMBOL = "^VIX"
    CACHE_DIR  = os.path.join(ROOT_DIR, "cache")

SYMBOLS_DIR  = os.path.join(CACHE_DIR, "symbols")
TIMEFRAME    = "15m"
# yfinance 15m → max 60 gün
PERIOD       = "60d"
RETRY_DELAY  = 5   # saniye

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("data_pipeline")


def download_symbol(symbol: str, retries: int = 3) -> pd.DataFrame | None:
    """
    Tek bir sembolün 15m verisini yfinance'tan indirir.
    Başarısız olursa retries kadar tekrar dener.
    """
    for attempt in range(1, retries + 1):
        try:
            ticker = yf.Ticker(symbol)
            df = ticker.history(period=PERIOD, interval=TIMEFRAME, auto_adjust=True)

            if df is None or df.empty:
                logger.warning("[%s] Boş veri döndü (deneme %d/%d)", symbol, attempt, retries)
                if attempt < retries:
                    time.sleep(RETRY_DELAY)
                continue

            # yfinance sütun isimlerini standartlaştır
            df.index.name = "Datetime"
            df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
            df = df.dropna(subset=["Open", "High", "Low", "Close"])
            df = df.sort_index()

            logger.info(
                "[%s] ✅ %d bar indirildi | %s → %s",
                symbol, len(df), df.index[0].date(), df.index[-1].date()
            )
            return df

        except Exception as e:
            logger.error("[%s] Hata (deneme %d/%d): %s", symbol, attempt, retries, e)
            if attempt < retries:
                time.sleep(RETRY_DELAY * attempt)

    logger.error("[%s] ❌ İndirme başarısız, atlanıyor.", symbol)
    return None


def download_spy_vix() -> tuple[pd.Series | None, pd.Series | None]:
    """SPY Close ve VIX Close serilerini indirir (feature olarak kullanılır)."""
    spy_close = vix_close = None

    try:
        spy_df = yf.Ticker(SPY_SYMBOL).history(period=PERIOD, interval=TIMEFRAME, auto_adjust=True)
        if not spy_df.empty:
            spy_close = spy_df["Close"].rename("SPY_Close")
            logger.info("[SPY] ✅ %d bar", len(spy_close))
    except Exception as e:
        logger.warning("[SPY] İndirilemedi: %s", e)

    try:
        vix_df = yf.Ticker(VIX_SYMBOL).history(period=PERIOD, interval=TIMEFRAME, auto_adjust=True)
        if not vix_df.empty:
            vix_close = vix_df["Close"].rename("VIX_Close")
            logger.info("[VIX] ✅ %d bar", len(vix_close))
    except Exception as e:
        logger.warning("[VIX] İndirilemedi: %s", e)

    return spy_close, vix_close


def save_symbol(symbol: str, df: pd.DataFrame, spy: pd.Series | None, vix: pd.Series | None) -> str:
    """DataFrame'i cache/symbols/{symbol}_15m.csv olarak kaydeder."""
    os.makedirs(SYMBOLS_DIR, exist_ok=True)

    # SPY ve VIX'i join et (yoksa proxy değerler)
    if spy is not None:
        df = df.join(spy, how="left")
    if "SPY_Close" not in df.columns:
        df["SPY_Close"] = df["Close"]

    if vix is not None:
        df = df.join(vix, how="left")
    if "VIX_Close" not in df.columns:
        df["VIX_Close"] = 15.0

    # Eksik SPY/VIX değerlerini doldur
    df["SPY_Close"] = df["SPY_Close"].ffill().bfill().fillna(df["Close"])
    df["VIX_Close"] = df["VIX_Close"].ffill().bfill().fillna(15.0)

    path = os.path.join(SYMBOLS_DIR, f"{symbol}_15m.csv")
    df.to_csv(path, encoding="utf-8")
    size_kb = os.path.getsize(path) / 1024
    logger.info("[%s] 💾 Kaydedildi → %s (%.1f KB)", symbol, path, size_kb)
    return path


def run_pipeline(symbols: list[str] | None = None) -> dict[str, str]:
    """
    Tüm UNIVERSE (veya belirtilen liste) için veri indir ve kaydet.

    Returns
    -------
    dict[symbol → csv_path]  — başarıyla indirilen semboller
    """
    targets = symbols or UNIVERSE
    logger.info("=" * 60)
    logger.info("Veri pipeline başlatıldı | %d sembol | %s / %s", len(targets), TIMEFRAME, PERIOD)
    logger.info("=" * 60)

    spy_close, vix_close = download_spy_vix()

    saved: dict[str, str] = {}
    failed: list[str] = []

    for sym in targets:
        df = download_symbol(sym)
        if df is not None and not df.empty:
            path = save_symbol(sym, df, spy_close, vix_close)
            saved[sym] = path
        else:
            failed.append(sym)
        time.sleep(0.5)  # rate-limit dostu

    logger.info("=" * 60)
    logger.info("Pipeline tamamlandı | ✅ %d başarılı | ❌ %d başarısız", len(saved), len(failed))
    if failed:
        logger.warning("Başarısız semboller: %s", failed)
    logger.info("Veriler: %s", SYMBOLS_DIR)
    logger.info("=" * 60)

    return saved


if __name__ == "__main__":
    # Komut satırından tek sembol: python data/data_pipeline.py AAPL
    target = sys.argv[1:] if len(sys.argv) > 1 else None
    run_pipeline(target)
