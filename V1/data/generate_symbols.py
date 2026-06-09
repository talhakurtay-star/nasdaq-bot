"""
data/generate_symbols.py
------------------------
100 sembol için sektör-gerçekçi sentetik 15m veri üretir.

Her sembol:
  - QQQ baz hareketi × beta (sektöre özgü)
  - Sektöre özgü ek volatilite ve gürültü
  - Gerçek fiyat aralığına göre ölçeklendirilmiş başlangıç seviyesi

Çıktı: cache/symbols/{SYM}_15m.csv
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from config.settings import CSV_DIR, CSV_FILE_NAME, UNIVERSE, SECTOR_MAP

# ── Sektör varsayılan parametreleri ──────────────────────────────────────────
SECTOR_DEFAULTS: dict[str, dict] = {
    "TECH":       {"beta": 1.15, "vol": 0.009},
    "SEMI":       {"beta": 1.45, "vol": 0.018},
    "CONSUMER":   {"beta": 1.10, "vol": 0.012},
    "HEALTH":     {"beta": 0.80, "vol": 0.008},
    "FINTECH":    {"beta": 1.20, "vol": 0.013},
    "MEDIA":      {"beta": 1.10, "vol": 0.014},
    "INDUSTRIAL": {"beta": 0.85, "vol": 0.006},
    "ENERGY":     {"beta": 0.75, "vol": 0.010},  # tech ile zayıf korelasyon
    "MATERIALS":  {"beta": 0.90, "vol": 0.012},
    "ETF_TECH":   {"beta": 1.00, "vol": 0.004},
    "ETF_BROAD":  {"beta": 0.90, "vol": 0.003},
    "ETF_SMALL":  {"beta": 1.05, "vol": 0.007},
    "ETF_FIN":    {"beta": 0.80, "vol": 0.006},
    "ETF_ENERGY": {"beta": 0.70, "vol": 0.008},
    "ETF_HEALTH": {"beta": 0.75, "vol": 0.005},
    "ETF_GOLD":   {"beta": 0.25, "vol": 0.006},  # negatife yakın korelasyon
    "OTHER":      {"beta": 1.00, "vol": 0.010},
}

# ── Sembol başlangıç fiyatları (2023 ortası yaklaşık) ─────────────────────────
BASE_PRICES: dict[str, float] = {
    "AAPL": 185, "MSFT": 330, "GOOGL": 130, "META": 285, "ORCL": 115,
    "CRM":  215, "ADBE": 480, "NOW":   550, "INTU": 450, "PANW": 200,
    "ZS":   155, "CRWD": 150, "DDOG":  100, "NET":   72, "SNOW": 160,
    "PLTR":  17, "FTNT":  60, "OKTA":   90, "MDB":   360,"SPLK": 110,
    "NVDA": 420, "AMD":  105, "QCOM":  120, "AVGO":  820,"TXN":  170,
    "AMAT": 130, "LRCX": 700, "KLAC":  480, "MU":     65,"MRVL":  55,
    "ON":    85, "MPWR": 380, "SMCI":   90, "WOLF":   35,"SLAB":  75,
    "AMZN": 128, "TSLA": 260, "NFLX":  420, "COST":  520,"SBUX":  99,
    "BKNG":2700, "ABNB": 135, "UBER":   45, "LYFT":   12,"DASH":  65,
    "ISRG": 320, "VRTX": 320, "REGN":  780, "MRNA":  120,"ILMN": 200,
    "IDXX": 500, "DXCM":  85, "PODD":  180, "HALO":   50,"ALGN": 310,
    "PYPL":  65, "SQ":    65, "COIN":   55, "V":     235, "MA":  380,
    "AXP":  165, "SOFI":  10, "AFRM":   15, "HOOD":    5,"BILL":  95,
    "SPOT": 155, "PINS":  25, "SNAP":   10, "ROKU":   65,"TTD":   75,
    "MGNI":   8, "PARA":  15, "WBD":    12,
    "HON":  195, "GE":   110, "CAT":   250, "DE":    380,
    "RTX":   95, "LMT":  470, "NOC":   470, "BA":    215,
    "XOM":  105, "CVX":  155, "COP":   115, "SLB":    55,
    "HAL":   38, "BKR":   32, "OXY":    62,
    "FCX":   38, "AA":    28, "NEM":    40, "GOLD":   18,"CLF":   18,
    "QQQ":  365, "SPY":  440, "IWM":   185, "XLF":    35,
    "XLE":   88, "XLV":  132, "GLD":   185,
}


def load_base_data() -> pd.DataFrame:
    csv_path = os.path.join(CSV_DIR, CSV_FILE_NAME)
    df = pd.read_csv(csv_path, sep="\t")
    df.columns = [c.strip().strip("<>").upper() for c in df.columns]
    if "DATE" in df.columns and "TIME" in df.columns:
        df["Datetime"] = pd.to_datetime(
            df["DATE"].astype(str) + " " + df["TIME"].astype(str),
            format="%Y.%m.%d %H:%M:%S", errors="coerce",
        )
    df = df.dropna(subset=["Datetime"]).set_index("Datetime")
    df = df.rename(columns={"OPEN": "Open", "HIGH": "High", "LOW": "Low", "CLOSE": "Close"})
    vol_col = next((c for c in ["VOL", "TICKVOL", "VOLUME"] if c in df.columns and df[c].sum() > 0), None)
    df["Volume"] = df[vol_col].fillna(0) if vol_col else 0.0
    return df[["Open", "High", "Low", "Close", "Volume"]].astype(float)


def generate_symbol(base_df: pd.DataFrame, symbol: str, seed: int) -> pd.DataFrame:
    sector  = SECTOR_MAP.get(symbol, "OTHER")
    params  = SECTOR_DEFAULTS.get(sector, SECTOR_DEFAULTS["OTHER"])
    beta    = params["beta"]
    vol     = params["vol"]
    price0  = BASE_PRICES.get(symbol, 100.0)

    rng = np.random.default_rng(seed)

    # QQQ log return'leri
    closes = base_df["Close"].values.astype(float)
    log_ret = np.zeros(len(closes))
    log_ret[1:] = np.diff(np.log(closes))

    # Sektöre özgü getiri: beta * piyasa + özel gürültü
    sym_ret = log_ret * beta + rng.normal(0, vol, size=len(log_ret))

    # ETF/Defansif için hafif negatif trendi QQQ ile kısmen ters çevir
    if sector in ("ETF_GOLD", "ENERGY", "ETF_ENERGY"):
        # %30 oranında ters hareket ekle (daha gerçekçi hedge etkisi)
        market_component = log_ret * beta * 0.7
        counter_component = -log_ret * 0.15  # hafif ters korelasyon
        sym_ret = market_component + counter_component + rng.normal(0, vol, size=len(log_ret))

    close = np.exp(np.cumsum(sym_ret)) * price0

    hl_range = (base_df["High"].values - base_df["Low"].values) / base_df["Close"].values
    hl_adj   = hl_range * beta * (price0 / closes)
    hl_adj   = np.clip(hl_adj, 0.0001, 0.10)

    open_arr = np.empty_like(close)
    open_arr[0] = close[0]
    open_arr[1:] = close[:-1] * (1 + rng.normal(0, vol * 0.3, size=len(close) - 1))

    sym_df = pd.DataFrame({
        "Open":   open_arr,
        "High":   close * (1 + hl_adj / 2),
        "Low":    close * (1 - hl_adj / 2),
        "Close":  close,
        "Volume": base_df["Volume"].values * rng.uniform(0.5, 1.5, size=len(base_df)),
    }, index=base_df.index)

    # High >= max(Open, Close), Low <= min(Open, Close) garantisi
    sym_df["High"] = sym_df[["Open", "Close", "High"]].max(axis=1)
    sym_df["Low"]  = sym_df[["Open", "Close", "Low"]].min(axis=1)

    return sym_df


def run() -> None:
    print("Baz veri yükleniyor (QQQ/NASDAQ 15m)...")
    base_df = load_base_data()
    print(f"  {len(base_df):,} bar | {base_df.index[0]} → {base_df.index[-1]}")

    out_dir = os.path.join(ROOT, "cache", "symbols")
    os.makedirs(out_dir, exist_ok=True)

    ok = 0
    for i, sym in enumerate(UNIVERSE):
        sym_df   = generate_symbol(base_df, sym, seed=100 + i)
        out_path = os.path.join(out_dir, f"{sym}_15m.csv")
        sym_df.to_csv(out_path)
        sector = SECTOR_MAP.get(sym, "OTHER")
        print(f"  [{i+1:3d}/100] {sym:<6} [{sector:<12}] → {len(sym_df):,} bar")
        ok += 1

    print(f"\n✅ {ok}/100 sembol oluşturuldu → {out_dir}/")


if __name__ == "__main__":
    run()
