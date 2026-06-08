"""
data/synthetic_generator.py
----------------------------
Generates synthetic OHLCV instrument caches from the TECH master cache.

Instruments:
  - XAUUSD_proxy : Gold-like price range ~1800-2000, higher volatility, AR(1) noise
  - EURUSD_proxy : Forex-like price range ~1.05-1.15, lower volatility, mean-reverting

Both inherit SPY_Close and VIX_Close from the original TECH cache.
All OHLCV relationships are enforced: H >= max(O,C), L <= min(O,C).

Usage:
    python data/synthetic_generator.py
"""

from __future__ import annotations

import os
import sys
import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

MASTER_CACHE = os.path.join(ROOT_DIR, "cache", "cache_15m_360d.csv")
CACHE_DIR    = os.path.join(ROOT_DIR, "cache")


def _load_master() -> pd.DataFrame:
    df = pd.read_csv(MASTER_CACHE, index_col=0, parse_dates=True)
    print(f"[SynGen] Master loaded: {len(df)} bars  {df.index[0]} → {df.index[-1]}")
    return df


def _enforce_ohlcv(open_: np.ndarray,
                   high: np.ndarray,
                   low: np.ndarray,
                   close: np.ndarray,
                   volume: np.ndarray) -> tuple:
    """Ensure OHLCV constraints are satisfied bar-by-bar."""
    high   = np.maximum(high,  np.maximum(open_, close))
    low    = np.minimum(low,   np.minimum(open_, close))
    volume = np.maximum(volume, 1.0)
    return open_, high, low, close, volume


def generate_xauusd_proxy(df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    """
    Gold-like synthetic:
      - Close rescaled to 1800-2000 range
      - AR(1) noise  φ=0.85, σ=ATR*1.5
      - HL spread ~ 1.5x original ratio
    Target correlation with TECH: 0.5-0.7
    """
    rng = np.random.default_rng(seed)
    n   = len(df)

    # ── 1. Rescale Close to [1800, 2000] ──────────────────────────────────
    orig_close = df["Close"].values.copy().astype(float)
    c_min, c_max = orig_close.min(), orig_close.max()
    scaled_close = 1800.0 + (orig_close - c_min) / max(c_max - c_min, 1e-9) * 200.0

    # ── 2. Compute ATR-like range from original ────────────────────────────
    orig_atr = (df["High"].values - df["Low"].values).astype(float)
    # Scale ATR proportionally
    scale_factor  = 200.0 / max(c_max - c_min, 1e-9)
    scaled_atr    = orig_atr * scale_factor

    # ── 3. AR(1) noise process  φ=0.97, large σ for ~0.5-0.7 decorrelation ──
    phi   = 0.97
    sigma = scaled_atr * 5.5   # tuned to achieve ~0.5-0.7 target correlation
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = phi * noise[i-1] + rng.normal(0.0, sigma[i])

    close = scaled_close + noise

    # ── 4. Build Open from previous close + small gap ─────────────────────
    gap_noise = rng.normal(0.0, scaled_atr * 0.2)
    open_     = np.empty(n)
    open_[0]  = close[0]
    open_[1:] = close[:-1] + rng.normal(0.0, scaled_atr[1:] * 0.15)

    # ── 5. High / Low  with 1.5x HL spread of original ────────────────────
    hl_half = scaled_atr * 0.75  # half-range
    high    = np.maximum(open_, close) + np.abs(rng.normal(0.0, hl_half))
    low     = np.minimum(open_, close) - np.abs(rng.normal(0.0, hl_half))

    # ── 6. Volume: rescale original volume ────────────────────────────────
    orig_vol = df["Volume"].values.astype(float)
    vol_mean = orig_vol.mean()
    volume   = orig_vol * rng.uniform(0.8, 1.2, n)

    # ── 7. Enforce constraints ─────────────────────────────────────────────
    open_, high, low, close, volume = _enforce_ohlcv(open_, high, low, close, volume)

    out = pd.DataFrame({
        "Open":      open_,
        "High":      high,
        "Low":       low,
        "Close":     close,
        "Volume":    volume,
    }, index=df.index)

    # Carry over macro columns
    for col in ("SPY_Close", "VIX_Close"):
        if col in df.columns:
            out[col] = df[col].values

    out.index.name = "time"
    return out


def generate_eurusd_proxy(df: pd.DataFrame, seed: int = 137) -> pd.DataFrame:
    """
    Forex-like synthetic:
      - Close rescaled to 1.05-1.15 range
      - Mean-reverting (Ornstein-Uhlenbeck) noise  θ=0.05, σ=ATR*0.4
      - Tight HL spread (lower volatility)
    Target correlation with TECH: 0.5-0.7
    """
    rng = np.random.default_rng(seed)
    n   = len(df)

    # ── 1. Rescale Close to [1.05, 1.15] ──────────────────────────────────
    orig_close = df["Close"].values.copy().astype(float)
    c_min, c_max = orig_close.min(), orig_close.max()
    scaled_close = 1.05 + (orig_close - c_min) / max(c_max - c_min, 1e-9) * 0.10

    # ── 2. Scale ATR proportionally ────────────────────────────────────────
    orig_atr     = (df["High"].values - df["Low"].values).astype(float)
    scale_factor = 0.10 / max(c_max - c_min, 1e-9)
    scaled_atr   = np.maximum(orig_atr * scale_factor, 1e-6)

    # ── 3. Ornstein-Uhlenbeck mean-reverting noise ────────────────────────
    theta = 0.02    # slow mean-reversion — let noise accumulate for decorrelation
    sigma = scaled_atr * 3.0   # large σ to achieve ~0.5-0.7 target correlation
    noise = np.zeros(n)
    mu    = 0.0     # long-run mean of noise
    for i in range(1, n):
        noise[i] = noise[i-1] + theta * (mu - noise[i-1]) + rng.normal(0.0, sigma[i])

    close = scaled_close + noise

    # ── 4. Open from previous close + tiny gap ────────────────────────────
    open_    = np.empty(n)
    open_[0] = close[0]
    open_[1:] = close[:-1] + rng.normal(0.0, scaled_atr[1:] * 0.08)

    # ── 5. Tight High / Low ────────────────────────────────────────────────
    hl_half = scaled_atr * 0.4
    high    = np.maximum(open_, close) + np.abs(rng.normal(0.0, hl_half))
    low     = np.minimum(open_, close) - np.abs(rng.normal(0.0, hl_half))

    # ── 6. Volume ──────────────────────────────────────────────────────────
    orig_vol = df["Volume"].values.astype(float)
    volume   = orig_vol * rng.uniform(0.5, 1.5, n)

    # ── 7. Enforce constraints ─────────────────────────────────────────────
    open_, high, low, close, volume = _enforce_ohlcv(open_, high, low, close, volume)

    out = pd.DataFrame({
        "Open":      open_,
        "High":      high,
        "Low":       low,
        "Close":     close,
        "Volume":    volume,
    }, index=df.index)

    for col in ("SPY_Close", "VIX_Close"):
        if col in df.columns:
            out[col] = df[col].values

    out.index.name = "time"
    return out


def _check_correlation(df_base: pd.DataFrame, df_synth: pd.DataFrame, name: str) -> None:
    corr = df_base["Close"].corr(df_synth["Close"])
    print(f"[SynGen] {name} ↔ TECH Close correlation: {corr:.3f}")


def generate_all() -> None:
    os.makedirs(CACHE_DIR, exist_ok=True)
    df_master = _load_master()

    # XAUUSD proxy
    df_xau = generate_xauusd_proxy(df_master, seed=42)
    _check_correlation(df_master, df_xau, "XAUUSD_proxy")
    xau_path = os.path.join(CACHE_DIR, "cache_XAUUSD_15m.csv")
    df_xau.to_csv(xau_path)
    print(f"[SynGen] Saved XAUUSD proxy → {xau_path}  ({len(df_xau)} bars)")

    # EURUSD proxy
    df_eur = generate_eurusd_proxy(df_master, seed=137)
    _check_correlation(df_master, df_eur, "EURUSD_proxy")
    eur_path = os.path.join(CACHE_DIR, "cache_EURUSD_15m.csv")
    df_eur.to_csv(eur_path)
    print(f"[SynGen] Saved EURUSD proxy → {eur_path}  ({len(df_eur)} bars)")

    print("[SynGen] Done.")


if __name__ == "__main__":
    generate_all()
