"""
V2/ml/features_v2.py
--------------------
Enhanced Feature Engine for V2 pipeline.

Inherits V1 FeatureEngine and adds regime-specific features:

New features added
------------------
  BB_Width         : (BB_Upper - BB_Lower) / BB_Middle          — absolute squeeze metric
  BB_Width_MA      : BB_Width.rolling(50).mean()                — baseline squeeze level
  BB_Width_Ratio   : BB_Width / BB_Width_MA                     — relative squeeze ratio
  ATR_MA_20        : ATR.rolling(20).mean()                     — smoothed volatility baseline
  ATR_Ratio        : ATR / ATR_MA_20                            — volatility expansion/contraction
  MACD_Hist_Slope  : MACD_Hist.diff(3)                          — 3-bar momentum slope
  RSI_Slope        : RSI.diff(5)                                — 5-bar RSI direction
  EMA9_Slope_Pct   : (EMA9 - EMA9.shift(10)) / EMA9.shift(10)  — 10-bar EMA speed

Note: ATR_Ratio already exists in V1 (ATR / 20-bar rolling mean).
      V2 adds BB_Width_Ratio and MACD_Hist_Slope as new regime columns.
"""

from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

# ── Path bootstrap ────────────────────────────────────────────────────────────
_V1_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'V1'))
# Insert V1 root BEFORE V2 root so V1 ml.features is found first
if _V1_ROOT in sys.path:
    sys.path.remove(_V1_ROOT)
sys.path.insert(0, _V1_ROOT)

# Temporarily hide V2/ml so the import resolves to V1/ml/features.py
import importlib
_v1_ml_path = os.path.join(_V1_ROOT, 'ml', 'features.py')
import importlib.util as _iutil
_spec = _iutil.spec_from_file_location("_v1_ml_features", _v1_ml_path)
_mod  = _iutil.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
_FeatureEngineV1 = _mod.FeatureEngine
WARMUP_BARS      = _mod.WARMUP_BARS
EMA_FAST         = _mod.EMA_FAST

try:
    from config.settings import EMA_FAST as _EFA
    EMA_FAST = _EFA
except ImportError:
    pass


class FeatureEngineV2(_FeatureEngineV1):
    """
    Drop-in replacement for V1 FeatureEngine that adds regime-specific columns.

    Usage
    -----
    Same as V1:
        fe = FeatureEngineV2()
        df = fe.calculate_indicators(raw_df)

    Additional columns in output
    ----------------------------
    BB_Width, BB_Width_MA, BB_Width_Ratio,
    ATR_MA_20 (already present as base for ATR_Ratio in V1),
    MACD_Hist_Slope, RSI_Slope, EMA9_Slope_Pct
    """

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        # ── V1 indicators first ───────────────────────────────────────────
        df = super().calculate_indicators(df)

        # ── V2 additions ──────────────────────────────────────────────────

        # BB_Width = (Upper - Lower) / Mid  (already computed in live_features; add as column)
        if {"BB_Upper", "BB_Lower", "BB_Mid"}.issubset(df.columns):
            mid = df["BB_Mid"].replace(0, np.nan)
            df["BB_Width"] = (df["BB_Upper"] - df["BB_Lower"]) / mid
        else:
            df["BB_Width"] = np.nan

        # BB_Width_MA (50-bar rolling mean of BB_Width)
        df["BB_Width_MA"] = df["BB_Width"].rolling(50, min_periods=10).mean()

        # BB_Width_Ratio: squeeze intensity relative to baseline
        df["BB_Width_Ratio"] = df["BB_Width"] / df["BB_Width_MA"].replace(0, np.nan)

        # ATR_MA_20 (named explicitly; ATR_Ratio from V1 uses same denominator)
        df["ATR_MA_20"] = df["ATR"].rolling(20, min_periods=5).mean()
        # ATR_Ratio is already present from V1; re-compute to ensure consistency
        df["ATR_Ratio"] = df["ATR"] / df["ATR_MA_20"].replace(0, np.nan)

        # MACD_Hist_Slope: 3-bar diff of MACD histogram
        if "MACD_Hist" in df.columns:
            df["MACD_Hist_Slope"] = df["MACD_Hist"].diff(3)
        else:
            df["MACD_Hist_Slope"] = np.nan

        # RSI_Slope: 5-bar diff of RSI
        if "RSI" in df.columns:
            df["RSI_Slope"] = df["RSI"].diff(5)
        else:
            df["RSI_Slope"] = np.nan

        # EMA9_Slope_Pct: 10-bar percentage change of EMA9
        ema9_col = f"EMA_{EMA_FAST}"
        if ema9_col in df.columns:
            prev10 = df[ema9_col].shift(10).replace(0, np.nan)
            df["EMA9_Slope_Pct"] = (df[ema9_col] - prev10) / prev10
        else:
            df["EMA9_Slope_Pct"] = np.nan

        return df

    def generate_live_features(self, df: pd.DataFrame, current_index: int) -> pd.DataFrame:
        """
        Extends V1 live features with V2 regime-specific columns.
        Returns a single-row DataFrame of all features for current bar.
        """
        # V1 features
        feat_df = super().generate_live_features(df, current_index)
        if feat_df is None or feat_df.empty:
            return feat_df

        bar = df.iloc[current_index]

        # ── Add V2 columns if available in df ────────────────────────────
        v2_cols = [
            "BB_Width", "BB_Width_MA", "BB_Width_Ratio",
            "ATR_MA_20", "ATR_Ratio",
            "MACD_Hist_Slope", "RSI_Slope", "EMA9_Slope_Pct",
        ]
        for col in v2_cols:
            if col in df.columns:
                val = bar.get(col, np.nan)
                feat_df[col] = float(val) if val is not None and not (isinstance(val, float) and np.isnan(val)) else np.nan
            else:
                feat_df[col] = np.nan

        return feat_df
