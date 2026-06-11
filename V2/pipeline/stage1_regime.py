"""
V2/pipeline/stage1_regime.py
-----------------------------
Stage 1 — Market Regime Filter.

Classifies each bar into one of five regimes:
    TREND_BULL  — strong upward trend
    TREND_BEAR  — strong downward trend
    BREAKOUT    — volatility spike (ATR expansion)
    CHOPPY      — low-momentum / range-bound market
    TRANSITION  — everything else (undefined regime)

Downstream effects
------------------
CHOPPY   → Stage 2 skips ML and returns HOLD immediately
BREAKOUT → Stage 2 uses tighter SL (0.5× ATR) and smaller position size (0.7×)
"""

from __future__ import annotations

import logging
import math
import os
import sys
from typing import Literal

import numpy as np
import pandas as pd

# ── V1 path ──────────────────────────────────────────────────────────────────
_V1_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'V1'))
if _V1_ROOT not in sys.path:
    sys.path.insert(0, _V1_ROOT)

try:
    from config.settings import EMA_FAST, EMA_SLOW
except ImportError:
    EMA_FAST, EMA_SLOW = 9, 21

try:
    from V2.config.settings import (
        REGIME_ADX_TREND,
        REGIME_ADX_CHOPPY,
        REGIME_ATR_BREAKOUT,
        REGIME_BB_CHOPPY,
    )
except ImportError:
    try:
        # When running from V2 root
        _V2_CFG = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
        if _V2_CFG not in sys.path:
            sys.path.insert(0, _V2_CFG)
        from config.settings import (
            REGIME_ADX_TREND,
            REGIME_ADX_CHOPPY,
            REGIME_ATR_BREAKOUT,
            REGIME_BB_CHOPPY,
        )
    except ImportError:
        REGIME_ADX_TREND    = 22.0
        REGIME_ADX_CHOPPY   = 18.0
        REGIME_ATR_BREAKOUT = 1.5
        REGIME_BB_CHOPPY    = 0.7

RegimeType = Literal["TREND_BULL", "TREND_BEAR", "BREAKOUT", "CHOPPY", "TRANSITION"]
logger = logging.getLogger(__name__)


class RegimeDetector:
    """
    Bar-level market regime classifier.

    Usage
    -----
    detector = RegimeDetector()
    regime = detector.detect(df, current_index)
    """

    def detect(self, df: pd.DataFrame, current_index: int) -> RegimeType:
        """
        Classify the current bar's market regime.

        Parameters
        ----------
        df            : DataFrame with calculated indicators (output of FeatureEngine)
        current_index : integer position of the current bar in df

        Returns
        -------
        RegimeType string literal
        """
        if current_index < 1 or current_index >= len(df):
            return "TRANSITION"

        bar = df.iloc[current_index]

        # ── Read required indicator values ────────────────────────────────
        try:
            adx       = float(bar.get("ADX",       np.nan) or np.nan)
            ema_fast  = float(bar.get(f"EMA_{EMA_FAST}", np.nan) or np.nan)
            ema_slow  = float(bar.get(f"EMA_{EMA_SLOW}", np.nan) or np.nan)
            ema_50    = float(bar.get("EMA_50",     np.nan) or np.nan)
            close     = float(bar.get("Close",      np.nan) or np.nan)
            macd_hist = float(bar.get("MACD_Hist",  0.0)    or 0.0)
            atr       = float(bar.get("ATR",        np.nan) or np.nan)
            atr_ratio = float(bar.get("ATR_Ratio",  np.nan) or np.nan)  # ATR / ATR_MA_20
            bb_upper  = float(bar.get("BB_Upper",   np.nan) or np.nan)
            bb_lower  = float(bar.get("BB_Lower",   np.nan) or np.nan)
            bb_mid    = float(bar.get("BB_Mid",     np.nan) or np.nan)
        except (TypeError, ValueError):
            return "TRANSITION"

        # ── Derived values ────────────────────────────────────────────────
        # BB width (normalized)
        if not any(math.isnan(v) for v in (bb_upper, bb_lower, bb_mid)) and bb_mid != 0:
            bb_width = (bb_upper - bb_lower) / bb_mid
        else:
            bb_width = float("nan")

        # BB width MA over last 50 bars (rolling; computed on-the-fly)
        bb_width_ma = self._bb_width_ma(df, current_index, window=50)

        # ── Regime 1: BREAKOUT — volatility spike ─────────────────────────
        # ATR_Ratio is already ATR / ATR_MA_20 (computed in FeatureEngine V1)
        if not math.isnan(atr_ratio) and atr_ratio >= REGIME_ATR_BREAKOUT:
            logger.debug(
                "Regime=BREAKOUT | bar=%d | ATR_Ratio=%.3f", current_index, atr_ratio
            )
            return "BREAKOUT"

        # ── Regime 2 & 3: CHOPPY — low ADX or narrow BB ─────────────────
        choppy_by_adx = not math.isnan(adx) and adx < REGIME_ADX_CHOPPY
        choppy_by_bb  = (
            not math.isnan(bb_width)
            and not math.isnan(bb_width_ma)
            and bb_width_ma > 0
            and bb_width < bb_width_ma * REGIME_BB_CHOPPY
        )
        if choppy_by_adx or choppy_by_bb:
            logger.debug(
                "Regime=CHOPPY | bar=%d | ADX=%.1f choppy_adx=%s bb_ratio=%s",
                current_index, adx if not math.isnan(adx) else -1,
                choppy_by_adx, choppy_by_bb,
            )
            return "CHOPPY"

        # ── Regime 4: TREND_BULL ──────────────────────────────────────────
        #   ADX > 22 AND EMA9 > EMA21 AND Close > EMA50 AND MACD_Hist > 0
        if (
            not math.isnan(adx) and adx > REGIME_ADX_TREND
            and not math.isnan(ema_fast) and not math.isnan(ema_slow)
            and ema_fast > ema_slow
            and not math.isnan(close) and not math.isnan(ema_50)
            and close > ema_50
            and macd_hist > 0
        ):
            logger.debug(
                "Regime=TREND_BULL | bar=%d | ADX=%.1f EMA_gap=%.4f",
                current_index, adx, (ema_fast - ema_slow),
            )
            return "TREND_BULL"

        # ── Regime 5: TREND_BEAR ─────────────────────────────────────────
        #   ADX > 22 AND EMA9 < EMA21 AND Close < EMA50 AND MACD_Hist < 0
        if (
            not math.isnan(adx) and adx > REGIME_ADX_TREND
            and not math.isnan(ema_fast) and not math.isnan(ema_slow)
            and ema_fast < ema_slow
            and not math.isnan(close) and not math.isnan(ema_50)
            and close < ema_50
            and macd_hist < 0
        ):
            logger.debug(
                "Regime=TREND_BEAR | bar=%d | ADX=%.1f EMA_gap=%.4f",
                current_index, adx, (ema_fast - ema_slow),
            )
            return "TREND_BEAR"

        # ── Default: TRANSITION ───────────────────────────────────────────
        return "TRANSITION"

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _bb_width_ma(df: pd.DataFrame, current_index: int, window: int = 50) -> float:
        """
        Compute the rolling mean of BB_Width over the last `window` bars up to
        (and including) current_index.  Returns NaN if insufficient data or if
        the required columns are not present.
        """
        needed = {"BB_Upper", "BB_Lower", "BB_Mid"}
        if not needed.issubset(df.columns):
            return float("nan")

        start = max(0, current_index - window + 1)
        sub   = df.iloc[start : current_index + 1]

        mid = sub["BB_Mid"].replace(0, np.nan)
        bw  = (sub["BB_Upper"] - sub["BB_Lower"]) / mid
        bw  = bw.dropna()

        if bw.empty:
            return float("nan")
        return float(bw.mean())

    def regime_summary(self, df: pd.DataFrame, current_index: int) -> dict:
        """Return a dict of key indicator values used for regime detection (for debugging)."""
        if current_index >= len(df):
            return {}
        bar = df.iloc[current_index]
        return {
            "regime":    self.detect(df, current_index),
            "ADX":       float(bar.get("ADX",      float("nan")) or float("nan")),
            "ATR_Ratio": float(bar.get("ATR_Ratio", float("nan")) or float("nan")),
            "MACD_Hist": float(bar.get("MACD_Hist", 0.0) or 0.0),
            "EMA_fast":  float(bar.get(f"EMA_{EMA_FAST}", float("nan")) or float("nan")),
            "EMA_slow":  float(bar.get(f"EMA_{EMA_SLOW}", float("nan")) or float("nan")),
        }
