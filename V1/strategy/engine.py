"""
strategy/engine.py — Yüksek kaliteli sinyal filtresi.

Sadece güçlü, erken-trend, momentum teyitli setuplarda işlem açar.
Daha az işlem ama daha yüksek win rate.
"""

from __future__ import annotations

import logging
import math
from typing import Literal

import pandas as pd

try:
    from config.settings import (
        EMA_FAST, EMA_SLOW, RSI_OVERBOUGHT, RSI_OVERSOLD,
    )
except ImportError:
    EMA_FAST       = 9
    EMA_SLOW       = 21
    RSI_OVERBOUGHT = 80
    RSI_OVERSOLD   = 20

# ── Strateji Parametreleri ────────────────────────────────────────────────────
ADX_STRONG      = 24.0   # Güçlü trend eşiği
MAX_ALIGN_BARS  = 20     # Trendin en geç kaçıncı barında girilebilir
MIN_ALIGN_BARS  = 2      # Çok erken (1.bar) girişi engelle
RSI_LONG_MIN    = 38     # Long için RSI alt sınır
RSI_LONG_MAX    = 70     # Long için RSI üst sınır
RSI_SHORT_MIN   = 30     # Short için RSI alt sınır
RSI_SHORT_MAX   = 62     # Short için RSI üst sınır
MACD_HIST_ACCEL = True   # MACD histogram ivmelenmelidir (değişim pozitif/negatif)

SignalType = Literal["STRONG_LONG", "STRONG_SHORT", "HOLD"]
logger = logging.getLogger(__name__)


class StrategyEngine:

    def __init__(self) -> None:
        logger.info(
            "StrategyEngine | EMA %d/%d | ADX>%.0f | AlignBars %d-%d | "
            "RSI L[%d-%d] S[%d-%d]",
            EMA_FAST, EMA_SLOW, ADX_STRONG, MIN_ALIGN_BARS, MAX_ALIGN_BARS,
            RSI_LONG_MIN, RSI_LONG_MAX, RSI_SHORT_MIN, RSI_SHORT_MAX,
        )

    def _get_bar(self, df: pd.DataFrame, idx: int) -> pd.Series | None:
        try:
            bar = df.iloc[idx]
        except IndexError:
            return None
        if not {f"EMA_{EMA_FAST}", f"EMA_{EMA_SLOW}", "RSI"}.issubset(df.columns):
            return None
        return bar

    def _prev_bar(self, df: pd.DataFrame, idx: int) -> pd.Series | None:
        if idx < 1:
            return None
        try:
            return df.iloc[idx - 1]
        except IndexError:
            return None

    def generate_base_signal(self, df: pd.DataFrame, current_index: int) -> SignalType:
        bar  = self._get_bar(df, current_index)
        prev = self._prev_bar(df, current_index)
        if bar is None:
            return "HOLD"

        try:
            ema_fast   = float(bar[f"EMA_{EMA_FAST}"])
            ema_slow   = float(bar[f"EMA_{EMA_SLOW}"])
            rsi        = float(bar["RSI"])
            adx        = float(bar.get("ADX",            0.0) or 0.0)
            macd_hist  = float(bar.get("MACD_Hist",      0.0) or 0.0)
            align_bars = float(bar.get("EMA_Align_Bars", 0.0) or 0.0)
            close_vs_ema50  = float(bar.get("Close_vs_EMA50", 0.0) or 0.0)
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("Bar %d okuma hatası: %s", current_index, exc)
            return "HOLD"

        if any(math.isnan(v) for v in (ema_fast, ema_slow, rsi)):
            return "HOLD"

        # MACD histogram ivmesi (bir önceki bardan değişim)
        if prev is not None:
            prev_hist = float(prev.get("MACD_Hist", macd_hist) or macd_hist)
            macd_accel_long  = (macd_hist - prev_hist) > 0   # histogram büyüyor
            macd_accel_short = (macd_hist - prev_hist) < 0   # histogram küçülüyor
        else:
            macd_accel_long = macd_accel_short = True  # bilinmiyorsa filtre atla

        adx_ok = not math.isnan(adx) and (adx >= ADX_STRONG)

        # ── STRONG_LONG ────────────────────────────────────────────────────
        if (
            ema_fast > ema_slow                          # yukarı trend
            and MIN_ALIGN_BARS <= align_bars <= MAX_ALIGN_BARS  # trend erken evre
            and RSI_LONG_MIN <= rsi <= RSI_LONG_MAX      # RSI momentum bölgesi
            and adx_ok                                   # güçlü trend
            and macd_hist > 0                            # momentum yukarı
            and macd_accel_long                          # momentum ivmeleniyor
            and close_vs_ema50 > -0.02                   # EMA50 desteği (ödünç değil)
        ):
            logger.debug(
                "Bar %d STRONG_LONG [ADX=%.1f AlignBars=%.0f RSI=%.1f MACDh=%.4f]",
                current_index, adx, align_bars, rsi, macd_hist,
            )
            return "STRONG_LONG"

        # ── STRONG_SHORT ───────────────────────────────────────────────────
        if (
            ema_fast < ema_slow
            and -MAX_ALIGN_BARS <= align_bars <= -MIN_ALIGN_BARS
            and RSI_SHORT_MIN <= rsi <= RSI_SHORT_MAX
            and adx_ok
            and macd_hist < 0
            and macd_accel_short
            and close_vs_ema50 < 0.02
        ):
            logger.debug(
                "Bar %d STRONG_SHORT [ADX=%.1f AlignBars=%.0f RSI=%.1f MACDh=%.4f]",
                current_index, adx, align_bars, rsi, macd_hist,
            )
            return "STRONG_SHORT"

        return "HOLD"
