"""
strategy/engine.py — İki katmanlı sinyal motoru.

Katman 1 — Erken Trend (EARLY):   EMA kesişimi sonrası ilk 1-25 bar
Katman 2 — Trend İçi Pullback:    Yerleşik trend (25-120 bar) + RSI geri çekilmesi

Her iki katman da aynı STRONG_LONG / STRONG_SHORT sinyalini döndürür.
Daha fazla işlem fırsatı + yüksek win rate hedefi.
"""

from __future__ import annotations

import logging
import math
from typing import Literal

import pandas as pd

try:
    from config.settings import (
        EMA_FAST, EMA_SLOW, RSI_OVERBOUGHT, RSI_OVERSOLD, ATR_MIN_ENTRY, ATR_MIN_PCT,
    )
except ImportError:
    EMA_FAST       = 9
    EMA_SLOW       = 21
    RSI_OVERBOUGHT = 80
    RSI_OVERSOLD   = 20
    ATR_MIN_ENTRY  = 20.0
    ATR_MIN_PCT    = 0.0005

# ── Katman 1: Erken Trend Parametreleri ──────────────────────────────────────
ADX_STRONG          = 22.0   # raised from 18 → filter out choppy/low-momentum markets
EARLY_MIN_BARS      = 1
EARLY_MAX_BARS      = 30
RSI_EARLY_LONG_MIN  = 38
RSI_EARLY_LONG_MAX  = 70
RSI_EARLY_SHORT_MIN = 30
RSI_EARLY_SHORT_MAX = 62

# ── Katman 2: Pullback Parametreleri ─────────────────────────────────────────
PULLBACK_MIN_BARS      = 25
PULLBACK_MAX_BARS      = 120
ADX_PULLBACK           = 25.0   # raised from 22 → kurulu trendde daha güçlü ADX = daha yüksek WR
RSI_PULLBACK_LONG_MIN  = 35
RSI_PULLBACK_LONG_MAX  = 55
RSI_PULLBACK_SHORT_MIN = 45
RSI_PULLBACK_SHORT_MAX = 65

# ── Üst Periyot / Rejim Filtresi ──────────────────────────────────────────────
# H4 trend filtresi: H4 yönüne karşı gelen işlemleri engeller
# Close_vs_EMA50 bug fix sonrası bu filtre de aktif ediliyor
H4_TREND_REQUIRED = False  # 15m verisinden hesaplanan H4 proxy güvenilir değil; Close_vs_EMA50 ile filtreleniyor
ADX_PERSIST_MIN   = 0  # Devre dışı — ADX eşiği (ADX_STRONG=22) zaten filtre görevi görür

SignalType = Literal["STRONG_LONG", "STRONG_SHORT", "HOLD"]
logger = logging.getLogger(__name__)


class StrategyEngine:

    def __init__(self) -> None:
        logger.info(
            "StrategyEngine v2 | EMA %d/%d | ADX>%.0f | "
            "Early[%d-%d bars] RSI L[%d-%d] S[%d-%d] | "
            "Pullback[%d-%d bars] RSI L[%d-%d] S[%d-%d]",
            EMA_FAST, EMA_SLOW, ADX_STRONG,
            EARLY_MIN_BARS, EARLY_MAX_BARS,
            RSI_EARLY_LONG_MIN, RSI_EARLY_LONG_MAX,
            RSI_EARLY_SHORT_MIN, RSI_EARLY_SHORT_MAX,
            PULLBACK_MIN_BARS, PULLBACK_MAX_BARS,
            RSI_PULLBACK_LONG_MIN, RSI_PULLBACK_LONG_MAX,
            RSI_PULLBACK_SHORT_MIN, RSI_PULLBACK_SHORT_MAX,
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
            ema_fast       = float(bar[f"EMA_{EMA_FAST}"])
            ema_slow       = float(bar[f"EMA_{EMA_SLOW}"])
            rsi            = float(bar["RSI"])
            adx            = float(bar.get("ADX",            0.0) or 0.0)
            macd_hist      = float(bar.get("MACD_Hist",      0.0) or 0.0)
            align_bars     = float(bar.get("EMA_Align_Bars", 0.0) or 0.0)
            close_vs_ema50 = float(bar.get("Close_vs_EMA50", 0.0) or 0.0)
            h4_trend       = float(bar.get("H4_EMA_Trend",   0.0) or 0.0)
            adx_persist    = float(bar.get("ADX_Persistence", 5.0) or 5.0)
            bb_pct         = float(bar.get("BB_Pct",          0.5) or 0.5)
            rsi_change3    = float(bar.get("RSI_Change3",     0.0) or 0.0)
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning("Bar %d okuma hatası: %s", current_index, exc)
            return "HOLD"

        if any(math.isnan(v) for v in (ema_fast, ema_slow, rsi)):
            return "HOLD"

        # ATR çok düşükse piyasa sıkışık — giriş kalitesi düşük
        atr   = float(bar.get("ATR",   0.0) or 0.0)
        close = float(bar.get("Close", 0.0) or 0.0)
        open_ = float(bar.get("Open",  close) or close)
        if atr > 0 and close > 0 and (atr / close) < ATR_MIN_PCT:
            return "HOLD"


        # MACD histogram ivmesi
        if prev is not None:
            prev_hist        = float(prev.get("MACD_Hist", macd_hist) or macd_hist)
            macd_accel_long  = (macd_hist - prev_hist) > 0
            macd_accel_short = (macd_hist - prev_hist) < 0
        else:
            macd_accel_long = macd_accel_short = True

        adx_ok       = not math.isnan(adx) and adx >= ADX_STRONG
        adx_pullback = not math.isnan(adx) and adx >= ADX_PULLBACK

        uptrend   = ema_fast > ema_slow
        downtrend = ema_fast < ema_slow

        h4_long_ok  = (not H4_TREND_REQUIRED) or h4_trend >= 0
        h4_short_ok = (not H4_TREND_REQUIRED) or h4_trend <= 0

        # ══ KATMAN 1: ERKEN TREND (bar 1-25) ══════════════════════════════
        if (
            uptrend
            and EARLY_MIN_BARS <= align_bars <= EARLY_MAX_BARS
            and RSI_EARLY_LONG_MIN <= rsi <= RSI_EARLY_LONG_MAX
            and adx_ok
            and macd_hist > 0
            and close_vs_ema50 > -0.015
            and h4_long_ok
        ):
            logger.debug(
                "Bar %d EARLY_LONG [ADX=%.1f Bars=%.0f RSI=%.1f MACDh=%.4f]",
                current_index, adx, align_bars, rsi, macd_hist,
            )
            return "STRONG_LONG"

        if (
            downtrend
            and -EARLY_MAX_BARS <= align_bars <= -EARLY_MIN_BARS
            and RSI_EARLY_SHORT_MIN <= rsi <= RSI_EARLY_SHORT_MAX
            and adx_ok
            and macd_hist < 0
            and close_vs_ema50 < 0.015
            and h4_short_ok
        ):
            logger.debug(
                "Bar %d EARLY_SHORT [ADX=%.1f Bars=%.0f RSI=%.1f MACDh=%.4f]",
                current_index, adx, align_bars, rsi, macd_hist,
            )
            return "STRONG_SHORT"

        # ══ KATMAN 2: TREND İÇİ PULLBACK (bar 25-120) ═════════════════════
        # Yerleşik trend içinde RSI geri çekildikten sonra giriş
        if (
            uptrend
            and PULLBACK_MIN_BARS <= align_bars <= PULLBACK_MAX_BARS
            and RSI_PULLBACK_LONG_MIN <= rsi <= RSI_PULLBACK_LONG_MAX
            and adx_pullback
            and macd_hist > 0
            and close_vs_ema50 > -0.015
            and h4_long_ok
        ):
            logger.debug(
                "Bar %d PULLBACK_LONG [ADX=%.1f Bars=%.0f RSI=%.1f]",
                current_index, adx, align_bars, rsi,
            )
            return "STRONG_LONG"

        if (
            downtrend
            and -PULLBACK_MAX_BARS <= align_bars <= -PULLBACK_MIN_BARS
            and RSI_PULLBACK_SHORT_MIN <= rsi <= RSI_PULLBACK_SHORT_MAX
            and adx_pullback
            and macd_hist < 0
            and close_vs_ema50 < 0.015
            and h4_short_ok
        ):
            logger.debug(
                "Bar %d PULLBACK_SHORT [ADX=%.1f Bars=%.0f RSI=%.1f]",
                current_index, adx, align_bars, rsi,
            )
            return "STRONG_SHORT"

        return "HOLD"
