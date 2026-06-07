"""
strategy/engine.py
------------------
Teknik Strateji Motoru — ADX trend filtresi ve MACD momentum onayı eklenmiş versiyon.
"""

from __future__ import annotations

import logging
import math
from typing import Literal

import pandas as pd

try:
    from config.settings import (
        EMA_FAST,
        EMA_SLOW,
        RSI_OVERBOUGHT,
        RSI_OVERSOLD,
    )
except ImportError:
    EMA_FAST = 9
    EMA_SLOW = 21
    RSI_OVERBOUGHT = 80
    RSI_OVERSOLD = 20

# ADX minimum trend gücü eşiği — ranging piyasada sinyal üretilmez
ADX_MIN_TREND = 20.0

SignalType = Literal["STRONG_LONG", "STRONG_SHORT", "HOLD"]
logger = logging.getLogger(__name__)


class StrategyEngine:
    """
    Ham sinyal motoru.

    Kural Seti
    ----------
    STRONG_LONG:
        EMA_FAST > EMA_SLOW (yukarı trend)
        AND RSI < RSI_OVERBOUGHT
        AND ADX >= ADX_MIN_TREND (güçlü trend — ranging değil)
        AND MACD_Hist > 0 (momentum teyidi)

    STRONG_SHORT:
        EMA_FAST < EMA_SLOW (aşağı trend)
        AND RSI > RSI_OVERSOLD
        AND ADX >= ADX_MIN_TREND
        AND MACD_Hist < 0

    HOLD: Hiçbiri karşılanmıyorsa.
    """

    def __init__(self) -> None:
        logger.info(
            "StrategyEngine başlatıldı | "
            f"EMA_FAST={EMA_FAST}, EMA_SLOW={EMA_SLOW}, "
            f"RSI_OB={RSI_OVERBOUGHT}, RSI_OS={RSI_OVERSOLD}, "
            f"ADX_MIN={ADX_MIN_TREND}"
        )

    def _get_bar(self, df: pd.DataFrame, current_index: int) -> pd.Series | None:
        try:
            bar = df.iloc[current_index]
        except IndexError:
            logger.warning(f"current_index={current_index} sınır dışı (toplam: {len(df)}).")
            return None

        required_cols = {f"EMA_{EMA_FAST}", f"EMA_{EMA_SLOW}", "RSI"}
        missing = required_cols - set(df.columns)
        if missing:
            logger.error(f"Eksik sütunlar: {missing}. calculate_indicators() çalıştırıldı mı?")
            return None

        return bar

    def generate_base_signal(self, df: pd.DataFrame, current_index: int) -> SignalType:
        bar = self._get_bar(df, current_index)
        if bar is None:
            return "HOLD"

        try:
            ema_fast   = float(bar[f"EMA_{EMA_FAST}"])
            ema_slow   = float(bar[f"EMA_{EMA_SLOW}"])
            rsi        = float(bar["RSI"])
            adx        = float(bar.get("ADX", 0.0) or 0.0)
            macd_hist  = float(bar.get("MACD_Hist", 0.0) or 0.0)
        except (KeyError, TypeError, ValueError) as exc:
            logger.warning(f"Bar {current_index}: değer okuma hatası: {exc}")
            return "HOLD"

        if any(math.isnan(v) for v in (ema_fast, ema_slow, rsi)):
            return "HOLD"

        # ADX NaN ise filtre atla (ısınma periyodunda yumuşak geçiş)
        adx_ok = math.isnan(adx) or (adx >= ADX_MIN_TREND)
        macd_hist_ok_long  = math.isnan(macd_hist) or (macd_hist > 0)
        macd_hist_ok_short = math.isnan(macd_hist) or (macd_hist < 0)

        # ── STRONG_LONG ────────────────────────────────────────────────────
        if (
            ema_fast > ema_slow
            and rsi < RSI_OVERBOUGHT
            and adx_ok
            and macd_hist_ok_long
        ):
            logger.debug(
                f"Bar {current_index}: STRONG_LONG "
                f"[EMA={ema_fast:.2f}>{ema_slow:.2f}, RSI={rsi:.1f}, "
                f"ADX={adx:.1f}, MACDh={macd_hist:.4f}]"
            )
            return "STRONG_LONG"

        # ── STRONG_SHORT ───────────────────────────────────────────────────
        if (
            ema_fast < ema_slow
            and rsi > RSI_OVERSOLD
            and adx_ok
            and macd_hist_ok_short
        ):
            logger.debug(
                f"Bar {current_index}: STRONG_SHORT "
                f"[EMA={ema_fast:.2f}<{ema_slow:.2f}, RSI={rsi:.1f}, "
                f"ADX={adx:.1f}, MACDh={macd_hist:.4f}]"
            )
            return "STRONG_SHORT"

        logger.debug(f"Bar {current_index}: HOLD.")
        return "HOLD"
