"""
V2/pipeline/stage3_exit.py
--------------------------
Stage 3: Adaptive Exit Controller

Called EVERY BAR while a position is open.
Evaluates momentum health and recommends an exit action.

Exit score (0.0 – 1.0)
-----------------------
  0.0 → momentum dead  → EXIT_NOW
  1.0 → momentum strong → HOLD_POSITION

Features used
-------------
  - MACD_Hist slope over last 3 bars   (fading momentum → negative)
  - RSI trajectory over last 5 bars    (falling RSI on longs → negative)
  - ATR contraction                    (volatility drying → neutral / mild negative)
  - Unrealised profit in R-multiples   (gate: only EXIT early if > EXIT_MIN_PROFIT_R)
  - Bars since entry                   (time-decay factor)

Output
------
  ExitAction:
    EXIT_NOW       — Close position immediately (if profit > threshold)
    TIGHTEN_TRAIL  — Reduce trailing ATR multiplier to 1.5× temporarily
    HOLD_POSITION  — Keep running, momentum still healthy

Integration with simulator.py
------------------------------
  The orchestrator calls stage3.evaluate() every bar and interprets:
    EXIT_NOW      → portfolio.close_trade(bar_close, timestamp, reason="EARLY_EXIT")
    TIGHTEN_TRAIL → temporarily override TRAILING_ATR_MULT to 1.5
    HOLD_POSITION → no action
"""

from __future__ import annotations

import logging
import math
import os
import sys
from typing import Literal, Optional

import numpy as np
import pandas as pd

# ── Path bootstrap ────────────────────────────────────────────────────────────
_V2_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
_V1_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'V1'))
for _p in (_V2_ROOT, _V1_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

try:
    from config.settings import (
        EXIT_MOMENTUM_THRESHOLD,
        EXIT_MIN_PROFIT_R,
        EXIT_TIGHTEN_THRESHOLD,
        ATR_MULTIPLIER,
    )
except ImportError:
    EXIT_MOMENTUM_THRESHOLD = 0.35
    EXIT_MIN_PROFIT_R       = 0.25
    EXIT_TIGHTEN_THRESHOLD  = 0.45
    ATR_MULTIPLIER          = 2.5

ExitAction = Literal["EXIT_NOW", "TIGHTEN_TRAIL", "HOLD_POSITION"]
logger = logging.getLogger(__name__)


class AdaptiveExit:
    """
    Adaptive Exit Controller — evaluates whether to exit early or tighten trail.

    This module is designed to be ML-upgradeable: replace _compute_score() with
    a trained regressor while keeping the same interface.

    Parameters
    ----------
    momentum_threshold : float
        Score below this → EXIT_NOW  (default 0.35)
    tighten_threshold  : float
        Score below this but above momentum_threshold → TIGHTEN_TRAIL (default 0.45)
    min_profit_r       : float
        Minimum profit in R-multiples before early exit is considered (default 0.25)
    """

    def __init__(
        self,
        momentum_threshold: float = EXIT_MOMENTUM_THRESHOLD,
        tighten_threshold:  float = EXIT_TIGHTEN_THRESHOLD,
        min_profit_r:       float = EXIT_MIN_PROFIT_R,
    ) -> None:
        self.momentum_threshold = momentum_threshold
        self.tighten_threshold  = tighten_threshold
        self.min_profit_r       = min_profit_r
        logger.info(
            "AdaptiveExit | exit_thr=%.2f  tighten_thr=%.2f  min_R=%.2f",
            momentum_threshold, tighten_threshold, min_profit_r,
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(
        self,
        df: pd.DataFrame,
        current_index: int,
        position,           # risk.portfolio.Position (duck-typed)
        regime: str = "TRANSITION",
    ) -> ExitAction:
        """
        Evaluate exit action for the current bar.

        Parameters
        ----------
        df            : indicator-enriched DataFrame
        current_index : current bar index
        position      : open Position object from PortfolioManager
                        Must expose: entry_price, stop_loss, is_long, atr_at_open,
                        open_time, sl_distance
        regime        : current RegimeType string from Stage 1

        Returns
        -------
        ExitAction
        """
        if position is None:
            return "HOLD_POSITION"

        try:
            bar       = df.iloc[current_index]
            bar_close = float(bar["Close"])
        except (IndexError, KeyError):
            return "HOLD_POSITION"

        # ── Gate: only consider early exit if we have enough profit ──────
        profit_r = self._unrealised_r(position, bar_close)
        if profit_r < self.min_profit_r:
            return "HOLD_POSITION"

        # ── Compute momentum score ────────────────────────────────────────
        score = self._compute_score(df, current_index, position, regime)

        if math.isnan(score):
            return "HOLD_POSITION"

        # ── Interpret score ───────────────────────────────────────────────
        if score <= self.momentum_threshold:
            logger.info(
                "Stage3 EXIT_NOW | bar=%d | score=%.3f profit_r=%.2fR regime=%s",
                current_index, score, profit_r, regime,
            )
            return "EXIT_NOW"

        if score <= self.tighten_threshold:
            logger.debug(
                "Stage3 TIGHTEN_TRAIL | bar=%d | score=%.3f profit_r=%.2fR",
                current_index, score, profit_r,
            )
            return "TIGHTEN_TRAIL"

        return "HOLD_POSITION"

    def score(
        self,
        df: pd.DataFrame,
        current_index: int,
        position,
        regime: str = "TRANSITION",
    ) -> float:
        """Return the raw momentum score (0.0 – 1.0) without taking action."""
        return self._compute_score(df, current_index, position, regime)

    # ------------------------------------------------------------------
    # Scoring logic (rule-based; ML-upgradeable)
    # ------------------------------------------------------------------

    def _compute_score(
        self,
        df: pd.DataFrame,
        current_index: int,
        position,
        regime: str,
    ) -> float:
        """
        Compute a momentum health score in [0.0, 1.0].

        Sub-scores (each 0-1, equally weighted unless noted):
          1. MACD_Hist slope over 3 bars      (weight 0.30)
          2. RSI trajectory over 5 bars        (weight 0.30)
          3. ATR expansion / contraction       (weight 0.15)
          4. Price vs partial TP level         (weight 0.15)
          5. Time decay (bars since entry)     (weight 0.10)
        """
        if current_index < 5:
            return 1.0  # Not enough history — assume healthy

        try:
            sub = df.iloc[max(0, current_index - 9): current_index + 1]
        except Exception:
            return 1.0

        is_long = position.is_long

        # ── 1. MACD_Hist slope ────────────────────────────────────────────
        macd_score = self._macd_slope_score(sub, is_long)

        # ── 2. RSI trajectory ─────────────────────────────────────────────
        rsi_score = self._rsi_slope_score(sub, is_long)

        # ── 3. ATR contraction ────────────────────────────────────────────
        atr_score = self._atr_score(sub)

        # ── 4. Price vs partial TP ────────────────────────────────────────
        bar_close = float(df.iloc[current_index].get("Close", 0.0) or 0.0)
        tp_score  = self._profit_position_score(position, bar_close)

        # ── 5. Time decay ─────────────────────────────────────────────────
        time_score = self._time_decay_score(position, df.index[current_index])

        # ── Weighted sum ──────────────────────────────────────────────────
        score = (
            0.30 * macd_score
            + 0.30 * rsi_score
            + 0.15 * atr_score
            + 0.15 * tp_score
            + 0.10 * time_score
        )

        logger.debug(
            "Stage3 score=%.3f | MACD=%.2f RSI=%.2f ATR=%.2f TP=%.2f Time=%.2f",
            score, macd_score, rsi_score, atr_score, tp_score, time_score,
        )
        return round(float(score), 4)

    # ------------------------------------------------------------------
    # Sub-score helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _macd_slope_score(sub: pd.DataFrame, is_long: bool) -> float:
        """Score 0–1 based on MACD histogram direction over last 3 bars."""
        if "MACD_Hist" not in sub.columns or len(sub) < 4:
            return 0.5
        vals = sub["MACD_Hist"].dropna().values
        if len(vals) < 2:
            return 0.5
        slope = float(vals[-1] - vals[max(0, len(vals)-4)])
        # Positive slope on long = good; negative slope on short = good
        if is_long:
            return _sigmoid_score(slope * 50)
        else:
            return _sigmoid_score(-slope * 50)

    @staticmethod
    def _rsi_slope_score(sub: pd.DataFrame, is_long: bool) -> float:
        """Score 0–1 based on RSI change over last 5 bars."""
        if "RSI" not in sub.columns or len(sub) < 6:
            return 0.5
        vals = sub["RSI"].dropna().values
        if len(vals) < 2:
            return 0.5
        slope = float(vals[-1] - vals[max(0, len(vals)-6)])
        if is_long:
            return _sigmoid_score(slope * 0.15)
        else:
            return _sigmoid_score(-slope * 0.15)

    @staticmethod
    def _atr_score(sub: pd.DataFrame) -> float:
        """Score 0–1 based on ATR expansion: expanding > 0.5, contracting < 0.5."""
        if "ATR" not in sub.columns or len(sub) < 5:
            return 0.5
        vals = sub["ATR"].dropna().values
        if len(vals) < 2:
            return 0.5
        ratio = vals[-1] / (vals[0] + 1e-10)
        # ratio > 1.0 = expanding volatility = slightly positive
        return _sigmoid_score((ratio - 1.0) * 5)

    @staticmethod
    def _profit_position_score(position, bar_close: float) -> float:
        """Score 0–1 based on unrealised profit vs SL distance."""
        try:
            sl_dist = abs(position.sl_distance)
            if sl_dist <= 0:
                return 0.5
            if position.is_long:
                profit = bar_close - position.entry_price
            else:
                profit = position.entry_price - bar_close
            r = profit / sl_dist
            # More profit in R → closer to 1.0
            return _sigmoid_score(r * 1.5)
        except Exception:
            return 0.5

    @staticmethod
    def _time_decay_score(position, current_ts) -> float:
        """Score decays as trade ages (beyond 50 bars = mild negative)."""
        try:
            bars_open = 0
            if hasattr(position, "open_time") and position.open_time is not None:
                delta = pd.Timestamp(current_ts) - pd.Timestamp(position.open_time)
                # Approximate: 15m bars
                bars_open = max(0, int(delta.total_seconds() / 900))
            # Fresh trade = 0.7, very old = 0.3 (time decay only mild)
            score = max(0.3, 0.7 - (bars_open / 200))
            return score
        except Exception:
            return 0.5

    # ------------------------------------------------------------------
    # Utility
    # ------------------------------------------------------------------

    @staticmethod
    def _unrealised_r(position, bar_close: float) -> float:
        """Unrealised profit measured in R-multiples (sl_distance = 1R)."""
        try:
            sl_dist = abs(position.sl_distance)
            if sl_dist <= 0:
                return 0.0
            if position.is_long:
                return (bar_close - position.entry_price) / sl_dist
            else:
                return (position.entry_price - bar_close) / sl_dist
        except Exception:
            return 0.0


def _sigmoid_score(x: float) -> float:
    """Map x ∈ (-∞, +∞) to (0, 1) with sigmoid. Clamp to [0.05, 0.95]."""
    try:
        v = 1.0 / (1.0 + math.exp(-x))
    except OverflowError:
        v = 0.0 if x < 0 else 1.0
    return max(0.05, min(0.95, v))
