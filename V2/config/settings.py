"""
V2/config/settings.py
---------------------
V2 Configuration — extends V1 settings wholesale.

All V1 parameters are inherited via wildcard import.
V2-specific overrides and additions are declared below.
"""

import os
import sys

# ── Inherit everything from V1 ────────────────────────────────────────────────
_V1_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'V1'))
if _V1_ROOT not in sys.path:
    sys.path.insert(0, _V1_ROOT)

from config.settings import *  # noqa: F401,F403  (intentional wildcard inherit)

# ── Stage 1: Market Regime Detection ─────────────────────────────────────────
# ADX thresholds for regime classification
REGIME_ADX_TREND   = 22.0   # ADX must exceed this for TREND_BULL / TREND_BEAR
REGIME_ADX_CHOPPY  = 18.0   # ADX below this → CHOPPY
# ATR expansion ratio that triggers BREAKOUT regime
REGIME_ATR_BREAKOUT = 1.5   # ATR > ATR_MA_20 * 1.5 → BREAKOUT
# BB width contraction ratio that confirms CHOPPY regime
REGIME_BB_CHOPPY   = 0.7    # BB_Width < BB_Width_MA_50 * 0.7 → CHOPPY

# Breakout regime adjustments
BREAKOUT_SL_MULT   = 0.5    # tighten SL to 0.5× normal ATR distance
BREAKOUT_SIZE_MULT = 0.7    # reduce position size to 70 % in breakout

# ── Stage 2: Ensemble Signal ──────────────────────────────────────────────────
ENSEMBLE_LGBM_WEIGHT  = 0.35   # weight of LightGBM model probability
ENSEMBLE_RULE_WEIGHT  = 0.65   # weight of V1 rule-engine signal
ENSEMBLE_MIN_SCORE    = 0.52   # minimum ensemble score to pass a LONG/SHORT
LGBM_MODEL_DIR        = os.path.join(os.path.dirname(__file__), '..', 'models')

# LightGBM confidence thresholds
LGBM_LONG_CONFIRM   = 0.52  # prob >= this AND rule_signal == LONG → LONG
LGBM_LONG_VETO      = 0.45  # prob <  this AND rule_signal == LONG → HOLD (veto)
# 0.45–0.52 range: rule engine wins alone

# ── Stage 3: Adaptive Exit ────────────────────────────────────────────────────
EXIT_MOMENTUM_THRESHOLD = 0.35  # score below this → EXIT_NOW
EXIT_MIN_PROFIT_R       = 0.25  # must have at least 0.25R profit to consider early exit
EXIT_TIGHTEN_THRESHOLD  = 0.45  # score below this (but above EXIT_MOMENTUM_THRESHOLD)
                                 #   → TIGHTEN_TRAIL
