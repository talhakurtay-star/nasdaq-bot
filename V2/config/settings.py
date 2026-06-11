"""
V2/config/settings.py
---------------------
V2 Configuration — extends V1 settings wholesale.

All V1 parameters are inherited via wildcard import.
V2-specific overrides and additions are declared below.
"""

import os
import sys
import importlib.util

# ── Inherit everything from V1 via direct file load (avoids circular import) ──
# We cannot use `from config.settings import *` because when V2 is first on
# sys.path, that resolves back to this file itself → circular import.
# Instead, load V1/config/settings.py directly via importlib.
_V1_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'V1'))
_V1_SETTINGS = os.path.join(_V1_ROOT, 'config', 'settings.py')

_spec = importlib.util.spec_from_file_location("_v1_settings", _V1_SETTINGS)
_v1_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_v1_mod)  # type: ignore[arg-type]

# Inject all V1 public names into this module's namespace
for _k, _v in vars(_v1_mod).items():
    if not _k.startswith('__'):
        globals()[_k] = _v

# Also make V1 available for submodule imports (risk/, data/, etc.)
if _V1_ROOT not in sys.path:
    sys.path.append(_V1_ROOT)

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
