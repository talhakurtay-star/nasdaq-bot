"""
V2/pipeline/orchestrator.py
----------------------------
Pipeline Orchestrator: Stage1 → Stage2 → Stage3

The orchestrator is the top-level entry point used by backtest_v2.py and
(eventually) the live engine.  It wires all three stages together and
applies the regime-based overrides defined in V2 config.

Bar-by-bar flow
---------------
1.  regime = stage1.detect(df, idx)
2.  signal, score = stage2.generate(df, idx, regime, feature_engine)
3.  If position open:
        exit_action = stage3.evaluate(df, idx, position, regime)
        Apply exit_action → may close trade or tighten trail
4.  If signal != HOLD and no position:
        Adjust position size / SL for BREAKOUT regime
        Open trade via PortfolioManager

Usage
-----
    from pipeline.orchestrator import PipelineOrchestrator

    orch = PipelineOrchestrator()
    orch.setup(portfolio, guardrails, simulator, feature_engine)
    result = orch.process_bar(df, current_index, timestamp, current_bar)
"""

from __future__ import annotations

import logging
import os
import sys
from typing import Optional, TYPE_CHECKING

import pandas as pd

# ── Path bootstrap ────────────────────────────────────────────────────────────
_V2_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..'))
_V1_ROOT = os.path.normpath(os.path.join(os.path.dirname(__file__), '..', '..', 'V1'))
for _p in (_V2_ROOT, _V1_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── Stage imports ─────────────────────────────────────────────────────────────
from pipeline.stage1_regime import RegimeDetector, RegimeType
from pipeline.stage2_signal import EnsembleSignal
from pipeline.stage3_exit   import AdaptiveExit

try:
    from config.settings import (
        BREAKOUT_SL_MULT,
        BREAKOUT_SIZE_MULT,
        EXIT_MIN_PROFIT_R,
        ATR_MULTIPLIER,
        TRAILING_ATR_MULT,
    )
except ImportError:
    BREAKOUT_SL_MULT   = 0.5
    BREAKOUT_SIZE_MULT = 0.7
    EXIT_MIN_PROFIT_R  = 0.25
    ATR_MULTIPLIER     = 2.5
    TRAILING_ATR_MULT  = 3.5

if TYPE_CHECKING:
    from risk.portfolio  import PortfolioManager, Position
    from risk.guardrails import RiskGuardrails
    from execution.simulator import BacktestSimulator

logger = logging.getLogger(__name__)


class PipelineOrchestrator:
    """
    Main V2 pipeline that connects Stage1 → Stage2 → Stage3.

    Lifecycle
    ---------
    1. Instantiate: ``orch = PipelineOrchestrator()``
    2. Wire components: ``orch.setup(portfolio, guardrails, simulator, feature_engine)``
    3. Per-bar call: ``result = orch.process_bar(df, idx, timestamp, bar)``
    """

    def __init__(self) -> None:
        self.stage1 = RegimeDetector()
        self.stage2 = EnsembleSignal()
        self.stage3 = AdaptiveExit()

        # Components wired via setup()
        self.portfolio:     Optional["PortfolioManager"]    = None
        self.guardrails:    Optional["RiskGuardrails"]      = None
        self.simulator:     Optional["BacktestSimulator"]   = None
        self.feature_engine = None

        # Runtime state
        self._tighten_trail_active: bool  = False
        self._tighten_trail_atr:    float = 1.5  # ATR mult when tightening

        logger.info("PipelineOrchestrator V2 initialized.")

    def setup(
        self,
        portfolio,
        guardrails,
        simulator,
        feature_engine=None,
    ) -> None:
        """Wire runtime components into the orchestrator."""
        self.portfolio      = portfolio
        self.guardrails     = guardrails
        self.simulator      = simulator
        self.feature_engine = feature_engine
        logger.info("PipelineOrchestrator.setup() done — all components wired.")

    # ------------------------------------------------------------------
    # Main per-bar entry point
    # ------------------------------------------------------------------

    def process_bar(
        self,
        df: pd.DataFrame,
        current_index: int,
        timestamp: pd.Timestamp,
        current_bar: pd.Series,
    ) -> dict:
        """
        Process a single bar through the full pipeline.

        Parameters
        ----------
        df            : DataFrame with all indicators calculated
        current_index : integer iloc position of current bar
        timestamp     : pandas Timestamp of current bar
        current_bar   : pd.Series for current bar (df.iloc[current_index])

        Returns
        -------
        dict with keys:
            regime        : RegimeType string
            signal        : "STRONG_LONG" | "STRONG_SHORT" | "HOLD"
            score         : float ensemble confidence
            exit_action   : "EXIT_NOW" | "TIGHTEN_TRAIL" | "HOLD_POSITION" | None
            trade_opened  : bool
            trade_closed  : bool (early exit)
            close_reason  : str | None
        """
        result = {
            "regime":       "TRANSITION",
            "signal":       "HOLD",
            "score":        0.0,
            "exit_action":  None,
            "trade_opened": False,
            "trade_closed": False,
            "close_reason": None,
        }

        # ── Stage 1: Regime Detection ─────────────────────────────────────
        regime = self.stage1.detect(df, current_index)
        result["regime"] = regime

        # ── Stage 3: Evaluate open position (if any) ─────────────────────
        pos = self.portfolio.open_position if self.portfolio else None
        if pos is not None:
            exit_action = self.stage3.evaluate(df, current_index, pos, regime)
            result["exit_action"] = exit_action

            if exit_action == "EXIT_NOW":
                close_price = float(current_bar.get("Close", 0.0))
                pnl, _ = self.portfolio.close_trade(
                    close_price, timestamp, reason="EARLY_EXIT"
                )
                result["trade_closed"] = True
                result["close_reason"] = "EARLY_EXIT"
                logger.info(
                    "Stage3 early exit | bar=%d | PnL=%+.2f | regime=%s",
                    current_index, pnl, regime,
                )
                self._tighten_trail_active = False
                return result

            elif exit_action == "TIGHTEN_TRAIL":
                # Flag for simulator to use reduced trail ATR
                self._tighten_trail_active = True
                # Modify the position's trailing directly
                if hasattr(pos, "atr_at_open") and pos.atr_at_open > 0:
                    bar_high  = float(current_bar.get("High", 0.0))
                    bar_low   = float(current_bar.get("Low",  0.0))
                    trail_dist = pos.atr_at_open * self._tighten_trail_atr
                    if pos.is_long:
                        new_sl = bar_high - trail_dist
                        if new_sl > pos.stop_loss:
                            pos.stop_loss = new_sl
                            logger.debug(
                                "TIGHTEN_TRAIL: new SL=%.4f (ATR×%.1f)",
                                new_sl, self._tighten_trail_atr,
                            )
                    else:
                        new_sl = bar_low + trail_dist
                        if new_sl < pos.stop_loss:
                            pos.stop_loss = new_sl
            else:
                self._tighten_trail_active = False

        # ── Stage 2: Signal Generation ────────────────────────────────────
        if pos is not None:
            # Position already open — no new entry signal needed
            result["signal"] = "HOLD"
            return result

        signal, score = self.stage2.generate(
            df, current_index, regime, self.feature_engine
        )
        result["signal"] = signal
        result["score"]  = score

        # ── Open trade if signal is actionable ────────────────────────────
        if signal in ("STRONG_LONG", "STRONG_SHORT") and self.portfolio is not None:
            close_price = float(current_bar.get("Close", 0.0))
            atr_value   = float(current_bar.get("ATR", 0.0) or 0.0)

            if atr_value <= 0:
                logger.debug("Bar %d: ATR=0, skipping entry.", current_index)
                return result

            # BREAKOUT regime: scale down SL distance and position size
            effective_atr = atr_value
            if regime == "BREAKOUT":
                effective_atr = atr_value * BREAKOUT_SL_MULT
                logger.debug(
                    "BREAKOUT regime: ATR scaled %.4f → %.4f",
                    atr_value, effective_atr,
                )

            pos_opened = self.portfolio.open_trade(
                direction=signal,
                current_price=close_price,
                atr_value=effective_atr,
                timestamp=timestamp,
            )
            if pos_opened is not None:
                result["trade_opened"] = True
                logger.info(
                    "Trade opened | %s | bar=%d | score=%.3f | regime=%s",
                    signal, current_index, score, regime,
                )

        return result

    # ------------------------------------------------------------------
    # Introspection helpers
    # ------------------------------------------------------------------

    def current_regime(self, df: pd.DataFrame, current_index: int) -> str:
        """Return the regime for the current bar (convenience wrapper)."""
        return self.stage1.detect(df, current_index)

    @property
    def is_trail_tightened(self) -> bool:
        """True if Stage 3 signalled to tighten the trailing stop."""
        return self._tighten_trail_active
