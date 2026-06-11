"""
V2/backtest_v2.py
-----------------
Pars Pipeline V2 Backtester

Replaces V1's direct StrategyEngine + VotingMechanism calls with the
three-stage PipelineOrchestrator (Stage1 Regime → Stage2 Ensemble → Stage3 Exit).

All V1 modules (risk/, execution/, data/) are reused directly.

Usage
-----
    # From V2 directory:
    python backtest_v2.py

    # Specify a different CSV:
    python backtest_v2.py --csv /path/to/nas100.csv

    # Run only IS or OOS window (same env vars as V1):
    STRESS_EVAL_WINDOW=OOS python backtest_v2.py

Results
-------
    - Console: Quant performance metrics (regime breakdown, Sharpe, MaxDD, WR, etc.)
    - logs/backtest_v2_YYYYMMDD_HHMMSS.log
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import sys
from datetime import datetime, date
from typing import List

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

import numpy as np
import pandas as pd

# ── Path bootstrap ────────────────────────────────────────────────────────────
_V2_ROOT = os.path.dirname(os.path.abspath(__file__))
_V1_ROOT = os.path.normpath(os.path.join(_V2_ROOT, "..", "V1"))
for _p in (_V2_ROOT, _V1_ROOT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# ── V2 pipeline ───────────────────────────────────────────────────────────────
from pipeline.stage1_regime import RegimeDetector
from pipeline.stage2_signal import EnsembleSignal
from pipeline.stage3_exit   import AdaptiveExit
from pipeline.orchestrator  import PipelineOrchestrator

# ── V2 feature engine ─────────────────────────────────────────────────────────
from ml.features_v2 import FeatureEngineV2

# ── V1 modules (reused) ───────────────────────────────────────────────────────
from data.data_feed      import DataFeed
from risk.portfolio      import PortfolioManager
from risk.guardrails     import RiskGuardrails
from execution.simulator import BacktestSimulator

# ── Config ────────────────────────────────────────────────────────────────────
from config.settings import (
    INITIAL_BALANCE,
    MAX_DAILY_DRAWDOWN_PCT,
    MAX_TOTAL_DRAWDOWN_PCT,
    REWARD_RISK_RATIO,
    RISK_PER_TRADE,
    ATR_PERIOD,
    EMA_FAST,
    EMA_SLOW,
    RSI_PERIOD,
)

# ── Logging ───────────────────────────────────────────────────────────────────

def _setup_logging(log_dir: str) -> logging.Logger:
    os.makedirs(log_dir, exist_ok=True)
    ts  = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"backtest_v2_{ts}.log")
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger("backtest_v2")
    logger.info("Log: %s", log_path)
    return logger


# ── Warmup & simulation bounds (mirror V1) ────────────────────────────────────

def _warmup_bars() -> int:
    return max(EMA_SLOW, RSI_PERIOD, ATR_PERIOD, 200, 26) + 50  # V2 needs extra for BB_Width_MA


def _simulation_bounds(df: pd.DataFrame, warmup: int, logger: logging.Logger):
    window = os.environ.get("STRESS_EVAL_WINDOW", "FULL").upper()
    if window == "FULL":
        return warmup, len(df), "FULL"
    oos_days = int(os.environ.get("STRESS_OOS_DAYS", "90"))
    cutoff   = df.index.max() - pd.Timedelta(days=oos_days)
    if window == "IS":
        pos = np.flatnonzero(df.index <= cutoff)
        if not len(pos):
            raise ValueError("IS window is empty.")
        return warmup, int(pos[-1]) + 1, "IS"
    if window == "OOS":
        pos = np.flatnonzero(df.index > cutoff)
        if not len(pos):
            raise ValueError("OOS window is empty.")
        return max(warmup, int(pos[0])), len(df), "OOS"
    return warmup, len(df), "FULL"


# ── Quant metrics (copied / adapted from V1 backtest.py) ─────────────────────

def _sharpe(equity_curve: list, bars_per_year: int = 17_472) -> float:
    if len(equity_curve) < 2:
        return 0.0
    ret = np.diff(equity_curve) / np.array(equity_curve[:-1])
    std = np.std(ret, ddof=1)
    return float(np.mean(ret) / std * math.sqrt(bars_per_year)) if std > 0 else 0.0


def _max_drawdown(equity_curve: list):
    arr  = np.array(equity_curve)
    peak = np.maximum.accumulate(arr)
    dd   = peak - arr
    mdd_usd = float(np.max(dd))
    mdd_pct = float(np.max(dd / np.where(peak == 0, 1, peak)) * 100)
    return mdd_usd, mdd_pct


def _profit_factor(trade_log: list) -> float:
    gross_win  = sum(t["net_pnl"] for t in trade_log if t["net_pnl"] > 0)
    gross_loss = abs(sum(t["net_pnl"] for t in trade_log if t["net_pnl"] < 0))
    if gross_loss == 0:
        return float("inf") if gross_win > 0 else 0.0
    return round(gross_win / gross_loss, 3)


# ── Report ─────────────────────────────────────────────────────────────────────

def _print_report(
    portfolio:    PortfolioManager,
    simulator:    BacktestSimulator,
    equity_curve: list,
    regime_counts: dict,
    start_time:   datetime,
    logger:       logging.Logger,
) -> None:
    elapsed  = (datetime.now() - start_time).total_seconds()
    summary  = portfolio.summary()
    sim_st   = simulator.stats()
    log      = portfolio.trade_log

    total  = len(log)
    wins   = sum(1 for t in log if t["net_pnl"] > 0)
    wr     = wins / total * 100 if total else 0.0
    sharpe = _sharpe(equity_curve)
    mdd_u, mdd_p = _max_drawdown(equity_curve)
    pf     = _profit_factor(log)
    ret    = summary["total_pnl_pct"]

    sep = "═" * 64

    # Regime distribution
    regime_str = "  ".join(
        f"{r}={c}" for r, c in sorted(regime_counts.items(), key=lambda x: -x[1])
    )

    report = f"""
{sep}
  Pars Pipeline V2  |  BACKTEST RESULTS
{sep}
  Elapsed              : {elapsed:.1f}s
  Bars processed       : {sim_st['bars_processed']:,}

  ── PORTFOLIO ──────────────────────────────────────────────────
  Initial balance      : ${INITIAL_BALANCE:>12,.2f}
  Final balance        : ${summary['balance']:>12,.2f}
  Net PnL              : ${summary['total_pnl']:>+12,.2f}  ({ret:+.2f}%)
  Max Drawdown         : ${mdd_u:>12,.2f}  ({mdd_p:.2f}%)
  Total commission     : ${summary['total_commission']:>12,.2f}

  ── TRADE STATS ────────────────────────────────────────────────
  Total trades         : {total}
  Wins / Losses        : {wins} / {total - wins}
  Win rate             : {wr:.1f}%
  Profit factor        : {pf:.3f}
  Sharpe ratio         : {sharpe:.4f}
  R:R target           : 1:{REWARD_RISK_RATIO}

  ── CLOSE REASONS ──────────────────────────────────────────────
  TP Hit               : {sim_st['tp_hits']}
  SL Hit               : {sim_st['sl_hits']}
  Early Exit (Stage3)  : {sum(1 for t in log if t.get('reason') == 'EARLY_EXIT')}
  Weekend Flatten      : {sim_st['weekend_flattens']}
  Guardrail            : {sim_st['guardrail_closes']}

  ── REGIME DISTRIBUTION ────────────────────────────────────────
  {regime_str}

  ── LIMITS ─────────────────────────────────────────────────────
  Daily DD limit       : {MAX_DAILY_DRAWDOWN_PCT*100:.1f}%
  Total DD limit       : {MAX_TOTAL_DRAWDOWN_PCT*100:.1f}%
{sep}"""

    logger.info(report)

    # Prop firm verdict
    dd_from_initial = max(0, (INITIAL_BALANCE - min(equity_curve)) / INITIAL_BALANCE * 100) if equity_curve else 0
    passed = dd_from_initial < MAX_TOTAL_DRAWDOWN_PCT * 100 and ret > 0 and sharpe > 0
    verdict = "PROP FIRM THRESHOLDS: PASSABLE" if passed else "PROP FIRM THRESHOLDS: FAILED — strategy needs revision"
    logger.info("  %s\n%s\n", verdict, sep)


# ── Main backtest loop ────────────────────────────────────────────────────────

def run_backtest_v2(csv_path: str | None = None) -> None:
    log_dir = os.path.join(_V2_ROOT, "logs")
    logger  = _setup_logging(log_dir)
    logger.info("=" * 64)
    logger.info("  Pars Pipeline V2 BACKTEST STARTED")
    logger.info("=" * 64)
    start_time = datetime.now()

    # ── Load data ─────────────────────────────────────────────────────
    if csv_path and os.path.exists(csv_path):
        logger.info("Loading CSV: %s", csv_path)
        raw = pd.read_csv(csv_path, parse_dates=["Datetime"], index_col="Datetime")
        raw.sort_index(inplace=True)
    else:
        # Fall back to V1 DataFeed (checks CSV_DIR / yfinance cache)
        logger.info("Using V1 DataFeed...")
        raw = DataFeed().download_historical_data()

    if raw is None or raw.empty:
        logger.error("No data available. Aborting.")
        return

    # ── Calculate V2 indicators ───────────────────────────────────────
    logger.info("Calculating V2 indicators...")
    fe = FeatureEngineV2()
    df = fe.calculate_indicators(raw)

    warmup              = _warmup_bars()
    sim_start, sim_end, eval_window = _simulation_bounds(df, warmup, logger)
    logger.info(
        "Bars total=%d  warmup=%d  simulate=%d  window=%s",
        len(df), warmup, sim_end - sim_start, eval_window,
    )

    # ── Build components ──────────────────────────────────────────────
    portfolio  = PortfolioManager()
    guardrails = RiskGuardrails()
    simulator  = BacktestSimulator()

    # Build orchestrator and wire feature engine for ML
    orchestrator = PipelineOrchestrator()
    orchestrator.setup(portfolio, guardrails, simulator, feature_engine=fe)

    # ── Simulation loop ───────────────────────────────────────────────
    equity_curve:  List[float] = []
    regime_counts: dict        = {}
    current_day:   date | None = None
    _kill_warned_day: date | None = None

    logger.info("Starting simulation loop...")

    for idx in range(sim_start, sim_end):
        current_bar = df.iloc[idx]
        timestamp   = df.index[idx]
        bar_date    = pd.Timestamp(timestamp).date()

        # Daily reset
        if current_day is None:
            current_day = bar_date
        elif bar_date > current_day:
            guardrails.reset_daily_drawdown(portfolio)
            portfolio.reset_daily_peak()
            current_day = bar_date

        equity_curve.append(portfolio.equity)

        # Kill-switch checks
        if guardrails.total_drawdown_triggered:
            logger.critical("Total DD limit breached. Stopping at bar %d.", idx)
            break

        if guardrails.kill_switch_active:
            if _kill_warned_day != current_day:
                _kill_warned_day = current_day
                logger.warning("Daily DD limit hit (%s) — no new trades today.", current_day)
            # Still update open positions via simulator
            if portfolio.open_position is not None:
                simulator.update_and_check_positions(current_bar, portfolio, guardrails)
            continue

        # ── Simulator: SL / TP / trailing on open position ───────────
        if portfolio.open_position is not None:
            sim_result = simulator.update_and_check_positions(current_bar, portfolio, guardrails)
            if sim_result is not None:
                # Position closed by SL/TP/guardrail — orchestrator should not try to close again
                pass
            # Note: if still open, orchestrator.process_bar handles Stage3 exit below

        # ── Trading allowed? ──────────────────────────────────────────
        if not guardrails.is_trading_allowed(portfolio, timestamp):
            continue

        # ── Pipeline Orchestrator ─────────────────────────────────────
        result = orchestrator.process_bar(df, idx, timestamp, current_bar)

        # Track regime distribution
        regime = result["regime"]
        regime_counts[regime] = regime_counts.get(regime, 0) + 1

        if result["trade_opened"]:
            guardrails.daily_trades_count += 1

    # Close any still-open position at end of simulation
    if portfolio.open_position is not None:
        last_close = float(df.iloc[sim_end - 1]["Close"])
        last_ts    = df.index[sim_end - 1]
        portfolio.close_trade(last_close, last_ts, reason="FORCE_CLOSE")
        logger.info("Open position force-closed at simulation end.")

    # ── Report ─────────────────────────────────────────────────────────
    _print_report(portfolio, simulator, equity_curve, regime_counts, start_time, logger)

    # Persist equity curve on portfolio for HTML report compatibility
    portfolio.equity_curve     = equity_curve
    portfolio.equity_timestamps = [str(ts) for ts in df.index[sim_start:sim_end]]

    # Try V1 HTML report generator
    try:
        sys.path.insert(0, _V1_ROOT)
        import results_generator
        scenario_name = os.environ.get("STRESS_SCENARIO_NAME", "V2_Pars_Pipeline")
        results_generator.generate_html_report(portfolio, INITIAL_BALANCE, scenario_name)
    except Exception as exc:
        logger.warning("HTML report skipped: %s", exc)


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pars Pipeline V2 Backtest")
    parser.add_argument(
        "--csv",
        default=None,
        help=(
            "Path to NAS100 15m CSV file. "
            "Defaults to V1/cache/cache_15m_360d.csv if it exists, "
            "otherwise uses V1 DataFeed."
        ),
    )
    args = parser.parse_args()

    # Try default cache location
    csv_path = args.csv
    if csv_path is None:
        default = os.path.join(_V1_ROOT, "cache", "cache_15m_360d.csv")
        if os.path.exists(default):
            csv_path = default

    run_backtest_v2(csv_path=csv_path)
