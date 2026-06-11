"""
V2/backtest_v2.py
-----------------
Pars Pipeline Backtest — V2

Runs a bar-by-bar simulation using the three-stage pipeline:
  Stage 1: RegimeDetector   → market regime classification
  Stage 2: EnsembleSignal   → V1 rule engine + LightGBM ensemble
  Stage 3: AdaptiveExit     → momentum-based early exit

Leverages V1 components unchanged:
  - FeatureEngineV2  (extends V1 FeatureEngine)
  - PortfolioManager (V1 risk/portfolio.py)
  - RiskGuardrails   (V1 risk/guardrails.py)
  - BacktestSimulator (V1 execution/simulator.py)
  - DataFeed          (V1 data/data_feed.py)

Usage
-----
    cd /home/user/nasdaq-bot/V2
    python backtest_v2.py

    # Override CSV path:
    CSV_PATH=/data/nas100.csv python backtest_v2.py

Output
------
  - Console: performance summary (Sharpe, MaxDD, WR, PF)
  - V2/logs/backtest_v2_YYYYMMDD_HHMMSS.log
"""

from __future__ import annotations

import logging
import math
import os
import sys
from datetime import datetime, date
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# ── Path bootstrap ────────────────────────────────────────────────────────────
_V2_ROOT = Path(__file__).resolve().parent
_V1_ROOT = _V2_ROOT.parent / "V1"

# Ensure sys.path order: V2 first, V1 second.
# When running as `python V2/backtest_v2.py`, Python auto-inserts V2/ at index 0.
# We must guarantee V1 is also on the path (append rather than insert to keep V2 at 0).
_paths_to_add = [str(_V2_ROOT), str(_V1_ROOT)]
for _p in _paths_to_add:
    # Remove duplicates, then re-insert in correct order at positions 0 and 1
    while _p in sys.path:
        sys.path.remove(_p)
# Now insert V1 at 0, then V2 at 0 → V2 ends up at 0, V1 at 1
sys.path.insert(0, str(_V1_ROOT))
sys.path.insert(0, str(_V2_ROOT))

# Encoding safety
try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except AttributeError:
    pass

# ── V2 config ─────────────────────────────────────────────────────────────────
os.environ.setdefault("STRESS_TEST_MODE", "1")  # suppress MT5 password warning

from config.settings import (  # noqa: E402
    INITIAL_BALANCE,
    RISK_PER_TRADE,
    REWARD_RISK_RATIO,
    MAX_DAILY_DRAWDOWN_PCT,
    MAX_TOTAL_DRAWDOWN_PCT,
    MAX_OPEN_POSITIONS,
    MAX_SECTOR_POSITIONS,
    SECTOR_MAP,
    EMA_FAST,
    EMA_SLOW,
    RSI_PERIOD,
    ATR_PERIOD,
    BREAKOUT_SL_MULT,
)

# ── V2 pipeline ───────────────────────────────────────────────────────────────
from pipeline.orchestrator import PipelineOrchestrator  # noqa: E402

# ── V2 feature engine ─────────────────────────────────────────────────────────
from ml.features_v2 import FeatureEngineV2  # noqa: E402

# ── V1 risk / execution ───────────────────────────────────────────────────────
from risk.portfolio  import PortfolioManager   # noqa: E402
from risk.guardrails import RiskGuardrails     # noqa: E402
from execution.simulator import BacktestSimulator  # noqa: E402
from data.data_feed import DataFeed            # noqa: E402


# ── Logging ───────────────────────────────────────────────────────────────────

def _setup_logging() -> logging.Logger:
    log_dir = _V2_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = log_dir / f"backtest_v2_{ts}.log"

    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(str(log_path), encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger("backtest_v2")
    logger.info("Log: %s", log_path)
    return logger


# ── Warm-up ───────────────────────────────────────────────────────────────────

def _warmup_bars() -> int:
    return max(EMA_SLOW, RSI_PERIOD, ATR_PERIOD, 200, 26) + 50 + 10


# ── Metrics ──────────────────────────────────────────────────────────────────

def _sharpe(equity_curve: List[float], bars_per_year: int = 17_472) -> float:
    if len(equity_curve) < 2:
        return 0.0
    returns = np.diff(equity_curve) / np.array(equity_curve[:-1])
    std = np.std(returns, ddof=1)
    if std == 0:
        return 0.0
    return float(np.mean(returns) / std * math.sqrt(bars_per_year))


def _max_dd(equity_curve: List[float]) -> tuple[float, float]:
    arr  = np.array(equity_curve)
    peak = np.maximum.accumulate(arr)
    dd   = peak - arr
    mdd_usd = float(np.max(dd))
    mdd_pct = float(np.max(dd / np.where(peak == 0, 1, peak)) * 100)
    return mdd_usd, mdd_pct


def _aggregate_trades(trade_log: list) -> list:
    from collections import defaultdict
    groups: dict = defaultdict(lambda: {"net_pnl": 0.0, "reason": ""})
    for t in trade_log:
        key = t.get("open_time", id(t))
        groups[key]["net_pnl"] += t.get("net_pnl", 0.0)
        if t.get("reason") not in (None, "PARTIAL_TP"):
            groups[key]["reason"] = t.get("reason", "")
    return list(groups.values())


def _profit_factor(trade_log: list) -> float:
    agg = _aggregate_trades(trade_log)
    gross_profit = sum(t["net_pnl"] for t in agg if t["net_pnl"] > 0)
    gross_loss   = abs(sum(t["net_pnl"] for t in agg if t["net_pnl"] < 0))
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return round(gross_profit / gross_loss, 3)


# ── Report ────────────────────────────────────────────────────────────────────

def _print_report(
    portfolio: PortfolioManager,
    simulator: BacktestSimulator,
    equity_curve: List[float],
    start_time: datetime,
    logger: logging.Logger,
) -> None:
    elapsed   = (datetime.now() - start_time).total_seconds()
    summary   = portfolio.summary()
    sim_stats = simulator.stats()

    agg    = _aggregate_trades(portfolio.trade_log)
    total  = len(agg)
    wins   = sum(1 for t in agg if t["net_pnl"] > 0)
    losses = total - wins
    wr     = wins / total * 100 if total else 0.0

    sharpe            = _sharpe(equity_curve)
    mdd_usd, mdd_pct  = _max_dd(equity_curve)
    pf                = _profit_factor(portfolio.trade_log)

    sep = "=" * 64
    report = f"""
{sep}
  V2 Pars Pipeline  |  BACKTEST RESULTS
{sep}
  Elapsed             : {elapsed:.1f}s
  Bars processed      : {sim_stats['bars_processed']:,}

  -- PORTFOLIO --------------------------------------------------
  Initial Balance     : ${INITIAL_BALANCE:>12,.2f}
  Final Balance       : ${summary['balance']:>12,.2f}
  Net PnL             : ${summary['total_pnl']:>+12,.2f}  ({summary['total_pnl_pct']:+.2f}%)
  Max Drawdown        : ${mdd_usd:>12,.2f}  ({mdd_pct:.2f}%)
  Total Commission    : ${summary['total_commission']:>12,.2f}

  -- TRADE STATS ------------------------------------------------
  Total trades        : {total}
  Winning             : {wins}
  Losing              : {losses}
  Win Rate            : {wr:.1f}%
  Profit Factor       : {pf:.3f}

  -- CLOSE REASONS ----------------------------------------------
  TP Hit              : {sim_stats['tp_hits']}
  SL Hit              : {sim_stats['sl_hits']}
  Adaptive Exit       : {getattr(simulator, 'adaptive_exits', 0)}
  Weekend Flatten     : {sim_stats['weekend_flattens']}
  Guardrail           : {sim_stats['guardrail_closes']}

  -- QUANT METRICS ----------------------------------------------
  Sharpe Ratio        : {sharpe:.4f}
  R:R Target          : 1:{REWARD_RISK_RATIO}
  Daily DD Limit      : {MAX_DAILY_DRAWDOWN_PCT*100:.1f}%
  Total DD Limit      : {MAX_TOTAL_DRAWDOWN_PCT*100:.1f}%
{sep}"""
    logger.info(report)


# ── Core simulation loop ──────────────────────────────────────────────────────

def run_backtest(csv_path: str | None = None) -> dict:
    """
    Run the V2 Pars Pipeline backtest.

    Parameters
    ----------
    csv_path : optional path to CSV; falls back to V1 DataFeed if None

    Returns
    -------
    dict with summary metrics
    """
    logger = _setup_logging()
    start_time = datetime.now()

    logger.info("=" * 64)
    logger.info("  V2 PARS PIPELINE BACKTEST STARTING")
    logger.info("=" * 64)

    # ── 1. Data ───────────────────────────────────────────────────────────
    if csv_path and os.path.exists(csv_path):
        logger.info("Loading CSV via V1 DataFeed: %s", csv_path)
        from data.data_feed import _load_mt5_csv, _ensure_spy_vix
        raw_df = _load_mt5_csv(csv_path)
        raw_df = _ensure_spy_vix(raw_df)
    else:
        logger.info("No CSV provided — using V1 DataFeed...")
        raw_df = DataFeed().download_historical_data()

    # Apply BACKTEST_DAYS window (same as V1 DataFeed)
    try:
        from config.settings import BACKTEST_DAYS
    except ImportError:
        BACKTEST_DAYS = 360
    from datetime import timedelta
    cutoff = raw_df.index.max() - timedelta(days=BACKTEST_DAYS)
    raw_df = raw_df[raw_df.index >= cutoff]
    logger.info("Raw data (last %d days): %d bars | %s → %s",
                BACKTEST_DAYS, len(raw_df), raw_df.index[0], raw_df.index[-1])

    # ── 2. Feature engineering ───────────────────────────────────────────
    logger.info("Computing V2 indicators...")
    feature_engine = FeatureEngineV2()
    df = feature_engine.calculate_indicators(raw_df)

    # ── 3. Init components ───────────────────────────────────────────────
    portfolio  = PortfolioManager()
    guardrails = RiskGuardrails()
    simulator  = BacktestSimulator()
    simulator.adaptive_exits = 0  # track adaptive exits

    orchestrator = PipelineOrchestrator()
    orchestrator.setup(portfolio, guardrails, simulator, feature_engine)

    # ── 4. Simulation loop ───────────────────────────────────────────────
    warmup    = _warmup_bars()
    sim_start = warmup
    sim_end   = len(df)

    logger.info(
        "Bars: total=%d | warmup=%d | simulating %d bars",
        len(df), warmup, sim_end - sim_start,
    )

    equity_curve: List[float] = []
    current_day: date | None  = None
    _kill_warned_day: date | None = None

    for idx in range(sim_start, sim_end):
        bar       = df.iloc[idx]
        timestamp = df.index[idx]
        bar_date  = pd.Timestamp(timestamp).date()

        # Daily reset
        if current_day is None:
            current_day = bar_date
        elif bar_date > current_day:
            guardrails.reset_daily_drawdown(portfolio)
            portfolio.reset_daily_peak()
            current_day = bar_date

        equity_curve.append(portfolio.equity)

        # Kill switches
        if guardrails.total_drawdown_triggered:
            logger.critical("Total DD limit hit — stopping simulation at bar %d", idx)
            break

        # ── Step 1: V1 simulator — SL/TP/trailing/weekend flatten ────────
        # Must run every bar regardless of kill switch (to close open positions)
        if portfolio.open_position is not None:
            simulator.update_and_check_positions(bar, portfolio, guardrails)

        if guardrails.kill_switch_active:
            if _kill_warned_day != current_day:
                _kill_warned_day = current_day
                logger.warning("Daily DD limit hit on %s — no new trades today", current_day)
            continue

        if not guardrails.is_trading_allowed(portfolio, timestamp):
            continue

        # ── Step 2: Skip if position already open ────────────────────────
        if portfolio.open_position is not None:
            continue

        # ── Step 3: Pipeline — regime + ensemble signal (Stage 1 & 2) ────
        regime = orchestrator.stage1.detect(df, idx)
        signal, score = orchestrator.stage2.generate(df, idx, regime, feature_engine)

        # ── Step 4: Execute signal via portfolio (same as V1) ─────────────
        if signal in ("STRONG_LONG", "STRONG_SHORT"):
            close_price = float(bar.get("Close", 0.0))
            atr_value   = float(bar.get("ATR", 0.0) or 0.0)
            if atr_value > 0:
                # BREAKOUT regime: tighten SL distance
                effective_atr = atr_value * BREAKOUT_SL_MULT if regime == "BREAKOUT" else atr_value
                portfolio.open_trade(
                    direction=signal,
                    current_price=close_price,
                    atr_value=effective_atr,
                    timestamp=timestamp,
                )
                if portfolio.open_position is not None:
                    guardrails.daily_trades_count += 1

    # Close any still-open position at end of simulation
    if portfolio.open_position is not None:
        last_bar   = df.iloc[sim_end - 1]
        last_close = float(last_bar["Close"])
        last_ts    = df.index[sim_end - 1]
        portfolio.close_trade(last_close, last_ts, reason="FORCE_CLOSE_EOD")
        logger.info("Open position force-closed at end of simulation.")

    # ── 5. Report ─────────────────────────────────────────────────────────
    _print_report(portfolio, simulator, equity_curve, start_time, logger)

    summary = portfolio.summary()
    mdd_usd, mdd_pct = _max_dd(equity_curve)
    return {
        "net_pnl":        summary["total_pnl"],
        "net_pnl_pct":    summary["total_pnl_pct"],
        "win_rate":       summary.get("win_rate_pct", 0.0),
        "sharpe":         _sharpe(equity_curve),
        "max_dd_pct":     mdd_pct,
        "profit_factor":  _profit_factor(portfolio.trade_log),
        "total_trades":   len(_aggregate_trades(portfolio.trade_log)),
        "adaptive_exits": getattr(simulator, "adaptive_exits", 0),
    }


# ── CLI / demo ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Allow CSV path via env var or first CLI argument
    _csv = os.environ.get("CSV_PATH") or (sys.argv[1] if len(sys.argv) > 1 else None)

    # If no CSV, try V1 default cache location
    if _csv is None:
        _candidates = [
            str(_V1_ROOT / "csv" / "nasdaq_15m.csv"),
            str(_V1_ROOT / "cache" / "cache_15m_360d.csv"),
        ]
        for _c in _candidates:
            if os.path.exists(_c):
                _csv = _c
                break

    print(f"\nV2 Pars Pipeline Backtest")
    print(f"CSV: {_csv or 'V1 DataFeed (yfinance)'}")
    print("-" * 50)

    try:
        results = run_backtest(csv_path=_csv)
        print("\n-- SUMMARY --")
        for k, v in results.items():
            if isinstance(v, float):
                print(f"  {k:<20}: {v:.4f}")
            else:
                print(f"  {k:<20}: {v}")
    except KeyboardInterrupt:
        print("\nBacktest interrupted by user.")
    except Exception as exc:
        print(f"\nBacktest failed: {exc}")
        raise
