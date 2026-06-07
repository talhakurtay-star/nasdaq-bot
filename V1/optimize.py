"""
optimize.py
-----------
Walk-forward optimizer for the trade bot.

For each parameter combination:
  1. Train long/short XGBoost models on the in-sample window (first ~270 days).
  2. Run an in-sample backtest.
  3. Run an out-of-sample backtest on the last ~90 days using the same models.
  4. Select a champion only if OOS stays prop-firm safe and profitable.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

sys.stdout.reconfigure(encoding="utf-8")
sys.stderr.reconfigure(encoding="utf-8")

ROOT = Path(__file__).resolve().parent
TRAINER_SCRIPT = ROOT / "ml" / "trainer.py"
BACKTEST_SCRIPT = ROOT / "backtest.py"
METRICS_FILE = ROOT / "temp_metrics.json"
RESULTS_FILE = ROOT / "optimize_results.json"
LEADERBOARD_FILE = ROOT / "optimize_leaderboard.txt"
SETTINGS_FILE = ROOT / "config" / "settings.py"

MAX_DAILY_DD_PCT = 4.0
MAX_TOTAL_DD_PCT = 9.0
OOS_DAYS = 90
MIN_TRADES_IS = 75
MIN_TRADES_OOS = 20
TRAIN_TIMEOUT = 300
BACKTEST_TIMEOUT = 180

SEARCH_SPACE: Dict[str, List[Any]] = {
    "STRESS_XGB_THR": [0.54, 0.55, 0.56],
    "STRESS_RISK_PCT": [0.50, 0.75, 1.00],
    "STRESS_RR_RATIO": [1.2, 1.5, 1.8],
    "STRESS_ALLOW_WEEKEND": ["False"],
}

FIXED_ENV: Dict[str, str] = {
    "STRESS_OOS_DAYS": str(OOS_DAYS),
    "STRESS_START_HOUR": "13",
    "STRESS_END_HOUR": "23",
    "STRESS_BLOCK_FRIDAY": "False",
    "STRESS_THURSDAY_CUTOFF": "23",
    "STRESS_MAX_LOT": "1000.0",
    "STRESS_ATR_FLOOR": "0.0",
    "STRESS_MAX_NOTIONAL_LEVERAGE": "20.0",
    "STRESS_COST_RATIO": "0.75",
    "STRESS_RSI_OB": "80",
    "STRESS_RSI_OS": "20",
}


def _all_combinations() -> List[Dict[str, str]]:
    keys = list(SEARCH_SPACE.keys())
    values = list(SEARCH_SPACE.values())
    return [{k: str(v) for k, v in zip(keys, combo)} for combo in itertools.product(*values)]


def _random_combinations(n: int) -> List[Dict[str, str]]:
    combos = _all_combinations()
    random.shuffle(combos)
    return combos[:n]


def _build_env(combo: Dict[str, str], extra: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    env = os.environ.copy()
    env.update(FIXED_ENV)
    env.update(combo)
    if extra:
        env.update(extra)
    env["PYTHONIOENCODING"] = "utf-8"
    return env


def _clean_metrics() -> None:
    if METRICS_FILE.exists():
        METRICS_FILE.unlink()


def _read_metrics() -> Optional[Dict[str, Any]]:
    if not METRICS_FILE.exists():
        return None
    try:
        return json.loads(METRICS_FILE.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return None


def _run_process(label: str, script: Path, env: Dict[str, str], timeout: int) -> bool:
    try:
        result = subprocess.run(
            [sys.executable, str(script)],
            env=env,
            cwd=str(ROOT),
            capture_output=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        print(f"{label}: TIMEOUT after {timeout}s")
        return False

    if result.returncode != 0:
        stderr = result.stderr.decode("utf-8", errors="replace")
        stdout = result.stdout.decode("utf-8", errors="replace")
        print(f"{label}: FAILED exit={result.returncode}")
        print((stderr or stdout)[-800:])
        return False
    return True


def _run_training(combo: Dict[str, str]) -> bool:
    env = _build_env(combo, {"STRESS_TRAIN_WINDOW": "IS"})
    return _run_process("TRAIN(IS)", TRAINER_SCRIPT, env, TRAIN_TIMEOUT)


def _run_backtest(combo: Dict[str, str], window: str) -> Optional[Dict[str, Any]]:
    _clean_metrics()
    env = _build_env(combo, {"STRESS_EVAL_WINDOW": window})
    ok = _run_process(f"BACKTEST({window})", BACKTEST_SCRIPT, env, BACKTEST_TIMEOUT)
    if not ok:
        return None
    metrics = _read_metrics()
    if metrics is not None:
        metrics["eval_window"] = window
    return metrics


def _is_prop_safe(metrics: Optional[Dict[str, Any]]) -> bool:
    if not metrics:
        return False
    mdd_pct = abs(float(metrics.get("maks_drawdown_pct", 0.0)))
    return mdd_pct < MAX_DAILY_DD_PCT and mdd_pct < MAX_TOTAL_DD_PCT


def _score(is_metrics: Dict[str, Any], oos_metrics: Dict[str, Any]) -> float:
    oos_pnl = float(oos_metrics.get("net_pnl", 0.0))
    is_pnl = float(is_metrics.get("net_pnl", 0.0))
    oos_sharpe = float(oos_metrics.get("sharpe", 0.0))
    oos_dd = abs(float(oos_metrics.get("maks_drawdown_pct", 0.0)))
    is_trades = int(is_metrics.get("toplam_islem", 0))
    oos_trades = int(oos_metrics.get("toplam_islem", 0))

    is_trade_penalty = max(0, MIN_TRADES_IS - is_trades) * 2.0
    oos_trade_penalty = max(0, MIN_TRADES_OOS - oos_trades) * 10.0
    dd_penalty = oos_dd * 25.0

    return (
        oos_pnl
        + is_pnl * 0.20
        + oos_sharpe * 10.0
        - dd_penalty
        - is_trade_penalty
        - oos_trade_penalty
    )


def _passes_champion_gate(is_metrics: Dict[str, Any], oos_metrics: Dict[str, Any]) -> bool:
    return (
        _is_prop_safe(is_metrics)
        and _is_prop_safe(oos_metrics)
        and float(oos_metrics.get("net_pnl", 0.0)) > 0
        and int(oos_metrics.get("toplam_islem", 0)) >= MIN_TRADES_OOS
    )


def _save_results(records: List[Dict[str, Any]]) -> None:
    RESULTS_FILE.write_text(json.dumps(records, indent=2, ensure_ascii=False), encoding="utf-8")


def _metric(m: Optional[Dict[str, Any]], key: str, default: Any = 0) -> Any:
    return (m or {}).get(key, default)


def _print_leaderboard(records: List[Dict[str, Any]], champion: Optional[Dict[str, Any]]) -> None:
    lines: List[str] = []
    sep = "=" * 120
    lines.append(sep)
    lines.append("nasdaq_bot_v2 WALK-FORWARD OPTIMIZATION")
    lines.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Split: IS first ~{360 - OOS_DAYS} days | OOS last {OOS_DAYS} days")
    lines.append(sep)
    lines.append(
        f"{'#':>3} {'THR':>5} {'RISK':>5} {'RR':>4} "
        f"{'IS_PNL':>9} {'IS_TR':>5} {'IS_DD':>6} "
        f"{'OOS_PNL':>9} {'OOS_TR':>6} {'OOS_DD':>7} {'OOS_WR':>7} "
        f"{'SCORE':>9} {'STATUS':>10}"
    )
    lines.append("-" * 120)

    for i, r in enumerate(records, 1):
        p = r["params"]
        ism = r.get("is_metrics")
        oosm = r.get("oos_metrics")
        status = "PASS" if r.get("passed") else "FAIL"
        lines.append(
            f"{i:>3} {p.get('STRESS_XGB_THR'):>5} {p.get('STRESS_RISK_PCT'):>5} {p.get('STRESS_RR_RATIO'):>4} "
            f"{_metric(ism, 'net_pnl', 0):>+9.2f} {_metric(ism, 'toplam_islem', 0):>5} {abs(_metric(ism, 'maks_drawdown_pct', 0)):>6.2f} "
            f"{_metric(oosm, 'net_pnl', 0):>+9.2f} {_metric(oosm, 'toplam_islem', 0):>6} {abs(_metric(oosm, 'maks_drawdown_pct', 0)):>7.2f} "
            f"{_metric(oosm, 'win_rate_pct', 0):>7.1f} {r.get('score', float('-inf')):>+9.2f} {status:>10}"
        )

    lines.append(sep)
    if champion:
        cp = champion["params"]
        cm = champion["oos_metrics"]
        lines.append("CHAMPION")
        lines.append(f"  params={cp}")
        lines.append(
            f"  OOS net_pnl={cm.get('net_pnl', 0):+.2f} | trades={cm.get('toplam_islem', 0)} | "
            f"max_dd={abs(cm.get('maks_drawdown_pct', 0)):.2f}% | score={champion.get('score', 0):+.2f}"
        )
    else:
        lines.append("No champion passed the OOS gate.")

    output = "\n".join(lines)
    print(output)
    LEADERBOARD_FILE.write_text(output, encoding="utf-8")


def _replace_default(content: str, env_key: str, value: str) -> str:
    pattern = rf'(os\.getenv\("{re.escape(env_key)}",\s*")([^"]*)("\))'
    return re.sub(pattern, rf"\g<1>{value}\g<3>", content)


def _patch_settings(champion_params: Dict[str, str]) -> None:
    if not SETTINGS_FILE.exists():
        print("settings.py not found; champion was not persisted.")
        return

    content = SETTINGS_FILE.read_text(encoding="utf-8")
    for key, value in {**FIXED_ENV, **champion_params}.items():
        content = _replace_default(content, key, value)
    SETTINGS_FILE.write_text(content, encoding="utf-8")
    print(f"settings.py patched with champion params -> {SETTINGS_FILE}")


def run_optimization(combos: List[Dict[str, str]], dry_run: bool = False, skip_training: bool = False) -> None:
    if dry_run:
        for i, combo in enumerate(combos, 1):
            print(f"{i:>3}. {combo}")
        return

    print("=" * 80)
    print(f"Testing {len(combos)} combinations with IS training and OOS validation")
    print(f"OOS gate: net_pnl > 0, trades >= {MIN_TRADES_OOS}, MaxDD < {MAX_DAILY_DD_PCT}%")
    print("=" * 80)

    records: List[Dict[str, Any]] = []
    passed: List[Dict[str, Any]] = []
    t0 = time.time()

    for idx, combo in enumerate(combos, 1):
        eta = ((time.time() - t0) / max(1, idx - 1)) * (len(combos) - idx + 1) if idx > 1 else 0
        print(
            f"[{idx:>3}/{len(combos)}] THR={combo['STRESS_XGB_THR']} "
            f"RISK={combo['STRESS_RISK_PCT']} RR={combo['STRESS_RR_RATIO']} | ETA={eta/60:.1f}m"
        )

        train_ok = True if skip_training else _run_training(combo)
        if not train_ok:
            record = {
                "params": combo,
                "is_metrics": None,
                "oos_metrics": None,
                "passed": False,
                "score": float("-inf"),
                "error": "training_failed",
            }
            records.append(record)
            _save_results(records)
            continue

        is_metrics = _run_backtest(combo, "IS")
        oos_metrics = _run_backtest(combo, "OOS") if _is_prop_safe(is_metrics) else None

        passed_gate = bool(is_metrics and oos_metrics and _passes_champion_gate(is_metrics, oos_metrics))
        score = _score(is_metrics, oos_metrics) if is_metrics and oos_metrics else float("-inf")
        record = {
            "params": combo,
            "is_metrics": is_metrics,
            "oos_metrics": oos_metrics,
            "passed": passed_gate,
            "score": round(score, 4) if score != float("-inf") else score,
        }
        records.append(record)
        if passed_gate:
            passed.append(record)

        print(
            f"  IS pnl={_metric(is_metrics, 'net_pnl', 0):+.2f} tr={_metric(is_metrics, 'toplam_islem', 0)} "
            f"| OOS pnl={_metric(oos_metrics, 'net_pnl', 0):+.2f} tr={_metric(oos_metrics, 'toplam_islem', 0)} "
            f"| score={score:+.2f} | {'PASS' if passed_gate else 'FAIL'}"
        )
        _save_results(records)

    records_sorted = sorted(records, key=lambda r: r.get("score", float("-inf")), reverse=True)
    champion = max(passed, key=lambda r: r["score"]) if passed else None

    _print_leaderboard(records_sorted, champion)
    _save_results(records_sorted)

    if champion:
        _patch_settings(champion["params"])
    print(f"Results: {RESULTS_FILE}")
    print(f"Leaderboard: {LEADERBOARD_FILE}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Walk-forward optimizer")
    parser.add_argument("--random", type=int, default=0, metavar="N")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-training", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    combos = _random_combinations(args.random) if args.random > 0 else _all_combinations()
    run_optimization(combos, dry_run=args.dry_run, skip_training=args.skip_training)


if __name__ == "__main__":
    main()
