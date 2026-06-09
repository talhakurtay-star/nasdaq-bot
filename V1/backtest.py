"""
backtest.py
-----------
Ana Çalıştırıcı ve Raporlama Motoru — nasdaq_bot_v2

Tüm katmanları (Data → Features → Strategy → Risk → Execution) birleştirir
ve bar-by-bar simülasyonu kronolojik olarak akıtır.

Çalıştırmak için:
    python backtest.py

Çıktılar:
    - Konsol: Quant performans metrikleri (Sharpe, MaxDD, Win Rate vb.)
    - Log dosyası: logs/backtest_YYYYMMDD_HHMMSS.log
"""

from __future__ import annotations

import logging
import math
import os
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
from datetime import datetime, date
from typing import List

import numpy as np
import pandas as pd

# ── Proje kök dizini Python path'ine ekleniyor ──────────────────────────────
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

# ── Config ───────────────────────────────────────────────────────────────────
from config.settings import (
    ATR_PERIOD,
    EMA_FAST,
    EMA_SLOW,
    INITIAL_BALANCE,
    MAX_DAILY_DRAWDOWN_PCT,
    MAX_TOTAL_DRAWDOWN_PCT,
    RSI_PERIOD,
    REWARD_RISK_RATIO,
    RISK_PER_TRADE,
    MAX_OPEN_POSITIONS,
    MAX_SECTOR_POSITIONS,
    SECTOR_MAP,
)

# ── Katman İmportları ─────────────────────────────────────────────────────────
from data.data_feed          import DataFeed, load_all_symbols
from ml.features             import FeatureEngine
from strategy.engine         import StrategyEngine
from strategy.voting         import VotingMechanism
from risk.portfolio          import PortfolioManager
from risk.guardrails         import RiskGuardrails
from execution.simulator     import BacktestSimulator


# ── Loglama Kurulumu ──────────────────────────────────────────────────────────
def _setup_logging() -> logging.Logger:
    log_dir = os.path.join(ROOT_DIR, "logs")
    os.makedirs(log_dir, exist_ok=True)
    timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = os.path.join(log_dir, f"backtest_{timestamp_str}.log")

    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    logging.basicConfig(
        level=logging.INFO,
        format=fmt,
        handlers=[
            logging.FileHandler(log_path, encoding="utf-8"),
            logging.StreamHandler(sys.stdout),
        ],
    )
    logger = logging.getLogger("backtest")
    logger.info(f"Log dosyası: {log_path}")
    return logger


# ── Isınma Periyodu Hesabı ────────────────────────────────────────────────────
def _get_warmup_bars() -> int:
    # EMA_200 + MACD_slow(26) + bant için yeterli ısınma
    return max(EMA_SLOW, RSI_PERIOD, ATR_PERIOD, 200, 26) + 10


def _get_simulation_bounds(
    df: pd.DataFrame,
    warmup: int,
    logger: logging.Logger,
) -> tuple[int, int, str]:
    """
    Return [start, end) indexes for FULL/IS/OOS evaluation.

    STRESS_EVAL_WINDOW:
        FULL -> all bars after warm-up
        IS   -> bars up to the last STRESS_OOS_DAYS days
        OOS  -> only the last STRESS_OOS_DAYS days
    """
    window = os.environ.get("STRESS_EVAL_WINDOW", "FULL").upper()
    if window == "FULL":
        return warmup, len(df), "FULL"

    oos_days = int(os.environ.get("STRESS_OOS_DAYS", "90"))
    cutoff = df.index.max() - pd.Timedelta(days=oos_days)

    if window == "IS":
        positions = np.flatnonzero(df.index <= cutoff)
        if len(positions) == 0:
            raise ValueError("IS window is empty; reduce STRESS_OOS_DAYS.")
        start_idx = warmup
        end_idx = int(positions[-1]) + 1
    elif window == "OOS":
        positions = np.flatnonzero(df.index > cutoff)
        if len(positions) == 0:
            raise ValueError("OOS window is empty; reduce STRESS_OOS_DAYS.")
        start_idx = max(warmup, int(positions[0]))
        end_idx = len(df)
    else:
        logger.warning("Unknown STRESS_EVAL_WINDOW=%s; using FULL.", window)
        return warmup, len(df), "FULL"

    logger.info(
        "Evaluation window=%s | cutoff=%s | start=%d | end=%d | bars=%d",
        window,
        cutoff,
        start_idx,
        end_idx,
        max(0, end_idx - start_idx),
    )
    return start_idx, end_idx, window


# ── Quant Performans Metrikleri ───────────────────────────────────────────────

def _calculate_sharpe_ratio(
    equity_curve: List[float], risk_free_rate: float = 0.0, bars_per_year: int = 17_472
) -> float:
    """
    Yıllıklandırılmış Sharpe oranı (15m bar başına getiri üzerinden).
    bars_per_year: 252 gün × 6.5 saat × 4 bar/saat ≈ 6552 (NYSE)
                   Prop firmalar için geniş tut: 252×69=17472
    """
    if len(equity_curve) < 2:
        return 0.0
    returns = np.diff(equity_curve) / np.array(equity_curve[:-1])
    excess  = returns - (risk_free_rate / bars_per_year)
    std     = np.std(excess, ddof=1)
    if std == 0:
        return 0.0
    return float(np.mean(excess) / std * math.sqrt(bars_per_year))


def _calculate_max_drawdown(equity_curve: List[float]) -> tuple[float, float]:
    """
    Maksimum Drawdown (MDD) — mutlak ve yüzde olarak.

    Returns
    -------
    (mdd_usd, mdd_pct)
    """
    arr  = np.array(equity_curve)
    peak = np.maximum.accumulate(arr)
    dd   = peak - arr
    mdd_usd = float(np.max(dd))
    mdd_pct = float(np.max(dd / np.where(peak == 0, 1, peak)) * 100)
    return mdd_usd, mdd_pct


def _calculate_profit_factor(trade_log: list) -> float:
    """Gross kazanç / Gross kayıp oranı."""
    gross_profit = sum(t["net_pnl"] for t in trade_log if t["net_pnl"] > 0)
    gross_loss   = abs(sum(t["net_pnl"] for t in trade_log if t["net_pnl"] < 0))
    if gross_loss == 0:
        return float("inf") if gross_profit > 0 else 0.0
    return round(gross_profit / gross_loss, 3)


def _calculate_avg_rr(trade_log: list) -> float:
    """Kazanan işlemlerin ortalama R değeri (net_pnl / risk_amount)."""
    if not trade_log:
        return 0.0
    winners = [t for t in trade_log if t["net_pnl"] > 0]
    if not winners:
        return 0.0
    # risk_amount ≈ initial_balance * RISK_PER_TRADE (sabit ref değeri)
    risk_ref = INITIAL_BALANCE * RISK_PER_TRADE
    avg_r = np.mean([t["net_pnl"] / risk_ref for t in winners])
    return round(float(avg_r), 3)


# ── Raporlama ─────────────────────────────────────────────────────────────────

def _print_report(
    portfolio:  PortfolioManager,
    simulator:  BacktestSimulator,
    equity_curve: List[float],
    start_time: datetime,
    logger: logging.Logger,
) -> None:
    """Tüm quant metriklerini biçimli şekilde konsola ve log'a yazar."""
    end_time   = datetime.now()
    elapsed    = (end_time - start_time).total_seconds()
    summary    = portfolio.summary()
    trade_log  = portfolio.trade_log
    sim_stats  = simulator.stats()

    total_trades  = summary["total_trades"]
    winning       = summary["winning_trades"]
    losing        = total_trades - winning
    win_rate      = summary["win_rate_pct"]

    sharpe         = _calculate_sharpe_ratio(equity_curve)
    mdd_usd, mdd_pct = _calculate_max_drawdown(equity_curve)
    profit_factor  = _calculate_profit_factor(trade_log)
    avg_rr         = _calculate_avg_rr(trade_log)
    total_return   = summary["total_pnl_pct"]
    net_pnl        = summary["total_pnl"]

    sep = "═" * 62

    report = f"""
{sep}
  nasdaq_bot_v2  |  BACKTEST SONUÇLARI
{sep}
  📅  Simülasyon Süresi   : {elapsed:.1f} saniye
  📊  İşlenen Bar Sayısı  : {sim_stats['bars_processed']:,}

  ── PORTFÖY ──────────────────────────────────────────────
  💰  Başlangıç Bakiyesi  : ${INITIAL_BALANCE:>12,.2f}
  💵  Bitiş Bakiyesi      : ${summary['balance']:>12,.2f}
  📈  Net PnL             : ${net_pnl:>+12,.2f}  ({total_return:+.2f}%)
  📉  Maks. Drawdown      : ${mdd_usd:>12,.2f}  ({mdd_pct:.2f}%)
  💸  Toplam Komisyon     : ${summary['total_commission']:>12,.2f}

  ── TRADE İSTATİSTİKLERİ ─────────────────────────────────
  🔢  Toplam İşlem        : {total_trades}
  ✅  Kazanan             : {winning}
  ❌  Kaybeden            : {losing}
  🎯  Win Rate            : {win_rate:.1f}%
  ⚖️   Profit Factor       : {profit_factor:.3f}
  📐  Ort. Kazanan R      : {avg_rr:.2f}R

  ── KAPANIM SEBEPLERİ ────────────────────────────────────
  🟢  TP Hit              : {sim_stats['tp_hits']}
  🔴  SL Hit              : {sim_stats['sl_hits']}
  🗓️   Weekend Flatten     : {sim_stats['weekend_flattens']}
  🛡️   Guardrail Kapanışı  : {sim_stats['guardrail_closes']}

  ── QUANT METRİKLER ──────────────────────────────────────
  📊  Sharpe Oranı        : {sharpe:.4f}
  📏  R:R Hedefi          : 1:{REWARD_RISK_RATIO}
  ⚠️   Günlük DD Limiti   : %{MAX_DAILY_DRAWDOWN_PCT*100:.1f}
  ⚠️   Toplam DD Limiti   : %{MAX_TOTAL_DRAWDOWN_PCT*100:.1f}
{sep}"""

    logger.info(report)

    # Prop firm geçebilme ön değerlendirmesi
    passed = (
        mdd_pct < MAX_TOTAL_DRAWDOWN_PCT * 100
        and total_return > 0
        and sharpe > 0
    )
    verdict = "✅ PROP FIRM EŞİKLERİ GEÇİLEBİLİR GÖRÜNÜYOR" if passed else \
              "❌ PROP FIRM EŞİKLERİ AŞILDI — Strateji revize edilmeli"
    logger.info(f"\n  {verdict}\n{sep}\n")


import json
import math
import os
from pathlib import Path
 
 
def _stres_testi_metrikleri_yaz(
    portfolio_manager,         # PortfolioManager örneği
    trades: list,              # Kapatılan işlem listesi (dict)
    initial_balance: float,    # Başlangıç bakiyesi
) -> None:
    """
    Backtest sonuçlarını temp_metrics.json olarak kök dizine yazar.
    stres_testi_odasi.py bu dosyayı okuyarak metrikleri toplar.
 
    Parametreler
    ─────────────
    portfolio_manager : PortfolioManager örneği (bakiye, equity_curve için)
    trades            : {'pnl': float, ...} formatındaki kapatılmış işlemler
    initial_balance   : Başlangıç bakiyesi (genellikle config.INITIAL_BALANCE)
    """
 
    # ── Ham PnL listesi ────────────────────────────────────────────────────
    pnl_listesi: list[float] = []
    for t in trades:
        pnl = (
            t.get("pnl") or t.get("profit") or
            t.get("net_pnl") or t.get("realized_pnl") or 0.0
        )
        pnl_listesi.append(float(pnl))
 
    toplam_islem = len(pnl_listesi)
    if toplam_islem == 0:
        metrikler = {
            "net_pnl": 0.0, "net_pnl_pct": 0.0,
            "maks_drawdown_pct": 0.0, "win_rate_pct": 0.0,
            "sharpe": 0.0, "toplam_islem": 0,
            "kazanan": 0, "kaybeden": 0,
        }
    else:
        kazanan     = sum(1 for p in pnl_listesi if p > 0)
        kaybeden    = toplam_islem - kazanan
        net_pnl     = sum(pnl_listesi)
        win_rate    = kazanan / toplam_islem * 100.0
        net_pnl_pct = net_pnl / initial_balance * 100.0 if initial_balance else 0.0
 
        # Sharpe (işlem başına PnL serisi)
        if len(pnl_listesi) >= 2:
            n   = len(pnl_listesi)
            ort = sum(pnl_listesi) / n
            std = math.sqrt(sum((x - ort) ** 2 for x in pnl_listesi) / (n - 1))
            sharpe = ((ort / std) * math.sqrt(252.0)) if std > 1e-10 else 0.0
        else:
            sharpe = 0.0
 
        # Max Drawdown — equity_curve'den (varsa)
        equity_curve = getattr(portfolio_manager, "equity_curve", [])
        mdd = 0.0
        if len(equity_curve) >= 2:
            tepe = equity_curve[0]
            for deger in equity_curve[1:]:
                if deger > tepe:
                    tepe = deger
                if tepe > 1e-10:
                    dd = (deger - tepe) / tepe * 100.0
                    if dd < mdd:
                        mdd = dd
 
        weekend_flattens = sum(
            1 for t in trades
            if t.get("reason") == "WEEKEND_FLATTEN"
        )

        metrikler = {
            "net_pnl":            round(net_pnl, 4),
            "net_pnl_pct":        round(net_pnl_pct, 4),
            "maks_drawdown_pct":  round(mdd, 4),
            "win_rate_pct":       round(win_rate, 4),
            "sharpe":             round(sharpe, 4),
            "toplam_islem":       toplam_islem,
            "kazanan":            kazanan,
            "kaybeden":           kaybeden,
            "weekend_flattens":   weekend_flattens,
        }
 
    # ── Dosyaya yaz ────────────────────────────────────────────────────────
    kök = Path(__file__).resolve().parent
    yol = kök / "temp_metrics.json"
    try:
        yol.write_text(
            json.dumps(metrikler, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        print(f"[STRES-KANCA] temp_metrics.json yazıldı → {yol}")
    except OSError as e:
        print(f"[STRES-KANCA] UYARI: temp_metrics.json yazılamadı: {e}")

# ── Ana Backtest Fonksiyonu ───────────────────────────────────────────────────

def _run_single_symbol(
    sym: str,
    df: pd.DataFrame,
    feature_engine: FeatureEngine,
    strategy: StrategyEngine,
    voter: VotingMechanism,
    portfolio: PortfolioManager,
    guardrails: RiskGuardrails,
    simulator: BacktestSimulator,
    open_positions: dict,
    equity_curve: List[float],
    logger: logging.Logger,
) -> None:
    """Tek sembol için simülasyon döngüsünü çalıştırır (multi-symbol içinden çağrılır)."""
    warmup = _get_warmup_bars()
    sim_start, sim_end, _ = _get_simulation_bounds(df, warmup, logger)

    for current_index in range(sim_start, sim_end):
        current_bar = df.iloc[current_index]
        timestamp   = df.index[current_index]

        # Açık pozisyon varsa SL/TP kontrolü
        if sym in open_positions:
            portfolio.open_position = open_positions[sym]
            result = simulator.update_and_check_positions(current_bar, portfolio, guardrails)
            if result is not None:
                del open_positions[sym]
                portfolio.open_position = None
            else:
                open_positions[sym] = portfolio.open_position
                portfolio.open_position = None

        if guardrails.kill_switch_active:
            break

        # Max pozisyon kontrolü
        if len(open_positions) >= MAX_OPEN_POSITIONS:
            continue

        if sym in open_positions:
            continue

        if not guardrails.is_trading_allowed(portfolio, timestamp):
            continue

        base_signal  = strategy.generate_base_signal(df, current_index)
        final_signal = voter.decide_trade(df, current_index, base_signal, feature_engine)

        if final_signal in ("STRONG_LONG", "STRONG_SHORT"):
            close_price = float(current_bar["Close"])
            atr_value   = float(current_bar.get("ATR", 0))
            if atr_value > 0:
                portfolio.open_trade(
                    direction=final_signal,
                    current_price=close_price,
                    atr_value=atr_value,
                    timestamp=timestamp,
                )
                if portfolio.open_position is not None:
                    open_positions[sym] = portfolio.open_position
                    portfolio.open_position = None
                    guardrails.daily_trades_count += 1


def run_backtest() -> None:
    logger = _setup_logging()
    start_time = datetime.now()
    logger.info("=" * 62)
    logger.info("  nasdaq_bot_v2 BACKTEST BAŞLADI")
    logger.info("=" * 62)

    feature_engine = FeatureEngine()
    strategy   = StrategyEngine()
    voter      = VotingMechanism()
    portfolio  = PortfolioManager()
    guardrails = RiskGuardrails()
    simulator  = BacktestSimulator()

    # ── Multi-symbol mi, tek sembol mü? ──────────────────────────────────
    symbol_data = load_all_symbols()

    if symbol_data:
        logger.info("📊 Multi-symbol backtest: %d sembol", len(symbol_data))

        # Her sembol için indikatörleri hesapla
        dfs: dict[str, pd.DataFrame] = {}
        for sym, raw_df in symbol_data.items():
            try:
                dfs[sym] = feature_engine.calculate_indicators(raw_df)
            except Exception as e:
                logger.warning("%s indikatör hesaplamada hata: %s", sym, e)

        # Tüm zaman damgalarını birleştir (union), kronolojik sırala
        all_timestamps = sorted(set().union(*[set(df.index) for df in dfs.values()]))
        logger.info("Toplam benzersiz timestamp: %d", len(all_timestamps))

        equity_curve: List[float] = []
        open_positions: dict[str, object] = {}
        current_day: date | None = None

        logger.info("Korelasyon koruması: max %d pozisyon/sektör | max %d toplam",
                    MAX_SECTOR_POSITIONS, MAX_OPEN_POSITIONS)
        logger.info("Multi-symbol simülasyon başlıyor...")
        for ts in all_timestamps:
            bar_date = pd.Timestamp(ts).date()

            # Günlük sıfırlama
            if current_day is None:
                current_day = bar_date
            elif bar_date > current_day:
                guardrails.reset_daily_drawdown(portfolio)
                portfolio.reset_daily_peak()
                current_day = bar_date
                logger.info("Yeni gun: %s | Bakiye: $%.2f | Acik pos: %d",
                            bar_date, portfolio.balance, len(open_positions))

            if guardrails.kill_switch_active:
                break

            # ── 1. GEÇIŞ: Açık pozisyonları güncelle (SL/TP kontrolü) ────
            for sym in list(open_positions.keys()):
                if sym not in dfs or ts not in dfs[sym].index:
                    continue
                df = dfs[sym]
                current_index = df.index.get_loc(ts)
                portfolio.open_position = open_positions[sym]
                result = simulator.update_and_check_positions(
                    df.iloc[current_index], portfolio, guardrails
                )
                if result is not None:
                    guardrails.record_trade_result(won=(result == "TP"))
                    del open_positions[sym]
                    portfolio.open_position = None
                else:
                    open_positions[sym] = portfolio.open_position
                    portfolio.open_position = None

            if guardrails.kill_switch_active:
                break

            # ── 2. GEÇIŞ: Sinyal toplama + Probability sıralaması ─────────
            if (len(open_positions) < MAX_OPEN_POSITIONS
                    and guardrails.is_trading_allowed(portfolio, pd.Timestamp(ts))):

                signal_queue: list[tuple[str, str, float]] = []  # (sym, signal, prob)

                for sym, df in dfs.items():
                    if sym in open_positions:
                        continue
                    if ts not in df.index:
                        continue
                    current_index = df.index.get_loc(ts)
                    base_sig = strategy.generate_base_signal(df, current_index)
                    if base_sig == "HOLD":
                        continue
                    final_sig, prob = voter.get_signal_with_prob(
                        df, current_index, base_sig, feature_engine
                    )
                    if final_sig != "HOLD":
                        signal_queue.append((sym, final_sig, prob))

                # Probability'ye göre sırala (yüksekten düşüğe)
                signal_queue.sort(key=lambda x: x[2], reverse=True)

                # Sektör sayacını hesapla (mevcut açık pozisyonlardan)
                sector_counts: dict[str, int] = {}
                for open_sym in open_positions:
                    sec = SECTOR_MAP.get(open_sym, "OTHER")
                    sector_counts[sec] = sector_counts.get(sec, 0) + 1

                # En yüksek skorlu adayları sırayla değerlendir
                for sym, final_sig, prob in signal_queue:
                    if len(open_positions) >= MAX_OPEN_POSITIONS:
                        break

                    sector = SECTOR_MAP.get(sym, "OTHER")
                    if sector_counts.get(sector, 0) >= MAX_SECTOR_POSITIONS:
                        logger.debug("Sektör limiti: %s (%s) reddedildi | mevcut=%d",
                                     sym, sector, sector_counts.get(sector, 0))
                        continue

                    df = dfs[sym]
                    current_index = df.index.get_loc(ts)
                    bar = df.iloc[current_index]
                    atr = float(bar.get("ATR", 0))
                    if atr <= 0:
                        continue

                    portfolio.open_trade(
                        direction=final_sig,
                        current_price=float(bar["Close"]),
                        atr_value=atr,
                        timestamp=pd.Timestamp(ts),
                    )
                    if portfolio.open_position is not None:
                        open_positions[sym] = portfolio.open_position
                        portfolio.open_position = None
                        guardrails.daily_trades_count += 1
                        sector_counts[sector] = sector_counts.get(sector, 0) + 1
                        logger.info("Giris: %s [%s] prob=%.3f | sektor=%s",
                                    sym, final_sig, prob, sector)

            equity_curve.append(portfolio.equity)

        # Açık kalan pozisyonları kapat
        for sym, pos in list(open_positions.items()):
            df = dfs[sym]
            last_close = float(df.iloc[-1]["Close"])
            last_ts    = df.index[-1]
            portfolio.open_position = pos
            portfolio.close_trade(last_close, last_ts, reason="FORCE_CLOSE")
            portfolio.open_position = None

    else:
        # ── Tek sembol modu (eski davranış) ──────────────────────────────
        logger.info("📥 Tek sembol modu — veri yükleniyor...")
        raw_df = DataFeed().download_historical_data()
        df     = feature_engine.calculate_indicators(raw_df)

        total_bars = len(df)
        warmup     = _get_warmup_bars()
        sim_start, sim_end, eval_window = _get_simulation_bounds(df, warmup, logger)
        if sim_end <= sim_start:
            raise ValueError(f"Simülasyon penceresi boş: start={sim_start}, end={sim_end}")
        logger.info(
            "Toplam bar: %d | Isınma: %d | Simüle: %d | Pencere: %s",
            total_bars, warmup, sim_end - sim_start, eval_window,
        )

        equity_curve: List[float] = []
        current_day:  date | None = None
        logger.info("🚀 Simülasyon döngüsü başlıyor...")

        for current_index in range(sim_start, sim_end):
            current_bar = df.iloc[current_index]
            timestamp   = df.index[current_index]

            bar_date = pd.Timestamp(timestamp).date()
            if current_day is None:
                current_day = bar_date
            elif bar_date > current_day:
                guardrails.reset_daily_drawdown(portfolio)
                current_day = bar_date
                logger.info("Yeni gun: %s | Bakiye: $%.2f", bar_date, portfolio.balance)

            simulator.update_and_check_positions(current_bar, portfolio, guardrails)

            if guardrails.kill_switch_active:
                logger.critical("🚨 Kill-Switch aktif. Bar %d'de durduruldu.", current_index)
                break

            if portfolio.open_position is not None:
                equity_curve.append(portfolio.equity)
                continue

            if not guardrails.is_trading_allowed(portfolio, timestamp):
                equity_curve.append(portfolio.equity)
                continue

            base_signal  = strategy.generate_base_signal(df, current_index)
            final_signal = voter.decide_trade(df, current_index, base_signal, feature_engine)

            if final_signal in ("STRONG_LONG", "STRONG_SHORT"):
                close_price = float(current_bar["Close"])
                atr_value   = float(current_bar.get("ATR", 0))
                if atr_value > 0:
                    portfolio.open_trade(
                        direction=final_signal,
                        current_price=close_price,
                        atr_value=atr_value,
                        timestamp=timestamp,
                    )
                    guardrails.daily_trades_count += 1

            equity_curve.append(portfolio.equity)

        if portfolio.open_position is not None:
            last_close = float(df.iloc[sim_end - 1]["Close"])
            last_ts    = df.index[sim_end - 1]
            portfolio.close_trade(last_close, last_ts, reason="FORCE_CLOSE")
            logger.info("🔚 Açık pozisyon simülasyon sonunda zorla kapatıldı.")

    # ── 6. Sonuç Raporu ────────────────────────────────────────────────────
    _print_report(portfolio, simulator, equity_curve, start_time, logger)
    
    portfolio.equity_curve = equity_curve
    
    # Timestamps'leri eşleştirip portfolio nesnesine ekle
    timestamps = [str(ts) for ts in df.index[max(0, sim_start - 1):sim_end]]
    if len(equity_curve) == len(timestamps):
        portfolio.equity_timestamps = timestamps
    else:
        portfolio.equity_timestamps = [str(i) for i in range(len(equity_curve))]

    _stres_testi_metrikleri_yaz(portfolio, portfolio.trade_log, INITIAL_BALANCE)

    # HTML Raporu Oluştur
    try:
        import results_generator
        scenario_name = os.environ.get("STRESS_SCENARIO_NAME")
        results_generator.generate_html_report(portfolio, INITIAL_BALANCE, scenario_name)
    except Exception as e:
        logger.error(f"HTML Rapor oluşturulurken hata oluştu: {e}")

    # Telegram: Oturum Sonu Bildirimi
    try:
        from notifications.telegram_bot import notify_session_end
        from backtest import _calculate_sharpe_ratio, _calculate_max_drawdown, _calculate_profit_factor
        summary = portfolio.summary()
        _, mdd_pct = _calculate_max_drawdown(equity_curve)
        pf  = _calculate_profit_factor(portfolio.trade_log)
        sh  = _calculate_sharpe_ratio(equity_curve)
        notify_session_end(
            balance=portfolio.balance,
            net_pnl=summary["total_pnl"],
            total_trades=summary["total_trades"],
            win_rate=summary["win_rate_pct"],
            profit_factor=pf,
            max_dd_pct=mdd_pct,
            sharpe=sh,
        )
    except Exception:
        pass



# ── Giriş Noktası ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    run_backtest()


