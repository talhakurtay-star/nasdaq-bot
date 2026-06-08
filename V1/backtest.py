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
)

# ── Katman İmportları ─────────────────────────────────────────────────────────
from data.data_feed          import DataFeed
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

def run_backtest() -> None:
    logger = _setup_logging()
    start_time = datetime.now()
    logger.info("=" * 62)
    logger.info("  nasdaq_bot_v2 BACKTEST BAŞLADI")
    logger.info("=" * 62)

    # ── 1. Veriyi İndir ve İndikatörleri Hesapla ─────────────────────────
    logger.info("📥 Veri indiriliyor ve indikatörler hesaplanıyor...")
    data_feed      = DataFeed()
    feature_engine = FeatureEngine()

    raw_df = data_feed.download_historical_data()
    df     = feature_engine.calculate_indicators(raw_df)

    total_bars = len(df)
    warmup     = _get_warmup_bars()
    sim_start, sim_end, eval_window = _get_simulation_bounds(df, warmup, logger)
    if sim_end <= sim_start:
        raise ValueError(
            f"Simulasyon penceresi bos: window={eval_window}, start={sim_start}, end={sim_end}"
        )
    logger.info(
        f"Toplam bar: {total_bars} | Isınma periyodu: {warmup} bar | "
        f"Simüle edilecek bar: {sim_end - sim_start} | Pencere: {eval_window}"
    )

    # ── 2. Tüm Bileşenleri Örnekle ────────────────────────────────────────
    strategy  = StrategyEngine()
    voter     = VotingMechanism()
    portfolio = PortfolioManager()
    guardrails = RiskGuardrails()
    simulator = BacktestSimulator()

    # ── 3. Equity Curve ve Günlük İzleme Değişkenleri ────────────────────
    equity_curve: List[float] = []   # İlk bar sonrası doldurulur (off-by-one fix)
    current_day:  date | None = None

    # ── 4. Ana Bar-by-Bar Simülasyon Döngüsü ─────────────────────────────
    logger.info("🚀 Simülasyon döngüsü başlıyor...")

    for current_index in range(sim_start, sim_end):
        current_bar = df.iloc[current_index]
        timestamp   = df.index[current_index]

        # ── 4a. Günlük Drawdown Sıfırlama ─────────────────────────────────
        bar_date = pd.Timestamp(timestamp).date()
        if current_day is None:
            current_day = bar_date
        elif bar_date > current_day:
            # Yeni gün başladı — günlük sayaçları sıfırla
            guardrails.reset_daily_drawdown(portfolio)
            current_day = bar_date
            logger.info(f"📅 Yeni gün: {bar_date} | Bakiye: ${portfolio.balance:,.2f}")

        # ── 4b. Açık Pozisyon Güncelle (SL/TP/Flatten kontrolü) ───────────
        simulator.update_and_check_positions(current_bar, portfolio, guardrails)

        # ── 4c. Kill-Switch Aktifse döngüyü bitir ─────────────────────────
        if guardrails.kill_switch_active:
            logger.critical(
                f"🚨 Kill-Switch aktif. Simülasyon bar {current_index}'de durduruldu."
            )
            break

        # ── 4d. Yeni İşlem Açılabilir mi? ─────────────────────────────────
        if portfolio.open_position is not None:
            # Mevcut pozisyon henüz açık, yeni pozisyon açma
            equity_curve.append(portfolio.equity)
            continue

        if not guardrails.is_trading_allowed(portfolio, timestamp):
            equity_curve.append(portfolio.equity)
            continue

        # ── 4e. Teknik Sinyal Üret ─────────────────────────────────────────
        base_signal = strategy.generate_base_signal(df, current_index)

        # ── 4f. XGBoost Veto / Onay ────────────────────────────────────────
        final_signal = voter.decide_trade(
            df, current_index, base_signal, feature_engine
        )

        # ── 4g. Pozisyon Aç ────────────────────────────────────────────────
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
            else:
                logger.debug(f"Bar {current_index}: ATR=0, pozisyon atlandı.")

        # ── 4h. Equity Kaydı ───────────────────────────────────────────────
        equity_curve.append(portfolio.equity)

    # ── 5. Döngü Sonunda Açık Kalan Pozisyonu Kapat ───────────────────────
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


