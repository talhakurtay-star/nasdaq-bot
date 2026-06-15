"""
risk/guardrails.py
------------------
Prop Firm Zırhı ve Risk Filtreleri (Risk Guardrails)

Funding Pips / FTMO kurallarını ihlal etmemek için iki kritik savunma hattı:

  1. Drawdown Kill-Switch
     - Günlük drawdown > MAX_DAILY_DRAWDOWN_PCT  → kill-switch aktif
     - Toplam drawdown > MAX_TOTAL_DRAWDOWN_PCT  → kill-switch aktif
     Kill-switch aktifken hiçbir yeni işlem açılamaz.

  2. Zaman ve Gün Filtreleri
     - TRADE_START_HOUR – TRADE_END_HOUR penceresi dışında işlem yasak.
     - ALLOW_WEEKEND_HOLDING = False ise Cuma ZORLU_KAPANIS_SAATI'nde
       açık pozisyon zorla kapatılır (hafta sonu flatten).

Entegrasyon akışı
-----------------
    guardrails = RiskGuardrails()

    # Her bar döngüsünde:
    if guardrails.check_drawdown_limits(portfolio):
        break  # Günü/challenge'ı sonlandır

    if not guardrails.check_time_constraints(timestamp):
        continue  # Bu barda yeni işlem alma

    if guardrails.should_force_close(timestamp) and portfolio.open_position:
        portfolio.close_trade(current_price, timestamp, reason="WEEKEND_FLATTEN")
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from risk.portfolio import PortfolioManager

try:
    from config.settings import (
        ALLOW_WEEKEND_HOLDING,
        BLOCK_FRIDAY_ENTRIES,
        INITIAL_BALANCE,
        MAX_DAILY_DRAWDOWN_PCT,
        MAX_TOTAL_DRAWDOWN_PCT,
        THURSDAY_CUTOFF_HOUR,
        TRADE_END_HOUR,
        TRADE_START_HOUR,
        ZORLU_KAPANIS_SAATI,
    )
except ImportError:
    ALLOW_WEEKEND_HOLDING   = False
    BLOCK_FRIDAY_ENTRIES    = True
    INITIAL_BALANCE         = 100_000.0
    MAX_DAILY_DRAWDOWN_PCT  = 0.04
    MAX_TOTAL_DRAWDOWN_PCT  = 0.09
    THURSDAY_CUTOFF_HOUR    = 22
    TRADE_END_HOUR          = 23
    TRADE_START_HOUR        = 16
    ZORLU_KAPANIS_SAATI     = 23

logger = logging.getLogger(__name__)

# Cuma günü: Python'da weekday() → 0=Pazartesi … 4=Cuma
_FRIDAY   = 4
_THURSDAY = 3  # Perşembe

# ── Yüksek Etkili Haber Saatleri (UTC) ───────────────────────────────────────
# Funding Pips Master hesap kuralı: kırmızı haber ±5 dk yeni işlem yasak.
# Backtest/sinyal modunda bu saatler etrafında giriş engellenir.
# Kaynak: Forexfactory kırmızı haberler → genellikle aşağıdaki saatlerde gelir.
# HABER_FILTER_ENABLED env ile açılıp kapatılabilir.
_NEWS_FILTER_ENABLED = True  # Master hesapta zorunlu
_NEWS_BLOCK_MINUTES  = 5     # haber öncesi/sonrası kaç dakika engel

# Haftalık sabit yüksek etkili haber saatleri (UTC, (weekday, hour, minute, block_minutes))
# 0=Pazartesi … 4=Cuma  |  block_minutes: haber öncesi/sonrası kaç dakika blok
_HIGH_IMPACT_NEWS: list[tuple[int, int, int, int]] = [
    # Pazartesi
    (0, 14,  0,  5),   # ABD ISM İmalat PMI
    # Salı
    (1, 13, 30,  5),   # ABD JOLTS / Ticaret Dengesi
    # Çarşamba
    (2, 13, 15,  5),   # ADP İstihdam
    (2, 13, 30,  5),   # ABD Çekirdek TÜFE / PPI
    (2, 19,  0, 90),   # FOMC — 90 dk önce/sonra blok (mega volatilite)
    # Perşembe
    (3, 13, 30,  5),   # Haftalık İşsizlik Başvuruları
    (3, 15,  0,  5),   # ABD ISM Hizmetler PMI
    # Cuma
    (4, 13, 30, 60),   # NFP — 60 dk önce/sonra blok (en kritik haber)
    (4, 15,  0,  5),   # Michigan Tüketici Güveni
]


class RiskGuardrails:
    """
    Prop firm kurallarını zorunlu kılan askeri risk filtresi.

    Attributes
    ----------
    kill_switch_active : bool
        True olduğunda hiçbir yeni işlem açılamaz; ancak açık
        pozisyonlar hâlâ kapatılabilir (guardrail kapatır).
    daily_drawdown_triggered : bool
        O gün için günlük limit aşıldıysa True.
    total_drawdown_triggered : bool
        Başlangıçtan bu yana toplam limit aşıldıysa True.
    """

    def __init__(self) -> None:
        self.kill_switch_active:       bool = False
        self.daily_drawdown_triggered: bool = False
        self.total_drawdown_triggered: bool = False
        self.daily_trades_count:       int  = 0
        self.max_daily_trades:         int  = 99  # Günlük limit yok — circuit breaker yönetir
        self.consecutive_sl_count:     int  = 0
        self.consecutive_win_count:    int  = 0   # art arda kazananlar
        self.circuit_breaker_active:   bool = False
        self.max_consecutive_sl:       int  = 3   # 3 art arda SL → o gün dur

        logger.info(
            "RiskGuardrails başlatıldı | "
            f"Günlük Limit=%{MAX_DAILY_DRAWDOWN_PCT*100:.1f} | "
            f"Toplam Limit=%{MAX_TOTAL_DRAWDOWN_PCT*100:.1f} | "
            f"İşlem Saatleri={TRADE_START_HOUR:02d}:00–{TRADE_END_HOUR:02d}:59 UTC | "
            f"Hafta Sonu Holding={'İzinli' if ALLOW_WEEKEND_HOLDING else 'YASAK'}"
        )

    # ------------------------------------------------------------------
    # 1. Drawdown Kill-Switch
    # ------------------------------------------------------------------

    def check_drawdown_limits(self, portfolio: "PortfolioManager") -> bool:
        """
        Anlık equity değerini hem günlük hem toplam drawdown limitleriyle karşılaştırır.

        Parameters
        ----------
        portfolio : PortfolioManager
            Anlık equity ve bakiye bilgisine sahip portföy nesnesi.

        Returns
        -------
        bool
            True  → Kill-switch tetiklendi; yeni işlem yasak, açık pozisyonlar kapatılmalı.
            False → Her şey normal, trade devam edebilir.
        """
        # Daha önce zaten tetiklendiyse tekrar hesaplama yapmadan döndür
        if self.kill_switch_active:
            return True

        current_equity:    float = portfolio.equity
        daily_peak_equity: float = portfolio.daily_peak_equity

        # ── Günlük Drawdown Kontrolü ──────────────────────────────────────
        # Gün içindeki en yüksek equity'den düşüş oranı
        if daily_peak_equity > 0:
            daily_dd: float = (daily_peak_equity - current_equity) / daily_peak_equity
        else:
            daily_dd = 0.0

        if daily_dd >= MAX_DAILY_DRAWDOWN_PCT:
            self.daily_drawdown_triggered = True
            self.kill_switch_active       = True
            logger.critical(
                f"🚨 GÜNLÜK DRAWDOWN LİMİTİ AŞILDI! "
                f"Tepe Equity={daily_peak_equity:,.2f} | "
                f"Anlık Equity={current_equity:,.2f} | "
                f"Düşüş=%{daily_dd*100:.2f} >= %{MAX_DAILY_DRAWDOWN_PCT*100:.1f} | "
                "Kill-Switch AKTİF → Tüm işlemler durduruldu."
            )
            return True

        # ── Toplam Drawdown Kontrolü ──────────────────────────────────────
        # Başlangıç bakiyesinden düşüş oranı
        total_dd: float = (INITIAL_BALANCE - current_equity) / INITIAL_BALANCE

        if total_dd >= MAX_TOTAL_DRAWDOWN_PCT:
            self.total_drawdown_triggered = True
            self.kill_switch_active       = True
            logger.critical(
                f"🚨 TOPLAM DRAWDOWN LİMİTİ AŞILDI! "
                f"Başlangıç={INITIAL_BALANCE:,.2f} | "
                f"Anlık Equity={current_equity:,.2f} | "
                f"Düşüş=%{total_dd*100:.2f} >= %{MAX_TOTAL_DRAWDOWN_PCT*100:.1f} | "
                "Kill-Switch AKTİF → Challenge/Hesap kaybedildi."
            )
            return True

        # ── Normal Durum: Sadece debug log ───────────────────────────────
        logger.debug(
            f"Drawdown Durumu | "
            f"Günlük=%{daily_dd*100:.2f} (limit: %{MAX_DAILY_DRAWDOWN_PCT*100:.1f}) | "
            f"Toplam=%{total_dd*100:.2f} (limit: %{MAX_TOTAL_DRAWDOWN_PCT*100:.1f})"
        )
        return False

    def reset_daily_drawdown(self, portfolio: "PortfolioManager") -> None:
        """
        Her gün başında çağrılır. Günlük drawdown sayacını ve portföy
        tepe equity değerini sıfırlar. Toplam drawdown kill-switch'i
        sıfırlamaz (hesap ömrü boyunca geçerlidir).
        """
        if self.total_drawdown_triggered:
            logger.warning(
                "Toplam drawdown limiti aşılmış; günlük sıfırlama devre dışı."
            )
            return

        self.daily_drawdown_triggered = False
        # Kill-switch sadece günlük limit yüzünden aktifleştiyse sıfırla
        if not self.total_drawdown_triggered:
            self.kill_switch_active = False

        portfolio.reset_daily_peak()
        self.daily_trades_count     = 0
        self.consecutive_sl_count   = 0
        self.consecutive_win_count  = 0
        self.circuit_breaker_active = False
        logger.info("Günlük drawdown sayacı sıfırlandı.")

    # ------------------------------------------------------------------
    # 2. Zaman Penceresi Filtresi
    # ------------------------------------------------------------------

    def check_time_constraints(self, timestamp: datetime) -> bool:
        """
        Verilen zaman damgasının geçerli işlem penceresinde olup olmadığını kontrol eder.

        Kural: TRADE_START_HOUR <= saat < TRADE_END_HOUR + 1
               (Saat tam sayı karşılaştırması; 23:59 hâlâ 23. saate dahildir.)

        Parameters
        ----------
        timestamp : datetime
            Değerlendirilecek bar'ın zaman damgası (UTC varsayımı).

        Returns
        -------
        bool
            True  → İşlem penceresi içinde, yeni pozisyon alınabilir.
            False → Pencere dışı, yeni işlem yasak.
        """
        current_hour: int = timestamp.hour

        in_window: bool = TRADE_START_HOUR <= current_hour <= TRADE_END_HOUR

        if not in_window:
            logger.debug(
                f"⏰ Saat {current_hour:02d}:xx işlem penceresi dışında "
                f"({TRADE_START_HOUR:02d}:00–{TRADE_END_HOUR:02d}:59). "
                "Yeni işlem açılmıyor."
            )

        return in_window

    # ------------------------------------------------------------------
    # 3. Hafta Sonu Flatten (Weekend Kill)
    # ------------------------------------------------------------------

    def should_force_close(self, timestamp: datetime) -> bool:
        """
        Açık pozisyonun hafta sonu taşınmasını engelleyen zorla kapanış sinyali.

        Kural (ALLOW_WEEKEND_HOLDING = False):
          Gün = Cuma (weekday=4) VE saat >= ZORLU_KAPANIS_SAATI
          → True döndür (Portföy bu fiyattan kapatılmalı)

        Parameters
        ----------
        timestamp : datetime
            O anki barın zaman damgası.

        Returns
        -------
        bool
            True  → Pozisyon zorla kapatılmalı (weekend flatten tetiklendi).
            False → Normal devam.
        """
        # Hafta sonu pozisyon taşımaya izin veriliyorsa hiçbir şey yapma
        if ALLOW_WEEKEND_HOLDING:
            return False

        is_friday:         bool = timestamp.weekday() == _FRIDAY
        past_close_hour:   bool = timestamp.hour >= ZORLU_KAPANIS_SAATI

        if is_friday and past_close_hour:
            logger.warning(
                f"🗓️ HAFTA SONU FLATTEN — Cuma {timestamp.strftime('%H:%M')} UTC | "
                f"Zorla kapanış saati: {ZORLU_KAPANIS_SAATI:02d}:00. "
                "Açık pozisyon kapatılıyor..."
            )
            return True

        return False

    # ------------------------------------------------------------------
    # 4. Tek Nokta Kontrol Fonksiyonu (Convenience)
    # ------------------------------------------------------------------

    def is_trading_allowed(
        self,
        portfolio:  "PortfolioManager",
        timestamp:  datetime,
    ) -> bool:
        """
        Tüm filtreleri tek satırda kontrol eder.
        Backtest döngüsünde `if not guardrails.is_trading_allowed(...)` şeklinde kullanılır.

        Returns
        -------
        bool
            True  → Tüm filtreler geçti; yeni işlem açılabilir.
            False → En az bir filtre engelledi; yeni işlem yasak.
        """
        if self.check_drawdown_limits(portfolio):
            return False

        if not self.check_time_constraints(timestamp):
            return False

        # ── Circuit Breaker: 3 art arda SL → o gün trading dur ───────────
        if self.circuit_breaker_active:
            logger.debug("⚡ Circuit breaker aktif — %d art arda SL. Bugün işlem yok.",
                         self.consecutive_sl_count)
            return False

        # ── Perşembe/Cuma Yeni İşlem Engeli (Weekend Flatten Tuzak Önleyici) ──
        weekday = timestamp.weekday()

        # Günlük işlem limiti kontrolü
        if self.daily_trades_count >= self.max_daily_trades:
            logger.debug("Günlük max işlem sayısına ulaşıldı (%d). Yeni işlem açılmıyor.", self.max_daily_trades)
            return False

        # ── Haber Filtresi: Yüksek Etkili Haber ±5 dk Engeli ────────────────
        if _NEWS_FILTER_ENABLED and self._is_near_news(timestamp):
            return False

        if BLOCK_FRIDAY_ENTRIES and weekday == _FRIDAY:
            logger.debug(
                f"⏰ CUMA GÜNÜ — Yeni işlem açılmıyor (BLOCK_FRIDAY_ENTRIES=True). "
                f"Bar zamanı: {timestamp.strftime('%A %H:%M')}"
            )
            return False

        if weekday == _THURSDAY and timestamp.hour >= THURSDAY_CUTOFF_HOUR:
            logger.debug(
                f"⏰ PERŞEMBE {timestamp.hour:02d}:xx ≥ {THURSDAY_CUTOFF_HOUR:02d}:00 — "
                f"Yeni işlem açılmıyor (THURSDAY_CUTOFF_HOUR). "
                f"Bar zamanı: {timestamp.strftime('%A %H:%M')}"
            )
            return False

        return True

    # ------------------------------------------------------------------
    # 5. Durum Raporu
    # ------------------------------------------------------------------

    def _is_near_news(self, timestamp: datetime) -> bool:
        """
        Verilen bar, yüksek etkili bir haber saatine yakınsa True döndürür.
        Her haberin kendi blok süresi var (FOMC=90dk, NFP=60dk, diğerleri=5dk).
        """
        try:
            ts = pd.Timestamp(timestamp)
            bar_weekday = ts.weekday()
            bar_minutes = ts.hour * 60 + ts.minute

            for (news_day, news_hour, news_min, block_min) in _HIGH_IMPACT_NEWS:
                if bar_weekday != news_day:
                    continue
                news_total = news_hour * 60 + news_min
                if abs(bar_minutes - news_total) <= block_min:
                    logger.debug(
                        "📰 Haber filtresi: %s bölge içinde (%02d:%02d ±%d dk). İşlem engellendi.",
                        ts.strftime("%A %H:%M"), news_hour, news_min, block_min,
                    )
                    return True
        except Exception:
            pass
        return False

    def record_trade_result(self, won: bool) -> None:
        """Her işlem kapanışında çağrılır. Circuit breaker ve streak sayaçlarını günceller."""
        if won:
            self.consecutive_sl_count = 0
            self.consecutive_win_count += 1
        else:
            self.consecutive_sl_count += 1
            self.consecutive_win_count = 0
            if self.consecutive_sl_count >= self.max_consecutive_sl:
                self.circuit_breaker_active = True
                logger.warning(
                    "⚡ CIRCUIT BREAKER AKTİF — %d art arda SL. Bugün yeni işlem yok.",
                    self.consecutive_sl_count,
                )
                try:
                    from notifications.telegram_bot import notify_kill_switch
                    notify_kill_switch(
                        reason=f"{self.consecutive_sl_count} art arda SL — Circuit Breaker",
                        balance=0.0,
                        drawdown_pct=0.0,
                    )
                except Exception:
                    pass

    def get_streak_risk_mult(self) -> float:
        """
        Streak'e göre dinamik risk çarpanı.
        Kayıp streak: cascade drawdown önlemi.
        Kazanç streak: momentum ile risk artışı (max 1.2x, günlük DD limiti geçmez).
        """
        if self.consecutive_sl_count >= 2:
            return 0.5    # 2+ SL → yarı risk
        if self.consecutive_sl_count == 1:
            return 0.75   # 1 SL → %75 risk
        if self.consecutive_win_count >= 3:
            return 1.2    # 3+ kazanç → %120 risk (momentum)
        if self.consecutive_win_count == 2:
            return 1.1    # 2 kazanç → %110 risk
        return 1.0

    def get_daily_dd_risk_mult(self, portfolio: "PortfolioManager") -> float:
        """
        Günlük drawdown %3'ü aştığında risk yarıya iner.
        Prop firm %4 limitine buffer bırakır.
        """
        if portfolio.daily_peak_equity <= 0:
            return 1.0
        daily_dd = (portfolio.daily_peak_equity - portfolio.equity) / portfolio.daily_peak_equity
        if daily_dd >= 0.03:
            logger.debug("Günlük DD =%{:.2f} → risk 0.5x".format(daily_dd * 100))
            return 0.5
        return 1.0

    def status(self) -> dict:
        """Guardrail'lerin anlık durumunu dict olarak döndürür."""
        return {
            "kill_switch_active":       self.kill_switch_active,
            "daily_drawdown_triggered": self.daily_drawdown_triggered,
            "total_drawdown_triggered": self.total_drawdown_triggered,
            "circuit_breaker_active":   self.circuit_breaker_active,
            "consecutive_sl_count":     self.consecutive_sl_count,
            "weekend_holding_allowed":  ALLOW_WEEKEND_HOLDING,
            "trade_window":             f"{TRADE_START_HOUR:02d}:00–{TRADE_END_HOUR:02d}:59 UTC",
            "force_close_hour_friday":  ZORLU_KAPANIS_SAATI,
        }
