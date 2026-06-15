"""
execution/simulator.py
----------------------
Bar-by-Bar Emir Simülatörü (Backtest Simulator)

Her 15 dakikalık bar akarken açık pozisyonların durumunu kontrol eder:
  - SL / TP tetiklenme kontrolü (intra-bar çelişki → muhafazakâr kural: SL kazanır)
  - Weekend flatten sinyali geldiğinde kapanış emri verir
  - Drawdown limiti aşıldığında açık pozisyonu zorla kapatır

Entegrasyon:
    simulator = BacktestSimulator()
    simulator.update_and_check_positions(current_bar, portfolio, guardrails)
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from risk.portfolio import PortfolioManager
    from risk.guardrails import RiskGuardrails

try:
    from config.settings import (
        PARTIAL_CLOSE_R,
        PARTIAL_CLOSE_FRAC,
        TRAILING_ACTIVATION_R,
        TRAILING_ATR_MULT,
    )
except ImportError:
    PARTIAL_CLOSE_R       = 1.0
    PARTIAL_CLOSE_FRAC    = 0.25
    TRAILING_ACTIVATION_R = 1.0
    TRAILING_ATR_MULT     = 1.0

logger = logging.getLogger(__name__)


class BacktestSimulator:
    """
    Olay güdümlü (event-driven) bar-by-bar emir simülatörü.

    Kritik Kural — Intra-bar Çelişki Çözümü
    ----------------------------------------
    Bir bar içinde hem TP hem SL seviyesine temas varsa,
    fiyatın önce hangisine gittiğini bilemeyiz.
    Muhafazakâr (gerçekçi) yaklaşım: SL_HIT → zararla kapat.
    Bu, over-optimistic backtest sonuçlarını engeller.
    """

    # Slippage: SL'de %0.02 adverse (piyasa emri), TP'de %0.01 adverse (limit tam dolmayabilir)
    SL_SLIPPAGE_PCT = 0.0002
    TP_SLIPPAGE_PCT = 0.0001

    def __init__(self) -> None:
        self.bars_processed:   int = 0
        self.tp_hits:          int = 0
        self.sl_hits:          int = 0
        self.trail_hits:       int = 0   # trailing stop ile KÂRDA kapanan işlemler
        self.force_closes:     int = 0
        self.guardrail_closes: int = 0

        logger.info("BacktestSimulator başlatıldı.")

    # ------------------------------------------------------------------
    # Ana Güncelleme Fonksiyonu
    # ------------------------------------------------------------------

    def update_and_check_positions(
        self,
        current_bar:  pd.Series,
        portfolio:    "PortfolioManager",
        guardrails:   "RiskGuardrails",
    ) -> str | None:
        """
        Mevcut barı işler ve gerekirse açık pozisyonu kapatır.

        Parameters
        ----------
        current_bar : pd.Series
            O anki bar verisi. Zorunlu alanlar: High, Low, Close + DatetimeIndex.
        portfolio : PortfolioManager
            Aktif portföy nesnesi.
        guardrails : RiskGuardrails
            Prop firm kural seti.

        Returns
        -------
        str | None
            Kapanış gerçekleştiyse sebep kodu ("SL" | "TP" | "FORCE_CLOSE" |
            "GUARDRAIL" | "WEEKEND_FLATTEN"), pozisyon yoksa None.
        """
        self.bars_processed += 1

        # ── Erken çıkış: Açık pozisyon yoksa işlem gereksiz ──────────────
        if portfolio.open_position is None:
            return None

        pos       = portfolio.open_position
        timestamp = current_bar.name  # DatetimeIndex → datetime

        bar_high:  float = float(current_bar["High"])
        bar_low:   float = float(current_bar["Low"])
        bar_close: float = float(current_bar["Close"])

        # ── Equity mark-to-market güncelle ───────────────────────────────
        portfolio.update_equity_mark(bar_close)

        # ── 1. Weekend Flatten: Cuma zorla kapanış ────────────────────────
        if guardrails.should_force_close(timestamp):
            pnl, reason = portfolio.close_trade(
                bar_close, timestamp, reason="WEEKEND_FLATTEN"
            )
            self.force_closes += 1
            guardrails.record_trade_result(won=pnl > 0)
            logger.info(
                f"🗓️ WEEKEND_FLATTEN kapanışı @ {bar_close:.4f} | "
                f"PnL={pnl:+.2f} USD"
            )
            return "WEEKEND_FLATTEN"

        # ── 2. SL / TP Kontrolü (GUARDRAIL'DEN ÖNCE) ─────────────────────
        # Neden önce: Guardrail bar_close kullanır, SL ise gerçek SL fiyatını.
        # Gap/ani hareket durumunda SL bypass edilmeden doğru fiyattan çıkış sağlanır.
        if pos.is_long:
            sl_touched_early: bool = bar_low  <= pos.stop_loss
            tp_touched_early: bool = bar_high >= pos.take_profit
        else:
            sl_touched_early = bar_high >= pos.stop_loss
            tp_touched_early = bar_low  <= pos.take_profit

        if sl_touched_early and tp_touched_early:
            sl_fill = pos.stop_loss * (1 + self.SL_SLIPPAGE_PCT) if pos.is_long else pos.stop_loss * (1 - self.SL_SLIPPAGE_PCT)
            # Trailing SL kâra çekilmiş olabilir → kapanış sebebini PnL'e göre belirle
            is_trail_profit = pos.trailing_active and (sl_fill > pos.entry_price if pos.is_long else sl_fill < pos.entry_price)
            reason = "TRAIL" if is_trail_profit else "SL"
            pnl, _ = portfolio.close_trade(sl_fill, timestamp, reason=reason)
            won = pnl > 0
            if won:
                self.trail_hits += 1
            else:
                self.sl_hits += 1
            guardrails.record_trade_result(won=won)
            logger.debug(f"⚠️  INTRA-BAR ÇELİŞKİ → {reason} @ {sl_fill:.4f} | PnL={pnl:+.2f} USD")
            return reason

        if sl_touched_early:
            sl_fill = pos.stop_loss * (1 + self.SL_SLIPPAGE_PCT) if pos.is_long else pos.stop_loss * (1 - self.SL_SLIPPAGE_PCT)
            # Trailing SL kâra çekilmiş olabilir → kapanış sebebini PnL'e göre belirle
            is_trail_profit = pos.trailing_active and (sl_fill > pos.entry_price if pos.is_long else sl_fill < pos.entry_price)
            reason = "TRAIL" if is_trail_profit else "SL"
            pnl, _ = portfolio.close_trade(sl_fill, timestamp, reason=reason)
            won = pnl > 0
            if won:
                self.trail_hits += 1
            else:
                self.sl_hits += 1
            guardrails.record_trade_result(won=won)
            logger.debug(f"{'🟡 TRAIL' if won else '🔴 SL_HIT'} @ {sl_fill:.4f} | PnL={pnl:+.2f} USD | Bar={timestamp}")
            return reason

        if tp_touched_early:
            # TP limit emir: küçük adverse slippage gerçekçi dolum modeli
            tp_fill = pos.take_profit * (1 - self.TP_SLIPPAGE_PCT) if pos.is_long else pos.take_profit * (1 + self.TP_SLIPPAGE_PCT)
            pnl, _ = portfolio.close_trade(tp_fill, timestamp, reason="TP")
            self.tp_hits += 1
            guardrails.record_trade_result(won=True)
            logger.debug(f"🟢 TP_HIT @ {tp_fill:.4f} | PnL={pnl:+.2f} USD | Bar={timestamp}")
            return "TP"

        # ── 3. Drawdown Kill-Switch: SL/TP'den sonra kontrol ─────────────
        if guardrails.check_drawdown_limits(portfolio):
            pnl, reason = portfolio.close_trade(bar_close, timestamp, reason="GUARDRAIL")
            self.guardrail_closes += 1
            guardrails.record_trade_result(won=pnl > 0)
            logger.warning(
                f"🛡️ GUARDRAIL kapanışı | PnL={pnl:+.2f} USD | "
                f"Bar={timestamp}"
            )
            return "GUARDRAIL"

        # ── 5. Kısmi Kâr (Partial Close at 1R) ──────────────────────────────
        if not pos.partial_closed and pos.atr_at_open > 0:
            partial_dist = pos.sl_distance * PARTIAL_CLOSE_R
            if pos.is_long:
                partial_hit = bar_high >= pos.entry_price + partial_dist
            else:
                partial_hit = bar_low <= pos.entry_price - partial_dist

            if partial_hit:
                partial_price = (
                    pos.entry_price + partial_dist if pos.is_long
                    else pos.entry_price - partial_dist
                )
                portfolio.partial_close(partial_price, timestamp, fraction=PARTIAL_CLOSE_FRAC)
                # Breakeven SL: sadece mevcut SL entry'nin altındaysa çek
                # (trailing zaten SL'i entry üstüne taşımışsa geri düşürme)
                if pos.is_long and pos.stop_loss < pos.entry_price:
                    pos.stop_loss = pos.entry_price
                elif not pos.is_long and pos.stop_loss > pos.entry_price:
                    pos.stop_loss = pos.entry_price
                logger.debug(
                    "PARTIAL_CLOSE @ %.4f | SL=%.4f (breakeven=%.4f) | Bar=%s",
                    partial_price, pos.stop_loss, pos.entry_price, timestamp,
                )

        # ── 6. Trailing Stop Aktifleştirme + Güncelleme ───────────────────
        # TRAILING_ACTIVATION_R kadar kazanç varsa trailing başlat
        if not pos.trailing_active and pos.atr_at_open > 0:
            activation_dist = pos.sl_distance * TRAILING_ACTIVATION_R
            if pos.is_long:
                trailing_triggered = bar_high >= pos.entry_price + activation_dist
            else:
                trailing_triggered = bar_low <= pos.entry_price - activation_dist
            if trailing_triggered:
                pos.trailing_active = True

        if pos.trailing_active and pos.atr_at_open > 0:
            trail_dist = pos.atr_at_open * TRAILING_ATR_MULT
            if pos.is_long:
                new_sl = bar_high - trail_dist
                if new_sl > pos.stop_loss:
                    pos.stop_loss = new_sl
            else:
                new_sl = bar_low + trail_dist
                if new_sl < pos.stop_loss:
                    pos.stop_loss = new_sl

        # ── 7. Pozisyon Hâlâ Açık: Devam ─────────────────────────────────
        return None

    # ------------------------------------------------------------------
    # İstatistik Özeti
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Simülatör istatistiklerini döndürür."""
        total_closed = self.tp_hits + self.sl_hits + self.trail_hits + self.force_closes + self.guardrail_closes
        return {
            "bars_processed":   self.bars_processed,
            "total_closed":     total_closed,
            "tp_hits":          self.tp_hits,
            "sl_hits":          self.sl_hits,
            "trail_hits":       self.trail_hits,
            "weekend_flattens": self.force_closes,
            "guardrail_closes": self.guardrail_closes,
        }
