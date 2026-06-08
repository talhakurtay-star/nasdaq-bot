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
import random
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from risk.portfolio import PortfolioManager
    from risk.guardrails import RiskGuardrails

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

    # Slippage: SL'de %0.02, TP'de 0 (TP limit emir, SL piyasa emri)
    SL_SLIPPAGE_PCT = 0.0002

    def __init__(self) -> None:
        self.bars_processed:   int = 0
        self.tp_hits:          int = 0
        self.sl_hits:          int = 0
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

        # ── 1. Drawdown Kill-Switch: Guardrail aktifse zorla kapat ────────
        if guardrails.check_drawdown_limits(portfolio):
            pnl, reason = portfolio.close_trade(bar_close, timestamp, reason="GUARDRAIL")
            self.guardrail_closes += 1
            logger.warning(
                f"🛡️ GUARDRAIL kapanışı | PnL={pnl:+.2f} USD | "
                f"Bar={timestamp}"
            )
            return "GUARDRAIL"

        # ── 2. Weekend Flatten: Cuma zorla kapanış ────────────────────────
        if guardrails.should_force_close(timestamp):
            pnl, reason = portfolio.close_trade(
                bar_close, timestamp, reason="WEEKEND_FLATTEN"
            )
            self.force_closes += 1
            logger.info(
                f"🗓️ WEEKEND_FLATTEN kapanışı @ {bar_close:.4f} | "
                f"PnL={pnl:+.2f} USD"
            )
            return "WEEKEND_FLATTEN"

        # ── 3. Intra-bar SL / TP Kontrolü ────────────────────────────────
        if pos.is_long:
            sl_touched: bool = bar_low  <= pos.stop_loss
            tp_touched: bool = bar_high >= pos.take_profit
        else:  # SHORT
            sl_touched = bar_high >= pos.stop_loss
            tp_touched = bar_low  <= pos.take_profit

        # Intra-bar çelişki: hem SL hem TP aynı bar → muhafazakâr kural: SL kazanır
        if sl_touched and tp_touched:
            sl_fill = pos.stop_loss * (1 + self.SL_SLIPPAGE_PCT) if pos.is_long else pos.stop_loss * (1 - self.SL_SLIPPAGE_PCT)
            pnl, _ = portfolio.close_trade(sl_fill, timestamp, reason="SL")
            self.sl_hits += 1
            guardrails.record_trade_result(won=False)
            logger.debug(
                f"⚠️  INTRA-BAR ÇELİŞKİ → SL_HIT @ {sl_fill:.4f} (slip) | "
                f"PnL={pnl:+.2f} USD | Bar={timestamp}"
            )
            return "SL"

        if sl_touched:
            sl_fill = pos.stop_loss * (1 + self.SL_SLIPPAGE_PCT) if pos.is_long else pos.stop_loss * (1 - self.SL_SLIPPAGE_PCT)
            pnl, _ = portfolio.close_trade(sl_fill, timestamp, reason="SL")
            self.sl_hits += 1
            guardrails.record_trade_result(won=False)
            logger.debug(
                f"🔴 SL_HIT @ {sl_fill:.4f} (slip={self.SL_SLIPPAGE_PCT*100:.2f}%) | "
                f"PnL={pnl:+.2f} USD | Bar={timestamp}"
            )
            return "SL"

        if tp_touched:
            pnl, _ = portfolio.close_trade(pos.take_profit, timestamp, reason="TP")
            self.tp_hits += 1
            guardrails.record_trade_result(won=True)
            logger.debug(
                f"🟢 TP_HIT @ {pos.take_profit:.4f} | PnL={pnl:+.2f} USD | Bar={timestamp}"
            )
            return "TP"

        # ── 4. Pozisyon Hâlâ Açık: Devam ─────────────────────────────────
        return None

    # ------------------------------------------------------------------
    # İstatistik Özeti
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        """Simülatör istatistiklerini döndürür."""
        total_closed = self.tp_hits + self.sl_hits + self.force_closes + self.guardrail_closes
        return {
            "bars_processed":   self.bars_processed,
            "total_closed":     total_closed,
            "tp_hits":          self.tp_hits,
            "sl_hits":          self.sl_hits,
            "weekend_flattens": self.force_closes,
            "guardrail_closes": self.guardrail_closes,
        }
