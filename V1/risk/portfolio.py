"""
risk/portfolio.py
-----------------
Portfolio accounting and ATR-based position sizing.

This version uses raw ATR for stop distance. There is no ATR floor that widens
the stop artificially. Lot size is calculated from the intended cash risk and is
only capped by a dynamic notional leverage limit plus an absolute broker cap.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Literal, Optional, Tuple

try:
    from notifications.telegram_bot import notify_trade_open, notify_trade_close
    _TG_ENABLED = True
except Exception:
    _TG_ENABLED = False

try:
    from config.settings import (
        ATR_MULTIPLIER,
        COMMISSION_RATE,
        CONTRACT_SIZE,
        COST_BENEFIT_MAX_RATIO,
        INITIAL_BALANCE,
        MAX_LOT_LIMIT,
        MAX_NOTIONAL_LEVERAGE,
        REWARD_RISK_RATIO,
        RISK_PER_TRADE,
        SPREAD_PENALTY,
        PARTIAL_CLOSE_R,
        TRAILING_ACTIVATION_R,
        TRAILING_ATR_MULT,
    )
except ImportError:
    ATR_MULTIPLIER = 1.5
    COMMISSION_RATE = 0.0002
    CONTRACT_SIZE = 1.0
    COST_BENEFIT_MAX_RATIO = 0.60
    INITIAL_BALANCE = 100_000.0
    MAX_LOT_LIMIT = 1000.0
    MAX_NOTIONAL_LEVERAGE = 20.0
    REWARD_RISK_RATIO = 2.5
    RISK_PER_TRADE = 0.009
    SPREAD_PENALTY = 0.05
    PARTIAL_CLOSE_R = 1.0
    TRAILING_ACTIVATION_R = 1.0
    TRAILING_ATR_MULT = 1.0

DirectionType = Literal["STRONG_LONG", "STRONG_SHORT"]
CloseReasonType = Literal["SL", "TP", "TRAIL", "FORCE_CLOSE", "GUARDRAIL", "WEEKEND_FLATTEN"]

logger = logging.getLogger(__name__)


@dataclass
class Position:
    direction: DirectionType
    entry_price: float
    stop_loss: float
    take_profit: float
    lot_size: float
    sl_distance: float
    commission: float
    open_time: datetime
    metadata: Dict = field(default_factory=dict)

    # Trailing stop ve kısmi kâr takibi
    trailing_active: bool = False
    partial_closed: bool = False        # 1R'da %50 kapatıldı mı
    original_lot: float = 0.0          # Açılıştaki tam lot (kısmi kapanış için)
    atr_at_open: float = 0.0           # Açılıştaki ATR değeri

    @property
    def is_long(self) -> bool:
        return self.direction == "STRONG_LONG"


class PortfolioManager:
    """Single-position portfolio accounting engine."""

    def __init__(self) -> None:
        self.balance: float = INITIAL_BALANCE
        self.equity: float = INITIAL_BALANCE
        self.open_position: Optional[Position] = None
        self.daily_peak_equity: float = INITIAL_BALANCE
        self.trade_log: list = []
        self._total_commission: float = 0.0

        logger.info(
            "PortfolioManager initialized | balance=%.2f | risk/trade=%.2f%% | "
            "ATRx=%.2f | RR=1:%.2f | max_lot=%.2f | max_notional_leverage=%.1fx",
            INITIAL_BALANCE,
            RISK_PER_TRADE * 100,
            ATR_MULTIPLIER,
            REWARD_RISK_RATIO,
            MAX_LOT_LIMIT,
            MAX_NOTIONAL_LEVERAGE,
        )

    def _dynamic_lot_cap(self, current_price: float) -> float:
        if current_price <= 0 or CONTRACT_SIZE <= 0:
            return MAX_LOT_LIMIT
        notional_cap_lot = (self.balance * MAX_NOTIONAL_LEVERAGE) / (
            current_price * CONTRACT_SIZE
        )
        return max(0.0, min(MAX_LOT_LIMIT, notional_cap_lot))

    def calculate_position_size(self, current_price: float, atr_value: float) -> float:
        if atr_value <= 0:
            logger.warning("Invalid ATR %.6f. Lot size=0.", atr_value)
            return 0.0

        sl_distance = atr_value * ATR_MULTIPLIER
        if sl_distance <= 0:
            logger.warning("Invalid SL distance %.6f. Lot size=0.", sl_distance)
            return 0.0

        risk_amount = self.balance * RISK_PER_TRADE
        raw_lot = risk_amount / (sl_distance * CONTRACT_SIZE)
        lot_cap = self._dynamic_lot_cap(current_price)
        lot_size = min(raw_lot, lot_cap)

        if lot_size < raw_lot:
            logger.warning(
                "Lot cap active | raw=%.4f -> final=%.4f | cap=%.4f | risk_target=%.2f",
                raw_lot,
                lot_size,
                lot_cap,
                risk_amount,
            )

        logger.debug(
            "Position size | price=%.5f | raw_atr=%.5f | sl_distance=%.5f | "
            "risk=%.2f | raw_lot=%.4f | final_lot=%.4f",
            current_price,
            atr_value,
            sl_distance,
            risk_amount,
            raw_lot,
            lot_size,
        )
        return lot_size

    def open_trade(
        self,
        direction: DirectionType,
        current_price: float,
        atr_value: float,
        timestamp: datetime,
    ) -> Optional[Position]:
        if self.open_position is not None:
            logger.warning("Open position exists. New trade skipped.")
            return None

        lot_size = self.calculate_position_size(current_price, atr_value)
        if lot_size <= 0:
            logger.warning("Invalid lot size. Trade skipped.")
            return None

        sl_distance = atr_value * ATR_MULTIPLIER
        half_spread = SPREAD_PENALTY / 2.0

        if direction == "STRONG_LONG":
            entry_price = current_price + half_spread
            stop_loss = entry_price - sl_distance
            take_profit = entry_price + (sl_distance * REWARD_RISK_RATIO)
        else:
            entry_price = current_price - half_spread
            stop_loss = entry_price + sl_distance
            take_profit = entry_price - (sl_distance * REWARD_RISK_RATIO)

        tp_distance = sl_distance * REWARD_RISK_RATIO
        gross_tp = tp_distance * lot_size * CONTRACT_SIZE
        entry_comm = entry_price * lot_size * CONTRACT_SIZE * COMMISSION_RATE
        exit_comm = take_profit * lot_size * CONTRACT_SIZE * COMMISSION_RATE
        spread_cost = SPREAD_PENALTY * lot_size * CONTRACT_SIZE
        total_cost = entry_comm + exit_comm + spread_cost
        cost_ratio = total_cost / gross_tp if gross_tp > 0 else float("inf")

        if cost_ratio > COST_BENEFIT_MAX_RATIO:
            logger.warning(
                "Cost-benefit veto | cost=%.2f | gross_tp=%.2f | ratio=%.1f%% > limit=%.1f%%",
                total_cost,
                gross_tp,
                cost_ratio * 100,
                COST_BENEFIT_MAX_RATIO * 100,
            )
            return None

        commission = entry_comm
        self.balance -= commission
        self._total_commission += commission
        self._update_equity(current_price)

        position = Position(
            direction=direction,
            entry_price=entry_price,
            stop_loss=stop_loss,
            take_profit=take_profit,
            lot_size=lot_size,
            sl_distance=sl_distance,
            commission=commission,
            open_time=timestamp,
            original_lot=lot_size,
            atr_at_open=atr_value,
        )
        self.open_position = position

        logger.info(
            "Position opened [%s] @ %.4f | lot=%.4f | SL=%.4f | TP=%.4f | "
            "risk_target=%.2f | cost_ratio=%.1f%%",
            direction,
            entry_price,
            lot_size,
            stop_loss,
            take_profit,
            self.balance * RISK_PER_TRADE,
            cost_ratio * 100,
        )
        if _TG_ENABLED:
            try:
                notify_trade_open(
                    direction=direction,
                    entry_price=entry_price,
                    stop_loss=stop_loss,
                    take_profit=take_profit,
                    lot_size=lot_size,
                    timestamp=timestamp,
                    balance=self.balance,
                )
            except Exception:
                pass
        return position

    def close_trade(
        self,
        exit_price: float,
        timestamp: datetime,
        reason: CloseReasonType = "TP",
    ) -> Tuple[float, CloseReasonType]:
        if self.open_position is None:
            logger.warning("No open position to close.")
            return (0.0, reason)

        pos = self.open_position
        if pos.is_long:
            gross_pnl = (exit_price - pos.entry_price) * pos.lot_size * CONTRACT_SIZE
        else:
            gross_pnl = (pos.entry_price - exit_price) * pos.lot_size * CONTRACT_SIZE

        exit_commission = exit_price * pos.lot_size * CONTRACT_SIZE * COMMISSION_RATE
        net_pnl = gross_pnl - exit_commission

        self.balance += net_pnl
        self._total_commission += exit_commission
        self._update_equity(exit_price)

        if self.equity > self.daily_peak_equity:
            self.daily_peak_equity = self.equity

        record = {
            "direction": pos.direction,
            "open_time": pos.open_time,
            "close_time": timestamp,
            "entry_price": pos.entry_price,
            "exit_price": exit_price,
            "lot_size": pos.lot_size,
            "gross_pnl": round(gross_pnl, 4),
            "commission": round(pos.commission + exit_commission, 4),
            "net_pnl": round(net_pnl, 4),
            "reason": reason,
            "balance": round(self.balance, 4),
        }
        self.trade_log.append(record)
        self.open_position = None

        logger.info(
            "Position closed [%s] @ %.4f | net_pnl=%+.2f | balance=%.2f | time=%s",
            reason,
            exit_price,
            net_pnl,
            self.balance,
            timestamp,
        )
        if _TG_ENABLED:
            try:
                total = len(self.trade_log)
                wins  = sum(1 for t in self.trade_log if t["net_pnl"] > 0)
                notify_trade_close(
                    direction=pos.direction,
                    entry_price=pos.entry_price,
                    exit_price=exit_price,
                    net_pnl=net_pnl,
                    reason=reason,
                    balance=self.balance,
                    timestamp=timestamp,
                    trade_count=total,
                    win_count=wins,
                )
            except Exception:
                pass
        return (net_pnl, reason)

    def partial_close(
        self,
        exit_price: float,
        timestamp: datetime,
        fraction: float = 0.5,
    ) -> float:
        """
        Pozisyonun fraction kadarını kapatır (varsayılan %50).
        Geri kalan lot ile pozisyon açık kalır, trailing devam eder.
        """
        if self.open_position is None:
            return 0.0

        pos = self.open_position
        close_lot = pos.lot_size * fraction
        remain_lot = pos.lot_size * (1.0 - fraction)

        if pos.is_long:
            gross_pnl = (exit_price - pos.entry_price) * close_lot * CONTRACT_SIZE
        else:
            gross_pnl = (pos.entry_price - exit_price) * close_lot * CONTRACT_SIZE

        exit_commission = exit_price * close_lot * CONTRACT_SIZE * COMMISSION_RATE
        net_pnl = gross_pnl - exit_commission

        self.balance += net_pnl
        self._total_commission += exit_commission

        # Geri kalan lot ile devam et
        pos.lot_size = remain_lot
        pos.partial_closed = True

        record = {
            "direction": pos.direction,
            "open_time": pos.open_time,
            "close_time": timestamp,
            "entry_price": pos.entry_price,
            "exit_price": exit_price,
            "lot_size": close_lot,
            "gross_pnl": round(gross_pnl, 4),
            "commission": round(exit_commission, 4),
            "net_pnl": round(net_pnl, 4),
            "reason": "PARTIAL_TP",
            "balance": round(self.balance, 4),
        }
        self.trade_log.append(record)

        logger.info(
            "Partial close %d%% @ %.4f | net_pnl=%+.2f | remaining_lot=%.4f",
            int(fraction * 100), exit_price, net_pnl, remain_lot,
        )
        return net_pnl

    def _update_equity(self, current_price: float) -> None:
        if self.open_position is None:
            self.equity = self.balance
            return

        pos = self.open_position
        if pos.is_long:
            unrealized = (current_price - pos.entry_price) * pos.lot_size * CONTRACT_SIZE
        else:
            unrealized = (pos.entry_price - current_price) * pos.lot_size * CONTRACT_SIZE
        self.equity = self.balance + unrealized

    def update_equity_mark(self, current_price: float) -> float:
        self._update_equity(current_price)
        if self.equity > self.daily_peak_equity:
            self.daily_peak_equity = self.equity
        return self.equity

    def reset_daily_peak(self) -> None:
        self.daily_peak_equity = self.equity
        logger.debug("Daily peak equity reset -> %.2f", self.equity)

    def summary(self) -> Dict:
        # Mantıksal işlem bazında grupla: aynı open_time'a ait PARTIAL_TP + final
        # kapanış TEK işlem sayılır. Aksi halde kısmi kapanışlar ayrı "kazanan"
        # gibi görünüp win-rate'i yapay şişirir (örn. 380 işlem/%66 yerine 234/%50).
        from collections import defaultdict
        grouped: Dict = defaultdict(float)
        for t in self.trade_log:
            key = t.get("open_time", id(t))
            grouped[key] += t["net_pnl"]

        total_trades = len(grouped)
        winning = sum(1 for pnl in grouped.values() if pnl > 0)
        win_rate = (winning / total_trades * 100) if total_trades else 0.0
        total_pnl = self.balance - INITIAL_BALANCE

        return {
            "balance": round(self.balance, 2),
            "equity": round(self.equity, 2),
            "total_pnl": round(total_pnl, 2),
            "total_pnl_pct": round(total_pnl / INITIAL_BALANCE * 100, 2),
            "total_trades": total_trades,
            "winning_trades": winning,
            "win_rate_pct": round(win_rate, 2),
            "total_commission": round(self._total_commission, 2),
            "open_position": self.open_position is not None,
        }
