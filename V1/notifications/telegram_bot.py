"""
notifications/telegram_bot.py
------------------------------
Telegram bildirim modülü — İşlem açılış/kapanış/özet alertleri.
"""

from __future__ import annotations

import logging
import os
import urllib.request
import urllib.parse
import json
from datetime import datetime
from typing import Optional

logger = logging.getLogger(__name__)

TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN", "8697274237:AAGtTQb82SyUQdknTSenwJhqMm7pVSf6Fq4")
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "8627574593")

_BASE_URL = f"https://api.telegram.org/bot{TELEGRAM_TOKEN}/sendMessage"


def _send(text: str, parse_mode: str = "HTML") -> bool:
    try:
        data = urllib.parse.urlencode({
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text,
            "parse_mode": parse_mode,
        }).encode("utf-8")
        req = urllib.request.Request(_BASE_URL, data=data, method="POST")
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read())
            return result.get("ok", False)
    except Exception as exc:
        logger.warning("Telegram gönderim hatası: %s", exc)
        return False


def notify_trade_open(
    direction: str,
    entry_price: float,
    stop_loss: float,
    take_profit: float,
    lot_size: float,
    timestamp: datetime,
    balance: float,
) -> None:
    yön = "🟢 LONG" if "LONG" in direction else "🔴 SHORT"
    sl_pct = abs(entry_price - stop_loss) / entry_price * 100
    tp_pct = abs(take_profit - entry_price) / entry_price * 100
    msg = (
        f"<b>📈 YENİ POZİSYON AÇILDI</b>\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"Yön        : {yön}\n"
        f"Giriş      : <code>{entry_price:.4f}</code>\n"
        f"Stop Loss  : <code>{stop_loss:.4f}</code>  (-{sl_pct:.2f}%)\n"
        f"Take Profit: <code>{take_profit:.4f}</code>  (+{tp_pct:.2f}%)\n"
        f"Lot        : <code>{lot_size:.4f}</code>\n"
        f"Bakiye     : <code>${balance:,.2f}</code>\n"
        f"🕐 {timestamp.strftime('%Y-%m-%d %H:%M')} UTC"
    )
    _send(msg)


def notify_trade_close(
    direction: str,
    entry_price: float,
    exit_price: float,
    net_pnl: float,
    reason: str,
    balance: float,
    timestamp: datetime,
    trade_count: int,
    win_count: int,
) -> None:
    emoji = "✅" if net_pnl > 0 else "❌"
    reason_map = {
        "TP": "🎯 TP Hit",
        "SL": "🛑 SL Hit",
        "WEEKEND_FLATTEN": "📅 Hafta Sonu",
        "FORCE_CLOSE": "⏹ Zorla Kapanış",
        "GUARDRAIL": "🚨 Guardrail",
    }
    reason_str = reason_map.get(reason, reason)
    wr = win_count / trade_count * 100 if trade_count else 0
    pnl_str = f"+${net_pnl:.2f}" if net_pnl >= 0 else f"-${abs(net_pnl):.2f}"
    msg = (
        f"<b>{emoji} POZİSYON KAPANDI</b>\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"Sebep      : {reason_str}\n"
        f"Giriş      : <code>{entry_price:.4f}</code>\n"
        f"Çıkış      : <code>{exit_price:.4f}</code>\n"
        f"Net P&L    : <b>{pnl_str}</b>\n"
        f"Bakiye     : <code>${balance:,.2f}</code>\n"
        f"Win Rate   : <code>{win_count}/{trade_count}  ({wr:.1f}%)</code>\n"
        f"🕐 {timestamp.strftime('%Y-%m-%d %H:%M')} UTC"
    )
    _send(msg)


def notify_kill_switch(reason: str, balance: float, drawdown_pct: float) -> None:
    msg = (
        f"<b>🚨 KİLL SWİTCH AKTİF!</b>\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"Sebep    : {reason}\n"
        f"Drawdown : <code>{drawdown_pct:.2f}%</code>\n"
        f"Bakiye   : <code>${balance:,.2f}</code>\n"
        f"⚠️ Tüm işlemler durduruldu!"
    )
    _send(msg)


def notify_daily_summary(
    date_str: str,
    balance: float,
    daily_pnl: float,
    total_trades: int,
    win_rate: float,
    drawdown_pct: float,
) -> None:
    emoji = "📈" if daily_pnl >= 0 else "📉"
    pnl_str = f"+${daily_pnl:.2f}" if daily_pnl >= 0 else f"-${abs(daily_pnl):.2f}"
    msg = (
        f"<b>{emoji} GÜNLÜK ÖZET — {date_str}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"Günlük P&L  : <b>{pnl_str}</b>\n"
        f"Bakiye      : <code>${balance:,.2f}</code>\n"
        f"İşlem Sayısı: <code>{total_trades}</code>\n"
        f"Win Rate    : <code>{win_rate:.1f}%</code>\n"
        f"Max DD      : <code>{drawdown_pct:.2f}%</code>"
    )
    _send(msg)


def notify_session_start(balance: float, symbol: str, timeframe: str) -> None:
    msg = (
        f"<b>🤖 BOT BAŞLADI</b>\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"Sembol  : <code>{symbol}</code>\n"
        f"Periyot : <code>{timeframe}</code>\n"
        f"Bakiye  : <code>${balance:,.2f}</code>\n"
        f"🕐 {datetime.utcnow().strftime('%Y-%m-%d %H:%M')} UTC"
    )
    _send(msg)


def notify_session_end(
    balance: float,
    net_pnl: float,
    total_trades: int,
    win_rate: float,
    profit_factor: float,
    max_dd_pct: float,
    sharpe: float,
) -> None:
    emoji = "🏆" if net_pnl > 0 else "😔"
    pnl_str = f"+${net_pnl:.2f}" if net_pnl >= 0 else f"-${abs(net_pnl):.2f}"
    msg = (
        f"<b>{emoji} OTURUM BİTTİ</b>\n"
        f"━━━━━━━━━━━━━━━━━━━\n"
        f"Net P&L     : <b>{pnl_str}</b>\n"
        f"Son Bakiye  : <code>${balance:,.2f}</code>\n"
        f"İşlem       : <code>{total_trades}</code>\n"
        f"Win Rate    : <code>{win_rate:.1f}%</code>\n"
        f"Profit Fact : <code>{profit_factor:.3f}</code>\n"
        f"Max DD      : <code>{max_dd_pct:.2f}%</code>\n"
        f"Sharpe      : <code>{sharpe:.2f}</code>"
    )
    _send(msg)
