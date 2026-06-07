"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  execution/broker_bridge.py                                                  ║
║  MetaTrader 5 Bağlantı Köprüsü — NASDAQ BOT V2                              ║
║  Senior Trading Infrastructure Engineer · Live Execution Layer               ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  Sorumluluk Alanları:                                                        ║
║    · MT5 terminal bağlantı yönetimi (bağlan / kontrol et / kapat)           ║
║    · Canlı hesap ve pozisyon durumu sorgulama                                ║
║    · ATR tabanlı SL/TP ile piyasa emri iletimi                               ║
║    · Prop Firm kural uyumluluğu (embedded SL/TP, slippage kontrolü)         ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any

# MetaTrader5 kütüphanesi — pip install MetaTrader5
try:
    import MetaTrader5 as mt5
except ImportError as _mt5_err:
    raise ImportError(
        "MetaTrader5 kütüphanesi bulunamadı.\n"
        "Kurulum: pip install MetaTrader5\n"
        "Not: Yalnızca Windows ortamında çalışır."
    ) from _mt5_err

# ──────────────────────────────────────────────────────────────────────────────
# LOGLAMA
# ──────────────────────────────────────────────────────────────────────────────

logger = logging.getLogger("BrokerBridge")


# ──────────────────────────────────────────────────────────────────────────────
# DÖNÜŞ NESNELERİ
# ──────────────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class OrderResult:
    """
    Emir iletim sonucunu taşıyan değer nesnesi.
    Başarılı/başarısız fark etmeksizin her send_market_order çağrısından döner.
    """
    success:      bool
    ticket:       int           # MT5 işlem ticket numarası (0 = başarısız)
    order_type:   str           # "BUY" | "SELL"
    symbol:       str
    lot:          float
    price:        float         # Gerçekleşen fiyat
    sl:           float
    tp:           float
    retcode:      int           # mt5.TRADE_RETCODE_*
    comment:      str           # Hata mesajı veya "OK"
    timestamp:    datetime = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", datetime.utcnow())

    def __str__(self) -> str:
        durum = "✅ OK" if self.success else "❌ HATA"
        return (
            f"{durum} | {self.order_type} {self.symbol} "
            f"lot={self.lot:.2f} fiyat={self.price:.5f} "
            f"SL={self.sl:.5f} TP={self.tp:.5f} "
            f"ticket={self.ticket} retcode={self.retcode} | {self.comment}"
        )


@dataclass(frozen=True)
class AccountMetrics:
    """Anlık MT5 hesap durumu snapshot'ı."""
    balance:      float
    equity:       float
    margin:       float
    free_margin:  float
    margin_level: float   # Yüzde — 0 ise margin_level hesaplanamadı
    profit:       float
    currency:     str
    timestamp:    datetime = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "timestamp", datetime.utcnow())

    @property
    def drawdown_pct(self) -> float:
        """Anlık equity-bazlı drawdown yüzdesi (negatif değer)."""
        if self.balance < 1e-10:
            return 0.0
        return (self.equity - self.balance) / self.balance * 100.0


# ──────────────────────────────────────────────────────────────────────────────
# SABİTLER
# ──────────────────────────────────────────────────────────────────────────────

# MT5 sihirli numarası — bot emirlerini manuel emirlerden ayırt etmek için
_BOT_MAGIC_NUMBER: int    = 20240001

# Emir gönderiminde maksimum yeniden deneme sayısı
_MAX_RETRY:        int    = 3

# Yeniden denemeler arası bekleme (saniye)
_RETRY_DELAY:      float  = 1.5

# İzin verilen maksimum slippage (puan)
_MAX_SLIPPAGE:     int    = 30

# Bağlantı kontrol aralığı (saniye)
_HEARTBEAT_INTERVAL: int  = 60


# ──────────────────────────────────────────────────────────────────────────────
# MT5 BROKER BRIDGE SINIFI
# ──────────────────────────────────────────────────────────────────────────────

class MetaTraderBridge:
    """
    MetaTrader 5 terminal ile Python stratejisi arasındaki tek iletişim noktası.

    Tek Sorumluluk İlkesi:
    ─────────────────────
    Bu sınıf YALNIZCA broker ile konuşur.
    Risk hesaplaması, sinyal üretimi veya pozisyon yönetimi bu sınıfın
    sorumluluğu değildir — bu kararlar main.py tarafından verilir.

    Bağlantı Yaşam Döngüsü:
    ───────────────────────
        bridge = MetaTraderBridge(symbol="NAS100")
        try:
            ...
        finally:
            bridge.shutdown()
    """

    def __init__(
        self,
        symbol:        str   = "NAS100",
        account:       int   = 0,      # 0 → mevcut aktif hesap
        password:      str   = "",
        server:        str   = "",
        mt5_path:      str   = "",     # Boş → otomatik algıla
        magic_number:  int   = _BOT_MAGIC_NUMBER,
        max_slippage:  int   = _MAX_SLIPPAGE,
    ) -> None:
        self.symbol        = symbol
        self.account       = account
        self.password      = password
        self.server        = server
        self.mt5_path      = mt5_path
        self.magic_number  = magic_number
        self.max_slippage  = max_slippage
        self._connected    = False
        self._last_hb_ts   = 0.0

        self._initialize()

    # ── Bağlantı Yönetimi ─────────────────────────────────────────────────

    def _initialize(self) -> None:
        """MT5 terminalini başlatır ve kimlik doğrulaması yapar."""
        logger.info("🔌 MT5 terminal başlatılıyor…")

        # Terminal başlatma
        init_kwargs: dict[str, Any] = {}
        if self.mt5_path:
            init_kwargs["path"] = self.mt5_path

        if not mt5.initialize(**init_kwargs):
            hata = mt5.last_error()
            raise ConnectionError(
                f"MT5 terminal başlatılamadı: {hata}\n"
                "Olası nedenler:\n"
                "  · MT5 kurulu değil veya yol hatalı.\n"
                "  · Terminal kapalı veya başka bir süreç kilitli.\n"
                "  · Hesap bilgileri hatalı."
            )

        # Hesap girişi (bilgi verilmişse)
        if self.account and self.password:
            logger.info("🔑 MT5 hesabına giriş yapılıyor: %d", self.account)
            if not mt5.login(self.account, self.password, self.server):
                hata = mt5.last_error()
                mt5.shutdown()
                raise PermissionError(
                    f"MT5 hesap girişi başarısız (hesap={self.account}): {hata}"
                )

        # Terminal ve hesap bilgilerini logla
        info    = mt5.terminal_info()
        hesap   = mt5.account_info()
        if info is None or hesap is None:
            mt5.shutdown()
            raise ConnectionError(
                "MT5 terminal/hesap bilgisi alınamadı. "
                "Terminal bağlı ve oturum açık mı?"
            )

        logger.info(
            "✅ MT5 bağlantısı kuruldu\n"
            "   Terminal : %s  build=%d\n"
            "   Hesap    : %d  (%s)  Sunucu: %s\n"
            "   Bakiye   : %.2f %s  Equity: %.2f %s",
            info.name, info.build,
            hesap.login, hesap.name, hesap.server,
            hesap.balance, hesap.currency,
            hesap.equity, hesap.currency,
        )

        # Sembolü etkinleştir
        self._symbol_etkinlestir()
        self._connected = True

    def _symbol_etkinlestir(self) -> None:
        """Sembolün MarketWatch'ta görünür ve işlem yapılabilir olmasını sağlar."""
        if not mt5.symbol_select(self.symbol, True):
            raise ValueError(
                f"'{self.symbol}' sembolü MT5'te seçilemiyor.\n"
                f"Broker'ınızın NAS100 sembol adını kontrol edin "
                f"(US100, NAS100.cash, USTEC gibi farklı isimler olabilir)."
            )
        bilgi = mt5.symbol_info(self.symbol)
        if bilgi is None:
            raise ValueError(
                f"'{self.symbol}' sembolü için bilgi alınamadı."
            )
        logger.info(
            "📊 Sembol: %s | Spread: %.1f puan | Digits: %d | "
            "Min Lot: %.2f | Lot Adımı: %.2f",
            self.symbol, bilgi.spread, bilgi.digits,
            bilgi.volume_min, bilgi.volume_step,
        )

    def ensure_connected(self) -> bool:
        """
        Bağlantı kalp atışı kontrolü.
        Her _HEARTBEAT_INTERVAL saniyede bir terminal durumunu doğrular.
        Kopukluk tespit edilirse yeniden bağlanmayı dener.
        """
        simdi = time.time()
        if simdi - self._last_hb_ts < _HEARTBEAT_INTERVAL:
            return self._connected

        self._last_hb_ts = simdi
        info = mt5.terminal_info()
        if info is None or not info.connected:
            logger.warning("⚠️ MT5 bağlantısı kopuk — yeniden bağlanılıyor…")
            self._connected = False
            try:
                self._initialize()
            except Exception as exc:
                logger.error("MT5 yeniden bağlantı başarısız: %s", exc)
                return False

        return self._connected

    def shutdown(self) -> None:
        """MT5 terminalini güvenli biçimde kapatır."""
        logger.info("🔌 MT5 bağlantısı kapatılıyor…")
        mt5.shutdown()
        self._connected = False
        logger.info("✅ MT5 bağlantısı kapatıldı.")

    # ── Hesap ve Piyasa Durumu ─────────────────────────────────────────────

    def get_live_account_metrics(self) -> AccountMetrics:
        """
        MT5'ten anlık hesap metriklerini çeker.
        RiskGuardrails.check_drawdown_limits() tarafından çağrılır.

        Dönüş
        ──────
        AccountMetrics değer nesnesi.

        Hata
        ──────
        RuntimeError — MT5'ten hesap bilgisi alınamazsa.
        """
        hesap = mt5.account_info()
        if hesap is None:
            raise RuntimeError(
                f"MT5 hesap bilgisi alınamadı: {mt5.last_error()}"
            )
        return AccountMetrics(
            balance      = float(hesap.balance),
            equity       = float(hesap.equity),
            margin       = float(hesap.margin),
            free_margin  = float(hesap.margin_free),
            margin_level = float(hesap.margin_level or 0.0),
            profit       = float(hesap.profit),
            currency     = str(hesap.currency),
        )

    def get_open_positions(self) -> list[dict]:
        """
        Botun açık pozisyonlarını döner (magic number ile filtrelenir).

        Dönüş
        ──────
        Her pozisyon için: ticket, type, volume, price_open, sl, tp, profit
        """
        pozisyonlar = mt5.positions_get(symbol=self.symbol)
        if pozisyonlar is None:
            return []
        return [
            {
                "ticket":      p.ticket,
                "type":        "BUY" if p.type == mt5.POSITION_TYPE_BUY else "SELL",
                "volume":      p.volume,
                "price_open":  p.price_open,
                "sl":          p.sl,
                "tp":          p.tp,
                "profit":      p.profit,
                "magic":       p.magic,
            }
            for p in pozisyonlar
            if p.magic == self.magic_number
        ]

    def has_open_position(self) -> bool:
        """Bot tarafından açılmış aktif pozisyon var mı?"""
        return len(self.get_open_positions()) > 0

    def get_current_atr_price_info(self) -> dict[str, float]:
        """
        Anlık fiyat bilgisini döner: ask, bid, spread_points.
        Lot ve SL/TP hesaplaması öncesi main.py tarafından kullanılır.
        """
        tick = mt5.symbol_info_tick(self.symbol)
        if tick is None:
            raise RuntimeError(
                f"'{self.symbol}' sembolü için tick verisi alınamadı: "
                f"{mt5.last_error()}"
            )
        bilgi = mt5.symbol_info(self.symbol)
        return {
            "ask":           float(tick.ask),
            "bid":           float(tick.bid),
            "spread_points": float(bilgi.spread if bilgi else 0),
            "point":         float(bilgi.point if bilgi else 0.00001),
            "digits":        int(bilgi.digits if bilgi else 5),
        }

    # ── Emir İletimi ───────────────────────────────────────────────────────

    def send_market_order(
        self,
        action:   str,           # "STRONG_LONG" | "STRONG_SHORT"
        lot_size: float,
        sl_price: float,
        tp_price: float,
        comment:  str = "NasdaqBotV2",
    ) -> OrderResult:
        """
        Piyasaya anlık emir gönderir. SL ve TP emir içine gömülüdür
        (Prop Firm zorunluluğu: pozisyon açılırken SL/TP set edilmeli).

        Parametreler
        ─────────────
        action   : "STRONG_LONG" → BUY, "STRONG_SHORT" → SELL
        lot_size : ATR tabanlı hesaplanan lot büyüklüğü
        sl_price : Stop-Loss fiyatı (mutlak fiyat seviyesi)
        tp_price : Take-Profit fiyatı (mutlak fiyat seviyesi)
        comment  : MT5 emir yorumu (max 31 karakter)

        Dönüş
        ──────
        OrderResult değer nesnesi.
        """
        if not self._connected:
            raise ConnectionError("MT5 bağlantısı yok. ensure_connected() çağırın.")

        # Emir yönünü belirle
        if action == "STRONG_LONG":
            order_type  = mt5.ORDER_TYPE_BUY
            tip_str     = "BUY"
            tick        = mt5.symbol_info_tick(self.symbol)
            if tick is None:
                return self._basarisiz_sonuc(tip_str, lot_size, sl_price, tp_price,
                                             0, "Tick verisi alınamadı")
            fiyat = tick.ask

        elif action == "STRONG_SHORT":
            order_type  = mt5.ORDER_TYPE_SELL
            tip_str     = "SELL"
            tick        = mt5.symbol_info_tick(self.symbol)
            if tick is None:
                return self._basarisiz_sonuc(tip_str, lot_size, sl_price, tp_price,
                                             0, "Tick verisi alınamadı")
            fiyat = tick.bid

        else:
            return self._basarisiz_sonuc(
                "UNKNOWN", lot_size, sl_price, tp_price, 0,
                f"Geçersiz action değeri: '{action}'"
            )

        # Lot normalizasyonu — broker minimum/adım kurallarına uy
        lot_size = self._lot_normalize(lot_size)
        if lot_size <= 0:
            return self._basarisiz_sonuc(
                tip_str, lot_size, sl_price, tp_price, 0,
                "Lot büyüklüğü sıfır veya negatif — emir iptal edildi."
            )

        # SL/TP mantık kontrolü (Prop Firm güvenlik kapısı)
        dogrulama_hatasi = self._sl_tp_dogrula(tip_str, fiyat, sl_price, tp_price)
        if dogrulama_hatasi:
            return self._basarisiz_sonuc(
                tip_str, lot_size, sl_price, tp_price, 0, dogrulama_hatasi
            )

        # MT5 emir isteği nesnesi
        request = {
            "action":       mt5.TRADE_ACTION_DEAL,
            "symbol":       self.symbol,
            "volume":       lot_size,
            "type":         order_type,
            "price":        fiyat,
            "sl":           sl_price,
            "tp":           tp_price,
            "deviation":    self.max_slippage,
            "magic":        self.magic_number,
            "comment":      comment[:31],          # MT5 max 31 karakter
            "type_time":    mt5.ORDER_TIME_GTC,
            "type_filling": mt5.ORDER_FILLING_IOC,
        }

        logger.info(
            "📤 Emir gönderiliyor: %s %s  lot=%.2f  fiyat=%.5f  "
            "SL=%.5f  TP=%.5f",
            tip_str, self.symbol, lot_size, fiyat, sl_price, tp_price,
        )

        # Yeniden deneme döngüsü
        return self._emir_gonder_retry(request, tip_str, lot_size, sl_price, tp_price)

    def close_all_positions(self, reason: str = "Cuma zorla kapanış") -> list[OrderResult]:
        """
        RiskGuardrails.should_force_close() tarafından tetiklenebilir.
        Botun tüm açık pozisyonlarını piyasadan kapatır.
        """
        kapanan: list[OrderResult] = []
        pozisyonlar = self.get_open_positions()

        if not pozisyonlar:
            logger.info("Kapatılacak açık pozisyon yok.")
            return kapanan

        logger.warning(
            "⚠️ Tüm pozisyonlar kapatılıyor — Neden: %s  "
            "(Pozisyon sayısı: %d)", reason, len(pozisyonlar)
        )

        for poz in pozisyonlar:
            tick = mt5.symbol_info_tick(self.symbol)
            if tick is None:
                logger.error("Tick alınamadı — pozisyon kapatılamadı: %s", poz)
                continue

            # Ters taraf fiyat kullan
            if poz["type"] == "BUY":
                kapat_tip   = mt5.ORDER_TYPE_SELL
                kapat_fiyat = tick.bid
                tip_str     = "SELL (Kapat BUY)"
            else:
                kapat_tip   = mt5.ORDER_TYPE_BUY
                kapat_fiyat = tick.ask
                tip_str     = "BUY (Kapat SELL)"

            request = {
                "action":       mt5.TRADE_ACTION_DEAL,
                "symbol":       self.symbol,
                "volume":       poz["volume"],
                "type":         kapat_tip,
                "position":     poz["ticket"],
                "price":        kapat_fiyat,
                "deviation":    self.max_slippage,
                "magic":        self.magic_number,
                "comment":      reason[:31],
                "type_time":    mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

            sonuc = self._emir_gonder_retry(
                request, tip_str, poz["volume"], 0.0, 0.0
            )
            kapanan.append(sonuc)
            logger.info("Kapatma sonucu: %s", sonuc)

        return kapanan

    # ── Yardımcı / İç Metotlar ─────────────────────────────────────────────

    def _emir_gonder_retry(
        self,
        request:  dict,
        tip_str:  str,
        lot:      float,
        sl:       float,
        tp:       float,
    ) -> OrderResult:
        """Emir gönderimini _MAX_RETRY kez dener; başarısızlıkta OrderResult döner."""
        son_retcode = -1
        son_yorum   = "Bilinmeyen hata"

        for deneme in range(1, _MAX_RETRY + 1):
            sonuc = mt5.order_send(request)

            if sonuc is None:
                son_retcode = -1
                son_yorum   = f"mt5.order_send None döndü: {mt5.last_error()}"
                logger.warning(
                    "Deneme %d/%d başarısız: %s", deneme, _MAX_RETRY, son_yorum
                )
            elif sonuc.retcode == mt5.TRADE_RETCODE_DONE:
                logger.info(
                    "✅ Emir başarılı: ticket=%d  fiyat=%.5f  "
                    "lot=%.2f  retcode=%d",
                    sonuc.order, sonuc.price, lot, sonuc.retcode,
                )
                return OrderResult(
                    success    = True,
                    ticket     = sonuc.order,
                    order_type = tip_str,
                    symbol     = self.symbol,
                    lot        = lot,
                    price      = sonuc.price,
                    sl         = sl,
                    tp         = tp,
                    retcode    = sonuc.retcode,
                    comment    = "OK",
                )
            else:
                son_retcode = sonuc.retcode
                son_yorum   = getattr(sonuc, "comment", str(son_retcode))
                logger.warning(
                    "Deneme %d/%d — retcode=%d  yorum=%s",
                    deneme, _MAX_RETRY, son_retcode, son_yorum,
                )

            if deneme < _MAX_RETRY:
                time.sleep(_RETRY_DELAY)

        # Tüm denemeler tükendi
        logger.error(
            "❌ Emir gönderilemedi (%d deneme): retcode=%d  %s",
            _MAX_RETRY, son_retcode, son_yorum,
        )
        return self._basarisiz_sonuc(tip_str, lot, sl, tp, son_retcode, son_yorum)

    def _lot_normalize(self, lot: float) -> float:
        """Lot büyüklüğünü broker minimum ve adım kurallarına göre normalize eder."""
        bilgi = mt5.symbol_info(self.symbol)
        if bilgi is None:
            return round(lot, 2)

        min_lot  = bilgi.volume_min
        max_lot  = bilgi.volume_max
        lot_step = bilgi.volume_step

        # Adıma yuvarla
        if lot_step > 0:
            lot = round(round(lot / lot_step) * lot_step, 8)

        # Sınır kontrolleri
        lot = max(min_lot, min(lot, max_lot))
        return round(lot, 8)

    def _sl_tp_dogrula(
        self,
        tip:      str,
        fiyat:    float,
        sl_price: float,
        tp_price: float,
    ) -> str | None:
        """
        SL/TP mantık tutarlılığını kontrol eder.
        Hata varsa mesaj döner, yoksa None döner.
        """
        bilgi = mt5.symbol_info(self.symbol)
        if bilgi is None:
            return None  # Kontrol edilemiyor; geçtir

        min_stop = bilgi.trade_stops_level * bilgi.point

        if tip == "BUY":
            if sl_price >= fiyat:
                return (
                    f"BUY emrinde SL ({sl_price:.5f}) >= Giriş ({fiyat:.5f}) — geçersiz."
                )
            if tp_price <= fiyat:
                return (
                    f"BUY emrinde TP ({tp_price:.5f}) <= Giriş ({fiyat:.5f}) — geçersiz."
                )
            if (fiyat - sl_price) < min_stop:
                return (
                    f"BUY SL mesafesi ({fiyat - sl_price:.5f}) < broker minimum "
                    f"({min_stop:.5f})."
                )
        elif tip == "SELL":
            if sl_price <= fiyat:
                return (
                    f"SELL emrinde SL ({sl_price:.5f}) <= Giriş ({fiyat:.5f}) — geçersiz."
                )
            if tp_price >= fiyat:
                return (
                    f"SELL emrinde TP ({tp_price:.5f}) >= Giriş ({fiyat:.5f}) — geçersiz."
                )
            if (sl_price - fiyat) < min_stop:
                return (
                    f"SELL SL mesafesi ({sl_price - fiyat:.5f}) < broker minimum "
                    f"({min_stop:.5f})."
                )
        return None

    def _basarisiz_sonuc(
        self,
        tip_str:  str,
        lot:      float,
        sl:       float,
        tp:       float,
        retcode:  int,
        yorum:    str,
    ) -> OrderResult:
        return OrderResult(
            success    = False,
            ticket     = 0,
            order_type = tip_str,
            symbol     = self.symbol,
            lot        = lot,
            price      = 0.0,
            sl         = sl,
            tp         = tp,
            retcode    = retcode,
            comment    = yorum,
        )

    # ── Context Manager Desteği ────────────────────────────────────────────

    def __enter__(self) -> "MetaTraderBridge":
        return self

    def __exit__(self, *_: Any) -> None:
        self.shutdown()

    def __repr__(self) -> str:
        return (
            f"MetaTraderBridge(symbol={self.symbol!r}, "
            f"connected={self._connected}, magic={self.magic_number})"
        )