"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  main.py — NASDAQ BOT V2 · Canlı Çalıştırıcı                               ║
║  Senior Trading Infrastructure Engineer · Live Loop Engine                   ║
║  ─────────────────────────────────────────────────────────────────────────  ║
║  Mimari:                                                                     ║
║    · 15 dakikalık bar kapanışı senkronize zaman döngüsü                     ║
║    · MT5 piyasa emri iletimi (MetaTraderBridge)                             ║
║    · RiskGuardrails: günlük %4 / toplam %9 drawdown koruyucusu              ║
║    · Cuma zorla kapanış (Prop Firm uyumluluğu)                              ║
║    · Heartbeat bağlantı izleme + otomatik yeniden bağlanma                  ║
╚══════════════════════════════════════════════════════════════════════════════╝

ÇALIŞTIRMA
──────────
    python main.py

ORTAM GEREKSİNİMLERİ
────────────────────
    · Windows işletim sistemi (MT5 zorunluluğu)
    · MetaTrader 5 terminali açık ve demo hesabına bağlı
    · pip install MetaTrader5 yfinance xgboost pandas numpy ta
    · Eğitilmiş modeller: models/xgb_model_long.json ve models/xgb_model_short.json
"""

from __future__ import annotations

import logging
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

# ──────────────────────────────────────────────────────────────────────────────
# LOGLAMA — Her çalıştırma hem konsola hem log dosyasına yazar
# ──────────────────────────────────────────────────────────────────────────────

LOG_DIR  = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"live_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

_fmt = logging.Formatter(
    fmt     = "%(asctime)s  [%(levelname)-8s]  %(name)-22s  %(message)s",
    datefmt = "%Y-%m-%d %H:%M:%S",
)

_console_handler = logging.StreamHandler(sys.stdout)
_console_handler.setFormatter(_fmt)
_console_handler.setLevel(logging.INFO)

_file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
_file_handler.setFormatter(_fmt)
_file_handler.setLevel(logging.DEBUG)

logging.basicConfig(level=logging.DEBUG, handlers=[_console_handler, _file_handler])
logger = logging.getLogger("LiveEngine")

# ──────────────────────────────────────────────────────────────────────────────
# PROJE MODÜLLERİ
# ──────────────────────────────────────────────────────────────────────────────
# Proje kökünü sys.path'e ekle (farklı çalıştırma dizinlerine karşı)
sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import config.settings as cfg
    from data.data_feed            import DataFeed
    from execution.broker_bridge   import MetaTraderBridge, OrderResult
    from ml.features               import FeatureEngine
    from risk.guardrails           import RiskGuardrails
    from risk.portfolio            import PortfolioManager
    from strategy.engine           import StrategyEngine
    from strategy.voting           import VotingMechanism
except ImportError as exc:
    logger.critical(
        "❌ Proje modülü yüklenemedi: %s\n"
        "   Betiği projenin kök dizininden çalıştırdığınızdan emin olun.\n"
        "   Gerekli: config/, data/, execution/, ml/, risk/, strategy/",
        exc,
    )
    sys.exit(1)

# ──────────────────────────────────────────────────────────────────────────────
# SABİTLER
# ──────────────────────────────────────────────────────────────────────────────

# 15 dakikalık bar kapanma dakikaları (0-59 arası)
BAR_KAPANMA_DAKIKALARI: frozenset[int] = frozenset({14, 29, 44, 59})

# Bar kapanmasına kaç saniye kala tetikleme yapılacağı
TETIKLEME_SANIYE_ONCESI: int = 2

# Ana döngü uyku süresi (saniye) — CPU kullanımını düşürür
ANA_DONGU_UYKU: float = 0.8

# Veri indirirken ısınma için minimum bar sayısı
MIN_INDICATOR_WARMUP_BARS: int = 50

# Bağlantı kopması sonrası yeniden deneme beklemesi (saniye)
RECONNECT_BEKLEME: int = 30

# Günlük drawdown sıfırlama saati (UTC) — prop firm kuralına göre ayarla
GUNLUK_SIFIRLAMA_SAATI: int = 0    # 00:00 UTC

# ──────────────────────────────────────────────────────────────────────────────
# YARDIMCI FONKSİYONLAR
# ──────────────────────────────────────────────────────────────────────────────

def _sutun_ciz(karakter: str = "─", uzunluk: int = 72) -> str:
    return karakter * uzunluk


def _banner_yazdir() -> None:
    """Başlangıç bannerını terminale basar."""
    print(_sutun_ciz("═"))
    print("  NASDAQ BOT V2 — CANLI ÇALIŞTIRICISI")
    print(f"  Başlangıç : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Timeframe : {getattr(cfg, 'TIMEFRAME', '15m')}")
    print(f"  Veri      : {getattr(cfg, 'SYMBOL', 'QQQ')} (yfinance)  |  MT5: {getattr(cfg, 'MT5_SYMBOL', 'US100')}")
    print(f"  Log Dosya : {LOG_FILE}")
    print(_sutun_ciz("═"))


def _tetikleme_vakti_mi(dakika: int, saniye: int) -> bool:
    """
    Mevcut zaman bir 15m bar kapanmasından TETIKLEME_SANIYE_ONCESI saniye
    öncesinde mi? Evet ise True döner.
    """
    return (
        dakika in BAR_KAPANMA_DAKIKALARI
        and saniye == (60 - TETIKLEME_SANIYE_ONCESI)
    )


# ──────────────────────────────────────────────────────────────────────────────
# CANLI PORTFÖY PROXY — RiskGuardrails Köprüsü
# ──────────────────────────────────────────────────────────────────────────────

class _LivePortfolioProxy:
    """
    MT5 hesap metriklerini RiskGuardrails'in PortfolioManager API'sine adapte eden
    hafif köprü sınıfı.

    Sorun
    ──────
    PortfolioManager nesnesi bakiyesini kendi iç işlem kaydından hesaplar;
    canlı modda gerçek P&L MT5 terminalinde tutuluyor. Guardrails, her kontrolde
    `.equity` ve `.daily_peak_equity` erişimi beklediğinden PortfolioManager'ı
    doğrudan kullanamayız.

    Çözüm
    ──────
    Her bar kapanışında MT5'ten çekilen `AccountMetrics` verisiyle bu proxy
    anlık olarak oluşturulur ve guardrails çağrılarına argüman olarak geçirilir.
    PortfolioManager'a, guardrails.py'ye veya portfolio.py'ye dokunulmaz.
    """

    __slots__ = ("equity", "balance", "daily_peak_equity")

    def __init__(
        self,
        equity:      float,
        balance:     float,
        daily_peak:  float,
    ) -> None:
        self.equity            = equity
        self.balance           = balance
        self.daily_peak_equity = daily_peak

    def reset_daily_peak(self) -> None:
        """reset_daily_drawdown() tarafından çağrılan PortfolioManager uyumu."""
        self.daily_peak_equity = self.equity


def _atr_bazli_lot_ve_sl_tp(
    action:          str,
    entry_price:     float,
    atr_value:       float,
    balance:         float,
    risk_per_trade:  float,
    atr_multiplier:  float,
    reward_risk:     float,
    point:           float,
    lot_min:         float = 0.01,
    lot_max:         float = 100.0,
    lot_step:        float = 0.01,
    contract_size:   float = 1.0,
    max_notional_leverage: float = 20.0,
) -> tuple[float, float, float]:
    """
    ATR tabanlı lot büyüklüğü, SL ve TP seviyelerini hesaplar.
    PortfolioManager mantığını canlı modda yansıtır.

    Parametreler
    ─────────────
    action          : "STRONG_LONG" | "STRONG_SHORT"
    entry_price     : Giriş fiyatı (ask veya bid)
    atr_value       : Son barda hesaplanan ATR değeri
    balance         : Canlı hesap bakiyesi
    risk_per_trade  : config.RISK_PER_TRADE (örn. 0.01 → %1)
    atr_multiplier  : config.ATR_MULTIPLIER (örn. 1.5)
    reward_risk     : config.REWARD_RISK_RATIO (örn. 2.0)
    point           : Broker pip değeri
    contract_size   : Lot başına kontrat büyüklüğü

    Dönüş
    ──────
    (lot_size, sl_price, tp_price) üçlüsü
    """
    sl_mesafe = atr_value * atr_multiplier

    if action == "STRONG_LONG":
        sl_price = entry_price - sl_mesafe
        tp_price = entry_price + sl_mesafe * reward_risk
    else:  # STRONG_SHORT
        sl_price = entry_price + sl_mesafe
        tp_price = entry_price - sl_mesafe * reward_risk

    # Lot hesaplama: riske edilecek tutar / (SL mesafesi × pip değeri)
    riske_edilen_tutar = balance * risk_per_trade
    sl_mesafe_pip      = sl_mesafe / point if point > 1e-10 else sl_mesafe

    # Pip değeri hesabı (basit tahmin — broker'a göre farklılık gösterebilir)
    pip_degeri = point * contract_size
    if pip_degeri < 1e-10:
        pip_degeri = 1.0

    lot = riske_edilen_tutar / (sl_mesafe_pip * pip_degeri)

    if entry_price > 0 and contract_size > 0:
        notional_cap_lot = (balance * max_notional_leverage) / (entry_price * contract_size)
        lot_max = min(lot_max, notional_cap_lot)

    # Lot normalize et
    if lot_step > 0:
        lot = round(round(lot / lot_step) * lot_step, 8)
    lot = max(lot_min, min(lot, lot_max))

    logger.debug(
        "LOT HESAP | entry=%.5f  ATR=%.5f  SL_mesafe=%.5f  "
        "risk=$%.2f  lot=%.2f  SL=%.5f  TP=%.5f",
        entry_price, atr_value, sl_mesafe,
        riske_edilen_tutar, lot, sl_price, tp_price,
    )

    return round(lot, 8), round(sl_price, 5), round(tp_price, 5)


# ──────────────────────────────────────────────────────────────────────────────
# CANLI BOT MOTORU
# ──────────────────────────────────────────────────────────────────────────────

class LiveTradingEngine:
    """
    Botun canlı çalışma döngüsünü yöneten ana sınıf.

    Bileşen Bağlantı Şeması:
    ────────────────────────
    Zaman → [Bar Kapanma Tespiti]
             ↓
        DataFeed (yfinance)
             ↓
        FeatureEngine
             ↓
        StrategyEngine  →  base_signal
             ↓
        VotingMechanism →  final_signal (XGBoost vizesi)
             ↓
    [Pozisyon Kontrolü]  →  Açık pozisyon var mı?
             ↓ (hayır)
    [RiskGuardrails]    →  Drawdown limiti aşıldı mı? Cuma mı?
             ↓ (hayır)
    [ATR Lot / SL / TP Hesabı]
             ↓
    MetaTraderBridge.send_market_order()
    """

    def __init__(self) -> None:
        # Sistem bileşenleri
        self.data_feed      : DataFeed          | None = None
        self.feature_engine : FeatureEngine      | None = None
        self.strategy_engine: StrategyEngine     | None = None
        self.voting         : VotingMechanism    | None = None
        self.portfolio      : PortfolioManager   | None = None
        self.guardrails     : RiskGuardrails     | None = None
        self.bridge         : MetaTraderBridge   | None = None

        # Durum takibi
        self._son_tetikleme_dakika:  int   = -1   # Aynı barı iki kez işlemez
        self._gunluk_sifirlama_gun:  int   = -1   # Günlük DD sıfırlama takibi
        self._calisma_sayaci:        int   = 0
        self._islem_sayaci:          int   = 0
        self._hata_sayaci:           int   = 0
        # Günlük drawdown hesabı için MT5 equity tepe değeri
        # (ilk bar tetiklendiğinde canlı bakiyeyle güncellenecek)
        self._daily_peak_equity:     float = float(
            getattr(cfg, "INITIAL_BALANCE", 10_000.0)
        )

    # ── Başlangıç ─────────────────────────────────────────────────────────

    def basla(self) -> None:
        """
        Tüm bileşenleri ayağa kaldırır; hazır değilse ValueError fırlatır.
        main() tarafından çağrılır.
        """
        logger.info("⚙️  Bileşenler başlatılıyor…")

        # 1. Veri ve özellik katmanları
        self.data_feed      = DataFeed()
        self.feature_engine = FeatureEngine()
        logger.info("   ✔ DataFeed + FeatureEngine hazır.")

        # 2. Strateji ve model
        self.strategy_engine = StrategyEngine()
        self.voting          = VotingMechanism()   # Eğitilmiş modeli yükler
        logger.info("   ✔ StrategyEngine + VotingMechanism (XGBoost) hazır.")

        # 3. Risk katmanları
        # PortfolioManager yapıcısı parametre almaz — bakiye config'den okur.
        # Canlı modda gerçek P&L MT5'ten izlenir; portfolio lokal shadow görevi görür.
        self.portfolio  = PortfolioManager()
        self.guardrails = RiskGuardrails()
        logger.info(
            "   ✔ PortfolioManager + RiskGuardrails hazır."
        )

        # 4. MT5 köprüsü — kimlik bilgileri config'den
        # MT5_SYMBOL: broker'daki işlem sembolü (örn. "US100")
        # SYMBOL    : yfinance veri sembolü     (örn. "QQQ")  — bu ayrı kalır
        mt5_symbol = getattr(cfg, "MT5_SYMBOL", "US100")
        self.bridge = MetaTraderBridge(
            symbol   = mt5_symbol,
            account  = int(getattr(cfg,  "MT5_ACCOUNT",  0)),
            password = str(getattr(cfg,  "MT5_PASSWORD", "")),
            server   = str(getattr(cfg,  "MT5_SERVER",   "")),
            mt5_path = str(getattr(cfg,  "MT5_PATH",     "")),
        )
        logger.info(
            "   ✔ MetaTraderBridge hazır. "
            "MT5 Sembol: %s  |  Veri Sembolü (yfinance): %s",
            mt5_symbol, getattr(cfg, "SYMBOL", "QQQ"),
        )

        logger.info("✅ Tüm bileşenler hazır — Canlı döngü başlıyor.")

    def durdur(self) -> None:
        """Kaynakları temizler ve MT5 bağlantısını kapatır."""
        logger.info("🛑 Canlı motor durduruluyor…")
        if self.bridge:
            try:
                self.bridge.shutdown()
            except Exception:
                pass
        logger.info(
            "📊 Oturum özeti | Döngü: %d  İşlem: %d  Hata: %d",
            self._calisma_sayaci, self._islem_sayaci, self._hata_sayaci,
        )

    # ── Ana Döngü ──────────────────────────────────────────────────────────

    def calistir(self) -> None:
        """
        Sonsuz zaman döngüsü.
        Her ANA_DONGU_UYKU saniyede bir sistem saatini kontrol eder;
        bar kapanma anında tüm iş akışını tetikler.
        """
        logger.info("🔁 Ana döngü başladı. Çıkmak için Ctrl+C.")
        logger.info(
            "   Bar tetikleme: her 15m barın kapanmasından "
            "%d saniye önce.", TETIKLEME_SANIYE_ONCESI
        )

        while True:
            try:
                simdi   = datetime.now(tz=timezone.utc)
                dakika  = simdi.minute
                saniye  = simdi.second
                gun     = simdi.day

                # ── Günlük drawdown sıfırlama (00:00 UTC) ─────────────────
                if (simdi.hour == GUNLUK_SIFIRLAMA_SAATI
                        and gun != self._gunluk_sifirlama_gun):
                    self._gunluk_drawdown_sifirla()
                    self._gunluk_sifirlama_gun = gun

                # ── Heartbeat — MT5 bağlantısı hâlâ sağlıklı mı? ─────────
                if self.bridge and not self.bridge.ensure_connected():
                    logger.error(
                        "MT5 bağlantısı koptu. %d saniye sonra yeniden "
                        "denenecek…", RECONNECT_BEKLEME
                    )
                    time.sleep(RECONNECT_BEKLEME)
                    continue

                # ── Bar kapanma tespiti ────────────────────────────────────
                if (
                    _tetikleme_vakti_mi(dakika, saniye)
                    and dakika != self._son_tetikleme_dakika
                ):
                    self._son_tetikleme_dakika = dakika
                    self._calisma_sayaci += 1
                    logger.info(
                        _sutun_ciz("─") + "\n"
                        "🕐 BAR KAPANMA TETİKLEMESİ  "
                        "[%s UTC]  (#%d)",
                        simdi.strftime("%H:%M:%S"), self._calisma_sayaci,
                    )
                    # Bar kapanmasına kadar bekle (~2 saniye)
                    time.sleep(TETIKLEME_SANIYE_ONCESI + 0.5)
                    self._bar_kapanis_islemi(simdi)

                time.sleep(ANA_DONGU_UYKU)

            except KeyboardInterrupt:
                logger.info("\n⌨️  Kullanıcı tarafından durduruldu (Ctrl+C).")
                break

            except Exception as exc:
                # Beklenmedik hata — logla, çökme, kurtarma bekle
                self._hata_sayaci += 1
                logger.error(
                    "❌ Ana döngüde beklenmedik hata (#%d): %s",
                    self._hata_sayaci, exc,
                )
                logger.debug(traceback.format_exc())

                if self._hata_sayaci >= 10:
                    logger.critical(
                        "Ardışık hata sayısı 10'a ulaştı — güvenlik kapatması."
                    )
                    break

                logger.info("%d saniye beklenip devam ediliyor…", RECONNECT_BEKLEME)
                time.sleep(RECONNECT_BEKLEME)

    # ── Bar Kapanış İş Akışı ───────────────────────────────────────────────

    def _bar_kapanis_islemi(self, simdi: datetime) -> None:
        """
        Her 15m bar kapandığında çalışan tam iş akışı.
        Herhangi bir adımda hata oluşursa işlem iptal edilir,
        ana döngü çökemez.

        Parameters
        ----------
        simdi : datetime
            calistir() döngüsünden gelen UTC zaman damgası.
            Guardrails zaman filtreleri ve Cuma kontrolü için kullanılır.
        """
        try:
            # ── Adım 1: Cuma zorla kapanış kontrolü ───────────────────────
            # DÜZELTME: should_force_close(timestamp) imzasına uygun timestamp geçildi
            if self.guardrails and self.guardrails.should_force_close(simdi):
                logger.warning(
                    "📅 Cuma zorla kapanış sinyali — tüm pozisyonlar kapatılıyor."
                )
                if self.bridge:
                    self.bridge.close_all_positions(reason="Cuma Zorla Kapanış")
                return

            # ── Adım 2: Drawdown limit kontrolü ───────────────────────────
            # DÜZELTME: check_drawdown_limits(portfolio) imzasına uygun proxy geçildi.
            # _LivePortfolioProxy, MT5 equity'sini PortfolioManager API'siyle sarmalar.
            if self.bridge and self.guardrails:
                try:
                    canli_metrikler = self.bridge.get_live_account_metrics()

                    # Günlük tepe equity'yi MT5 gerçek değeriyle izle
                    if canli_metrikler.equity > self._daily_peak_equity:
                        self._daily_peak_equity = canli_metrikler.equity

                    proxy = _LivePortfolioProxy(
                        equity     = canli_metrikler.equity,
                        balance    = canli_metrikler.balance,
                        daily_peak = self._daily_peak_equity,
                    )
                    drawdown_ihlal = self.guardrails.check_drawdown_limits(proxy)
                    if drawdown_ihlal:
                        logger.warning(
                            "🚨 DRAWDOWN LİMİTİ AŞILDI — işlem durduruldu.\n"
                            "   Bakiye: %.2f  Equity: %.2f  "
                            "Drawdown: %.2f%%",
                            canli_metrikler.balance,
                            canli_metrikler.equity,
                            canli_metrikler.drawdown_pct,
                        )
                        return
                except Exception as exc:
                    logger.error(
                        "Hesap metrikleri alınamadı (guardrails atlandı): %s", exc
                    )

            # ── Adım 3: Açık pozisyon kontrolü ────────────────────────────
            if self.bridge and self.bridge.has_open_position():
                logger.info(
                    "📌 Açık pozisyon mevcut — yeni sinyal aranmıyor."
                )
                return

            # ── Adım 4: İşlem saati kontrolü ──────────────────────────────
            # DÜZELTME: is_trading_allowed(portfolio, timestamp) imzasına uygun
            # proxy ve timestamp geçildi; kırılgan getattr pattern kaldırıldı.
            if self.guardrails and self.portfolio:
                # Proxy'yi zaten yukarıda oluşturduk; bağlantı yoksa fallback
                try:
                    _proxy_for_time = proxy  # type: ignore[name-defined]
                except NameError:
                    _proxy_for_time = _LivePortfolioProxy(
                        equity     = self._daily_peak_equity,
                        balance    = self._daily_peak_equity,
                        daily_peak = self._daily_peak_equity,
                    )
                if not self.guardrails.is_trading_allowed(_proxy_for_time, simdi):
                    logger.info("🕐 İşlem saatleri dışında — atlanıyor.")
                    return

            # ── Adım 5: Canlı veri indir ve indikatörleri hesapla ─────────
            # DÜZELTME: force_refresh=True — her bar kapanışında taze veri alınır,
            # bayat cache'e asla düşülmez.
            logger.info("📥 Canlı veri indiriliyor (force_refresh=True)…")
            df_ham = self.data_feed.download_historical_data(force_refresh=True)

            if df_ham is None or df_ham.empty:
                logger.warning("Veri alınamadı — bar atlanıyor.")
                return

            if len(df_ham) < MIN_INDICATOR_WARMUP_BARS:
                logger.warning(
                    "Yetersiz bar (%d < %d) — ısınma bekleniyor.",
                    len(df_ham), MIN_INDICATOR_WARMUP_BARS,
                )
                return

            df_ind = self.feature_engine.calculate_indicators(df_ham)

            # ── Adım 6: Temel sinyal üret ──────────────────────────────────
            current_idx = len(df_ind) - 1
            base_signal = self.strategy_engine.generate_base_signal(
                df_ind, current_idx
            )
            logger.info("📡 Temel sinyal: %s", base_signal)

            if base_signal == "HOLD":
                logger.info("   → HOLD — XGBoost değerlendirmesine gerek yok.")
                return

            # ── Adım 7: XGBoost vizesi ─────────────────────────────────────
            # DÜZELTME: voting.vote() → voting.decide_trade() imzasına uygun çağrı.
            # decide_trade() içinde feature üretimi yapıldığından ayrı
            # generate_live_features() çağrısı kaldırıldı (look-ahead bias zırhı korundu).
            final_signal = self.voting.decide_trade(
                df            = df_ind,
                current_index = current_idx,
                base_signal   = base_signal,
                feature_engine = self.feature_engine,
            )
            logger.info("🤖 XGBoost final sinyal: %s", final_signal)

            if final_signal == "HOLD":
                logger.info("   → XGBoost VETO — işlem yapılmıyor.")
                return

            # ── Adım 8: ATR tabanlı lot / SL / TP hesapla ─────────────────
            canli_metrikler = self.bridge.get_live_account_metrics()
            fiyat_bilgisi   = self.bridge.get_current_atr_price_info()

            # ATR değerini son bardan al
            atr_kolonu = None
            for kol in ("ATR", "atr", "ATR_14"):
                if kol in df_ind.columns:
                    atr_kolonu = kol
                    break

            if atr_kolonu is None:
                logger.warning(
                    "ATR kolonu bulunamadı — işlem iptal edildi.\n"
                    "   Mevcut kolonlar: %s",
                    list(df_ind.columns[:20]),
                )
                return

            atr_degeri = float(df_ind[atr_kolonu].iloc[-1])
            if atr_degeri <= 0:
                logger.warning("Geçersiz ATR değeri (%.6f) — işlem iptal.", atr_degeri)
                return

            # Giriş fiyatı
            if final_signal == "STRONG_LONG":
                entry_price = fiyat_bilgisi["ask"]
            else:
                entry_price = fiyat_bilgisi["bid"]

            lot_size, sl_price, tp_price = _atr_bazli_lot_ve_sl_tp(
                action          = final_signal,
                entry_price     = entry_price,
                atr_value       = atr_degeri,
                balance         = canli_metrikler.balance,
                risk_per_trade  = float(getattr(cfg, "RISK_PER_TRADE",  0.01)),
                atr_multiplier  = float(getattr(cfg, "ATR_MULTIPLIER",  1.5)),
                reward_risk     = float(getattr(cfg, "REWARD_RISK_RATIO", 2.0)),  # settings: 2.0
                point           = fiyat_bilgisi["point"],
                lot_max         = float(getattr(cfg, "MAX_LOT_LIMIT", 1000.0)),
                contract_size   = float(getattr(cfg, "CONTRACT_SIZE",  1.0)),
                max_notional_leverage = float(getattr(cfg, "MAX_NOTIONAL_LEVERAGE", 20.0)),
            )

            logger.info(
                "💰 Emir parametreleri:\n"
                "   Sinyal    : %s\n"
                "   Giriş     : %.5f\n"
                "   Lot       : %.2f\n"
                "   SL        : %.5f  (mesafe: %.5f)\n"
                "   TP        : %.5f  (mesafe: %.5f)\n"
                "   ATR       : %.5f\n"
                "   Bakiye    : %.2f %s",
                final_signal,
                entry_price,
                lot_size,
                sl_price,  abs(entry_price - sl_price),
                tp_price,  abs(tp_price    - entry_price),
                atr_degeri,
                canli_metrikler.balance, canli_metrikler.currency,
            )

            # ── Adım 9: Emri gönder ────────────────────────────────────────
            sonuc: OrderResult = self.bridge.send_market_order(
                action   = final_signal,
                lot_size = lot_size,
                sl_price = sl_price,
                tp_price = tp_price,
                comment  = f"NasdaqV2_{self._islem_sayaci + 1}",
            )

            if sonuc.success:
                self._islem_sayaci += 1
                self._hata_sayaci = 0   # Başarılı işlem hata sayacını sıfırlar
                logger.info(
                    "✅ İŞLEM AÇILDI #%d\n   %s",
                    self._islem_sayaci, sonuc,
                )
            else:
                logger.error("❌ İŞLEM BAŞARISIZ: %s", sonuc)

        except Exception as exc:
            self._hata_sayaci += 1
            logger.error(
                "❌ Bar kapanış işleminde hata (#%d): %s",
                self._hata_sayaci, exc,
            )
            logger.debug(traceback.format_exc())

    # ── Günlük Drawdown Sıfırlama ──────────────────────────────────────────

    def _gunluk_drawdown_sifirla(self) -> None:
        """
        Gün başında RiskGuardrails'in günlük drawdown sayacını sıfırlar.
        Prop Firm kuralı: günlük limit her gün 00:00 UTC'de yenilenir.

        MT5'ten anlık equity çekilir, _daily_peak_equity güncellenir ve
        guardrails.reset_daily_drawdown(portfolio) PortfolioManager uyumlu
        proxy ile çağrılır — guardrails.py'ye dokunulmaz.
        """
        logger.info("🔄 Günlük drawdown sayacı sıfırlanıyor (00:00 UTC).")
        if not (self.guardrails and self.bridge):
            return

        try:
            metrikler = self.bridge.get_live_account_metrics()
            # Günlük tepe equity'yi MT5 gerçek değeriyle sıfırla
            self._daily_peak_equity = metrikler.equity
            proxy = _LivePortfolioProxy(
                equity     = metrikler.equity,
                balance    = metrikler.balance,
                daily_peak = metrikler.equity,
            )
            self.guardrails.reset_daily_drawdown(proxy)
            logger.info(
                "   ✔ Günlük drawdown sıfırlandı. "
                "Yeni tepe equity: %.2f %s",
                metrikler.equity, metrikler.currency,
            )
        except Exception as exc:
            logger.warning(
                "Günlük sıfırlama sırasında MT5 metrikleri alınamadı: %s\n"
                "   Guardrails manuel sıfırlama gerekebilir.", exc
            )


# ──────────────────────────────────────────────────────────────────────────────
# ÖN KONTROLLER
# ──────────────────────────────────────────────────────────────────────────────

def on_kontroller() -> None:
    """Kritik dosya ve konfigürasyonları doğrular; sorun varsa çıkış yapar."""
    hatalar: list[str] = []

    model_dir = Path(getattr(cfg, "MODEL_DIR", "models"))
    for model_adi in ("xgb_model_long.json", "xgb_model_short.json"):
        model_yolu = model_dir / model_adi
        if not model_yolu.exists():
            hatalar.append(
                f"Eğitilmiş model bulunamadı: {model_yolu}\n"
                "   Çözüm: python ml/trainer.py çalıştırın."
            )

    settings_yolu = Path("config") / "settings.py"
    if not settings_yolu.exists():
        hatalar.append("config/settings.py bulunamadı.")

    for kl in ("data", "ml", "strategy", "risk", "execution"):
        if not Path(kl).is_dir():
            hatalar.append(f"Proje klasörü eksik: {kl}/")

    if hatalar:
        for h in hatalar:
            logger.critical("❌ %s", h)
        sys.exit(2)

    # Önemli config değerlerini logla
    logger.info(
        "📋 Aktif Konfigürasyon:\n"
        "   TIMEFRAME              = %s\n"
        "   XGB_PROBABILITY_THR    = %.2f\n"
        "   REWARD_RISK_RATIO      = %.2f\n"
        "   RISK_PER_TRADE         = %.2f%%\n"
        "   ATR_MULTIPLIER         = %.2f\n"
        "   INITIAL_BALANCE        = %.2f",
        getattr(cfg, "TIMEFRAME",                "15m"),
        getattr(cfg, "XGB_PROBABILITY_THRESHOLD", 0.60),
        getattr(cfg, "REWARD_RISK_RATIO",          2.0),
        getattr(cfg, "RISK_PER_TRADE",             0.01) * 100,
        getattr(cfg, "ATR_MULTIPLIER",             1.5),
        getattr(cfg, "INITIAL_BALANCE",        10_000.0),
    )


# ──────────────────────────────────────────────────────────────────────────────
# ENTRYPOINT
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    _banner_yazdir()

    # ── Ön kontroller ─────────────────────────────────────────────────────────
    on_kontroller()

    # ── Motor nesnesi ──────────────────────────────────────────────────────────
    engine = LiveTradingEngine()

    try:
        engine.basla()
        engine.calistir()

    except ConnectionError as exc:
        logger.critical(
            "❌ MT5 BAĞLANTI HATASI: %s\n"
            "   MetaTrader 5 terminalinin açık ve demo hesabına\n"
            "   bağlı olduğundan emin olun, ardından yeniden deneyin.",
            exc,
        )
        sys.exit(3)

    except PermissionError as exc:
        logger.critical(
            "❌ MT5 KİMLİK DOĞRULAMA HATASI: %s\n"
            "   config/settings.py içindeki MT5_ACCOUNT, MT5_PASSWORD,\n"
            "   MT5_SERVER değerlerini kontrol edin.",
            exc,
        )
        sys.exit(4)

    except KeyboardInterrupt:
        logger.info("Kullanıcı tarafından durduruldu.")

    except Exception as exc:
        logger.critical(
            "❌ FATAL HATA: %s\n%s",
            exc, traceback.format_exc(),
        )
        sys.exit(5)

    finally:
        engine.durdur()
        logger.info("👋 Bot kapandı.")


if __name__ == "__main__":
    main()
