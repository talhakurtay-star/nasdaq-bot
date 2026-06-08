"""
config/settings.py
------------------
Merkezi Yapılandırma Dosyası — nasdaq_bot_v2

Tüm modüller bu dosyadan parametrelerini import eder.
Dinamik parametreler (XGB threshold, Timeframe, Limitler vb.) öncelikle 
ortam değişkenlerinden (os.getenv) okunarak, fiziksel dosya yaması yapmadan
stres testleri ve otomatik senaryolarla yönetilebilir hale getirilmiştir.
"""

import os
from pathlib import Path

# ── Dizinler ──────────────────────────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent
CACHE_DIR = os.path.join(str(BASE_DIR), "cache")
MODEL_DIR = os.path.join(str(BASE_DIR), "models")

# Dizinlerin varlığını kontrol et
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

# ── Zaman ve Sembol Ayarları ──────────────────────────────────────────────────
# SYMBOL      → yfinance veri çekme sembolü (QQQ/NASDAQ ETF proxy)
# MT5_SYMBOL  → broker/MT5 işlem sembolü (ICMarkets: "NAS100", XM: "USTEC" vb.)
SYMBOL = os.getenv("STRESS_SYMBOL", "TECH")
MT5_SYMBOL = os.getenv("STRESS_MT5_SYMBOL", "TECH")

VIX_SYMBOL = os.getenv("STRESS_VIX_SYMBOL", "^VIX")  # Volatilite endeksi
SPY_SYMBOL = os.getenv("STRESS_SPY_SYMBOL", "SPY")    # Korelasyon referansı (S&P 500)
WATCHLIST = [SYMBOL, SPY_SYMBOL, VIX_SYMBOL]

# TIMEFRAME: Grafik periyodu ("15m", "1h", "5m")
TIMEFRAME = "15m"

# BACKTEST_DAYS: Geriye dönük çekilecek veri gün sayısı
BACKTEST_DAYS = 360

# ── Teknik İndikatör Parametreleri ───────────────────────────────────────────
EMA_FAST = int(os.getenv("STRESS_EMA_FAST", "9"))
EMA_SLOW = int(os.getenv("STRESS_EMA_SLOW", "21"))
RSI_PERIOD = int(os.getenv("STRESS_RSI_PERIOD", "14"))
ATR_PERIOD = int(os.getenv("STRESS_ATR_PERIOD", "14"))

# RSI Eşikleri
RSI_OVERBOUGHT = int(os.getenv("STRESS_RSI_OB", "80"))
RSI_OVERSOLD = int(os.getenv("STRESS_RSI_OS", "20"))

# ── XGBoost Olasılık Eşiği (Veto Mekanizması) ────────────────────────────────
# Modelin güven skoru bu eşiğin altındaysa işlem VETO edilir.
XGB_PROBABILITY_THRESHOLD = 0.54

# ── Portföy ve Sermaye Ayarları ──────────────────────────────────────────────
INITIAL_BALANCE = float(os.getenv("STRESS_INITIAL_BALANCE", "100000.0"))
REWARD_RISK_RATIO = 1.5
ATR_MULTIPLIER = float(os.getenv("STRESS_ATR_MULT", "1.5"))     # Stop Loss için ATR katsayısı

# ── Risk Yönetimi ve Prop Firm Limitleri ──────────────────────────────────────
# RISK_PER_TRADE: İşlem başına risk yüzdesi (Örn: 1.00 -> %1)
RISK_PER_TRADE_PCT = float(os.getenv("STRESS_RISK_PCT", "0.75"))
RISK_PER_TRADE = RISK_PER_TRADE_PCT / 100.0                     # Lojik işlemlerde kullanılan decimal değer

# DAILY_DRAWDOWN_LIMIT: Günlük maksimum kayıp limiti (% cinsinden)
DAILY_DRAWDOWN_LIMIT = float(os.getenv("STRESS_DAILY_DD", "4.0"))  # Varsayılan %4.0
MAX_DAILY_DRAWDOWN_PCT = DAILY_DRAWDOWN_LIMIT / 100.0

# TOTAL_DRAWDOWN_LIMIT: Hesap genelinde maksimum kayıp limiti (% cinsinden)
TOTAL_DRAWDOWN_LIMIT = float(os.getenv("STRESS_TOTAL_DD", "9.0"))  # Varsayılan %9.0
MAX_TOTAL_DRAWDOWN_PCT = TOTAL_DRAWDOWN_LIMIT / 100.0

# ── Zaman Filtreleri ──────────────────────────────────────────────────────────
TRADE_START_HOUR = int(os.getenv("STRESS_START_HOUR", "13"))     # UTC saat - NYSE pre/open çevresi
TRADE_END_HOUR = int(os.getenv("STRESS_END_HOUR", "23"))         # UTC saat - Yeni işlem almama saati
ZORLU_KAPANIS_SAATI = int(os.getenv("STRESS_FORCE_CLOSE_HOUR", "23"))
ALLOW_WEEKEND_HOLDING = os.getenv("STRESS_ALLOW_WEEKEND", "False").lower() in ("true", "1", "yes")

# Hafta sonu tuzak önleme: Perşembe/Cuma yeni işlem engeli
# THURSDAY_CUTOFF_HOUR: Perşembe bu saatten sonra YENİ İŞLEM AÇMA
THURSDAY_CUTOFF_HOUR = int(os.getenv("STRESS_THURSDAY_CUTOFF", "23"))
# BLOCK_FRIDAY_ENTRIES: Cuma günü hiç yeni işlem açma (var olan kapatmaya devam eder)
BLOCK_FRIDAY_ENTRIES = os.getenv("STRESS_BLOCK_FRIDAY", "False").lower() in ("true", "1", "yes")

# ── Pozisyon Boyutlandırma Güvenlik Limitleri ────────────────────────────────
# MAX_LOT_LIMIT: ATR çok daralsa bile lot büyüklüğü bu değeri aşamaz.
#   → Prop firm hesaplarında marjin patlamasını önler.
MAX_LOT_LIMIT = float(os.getenv("STRESS_MAX_LOT", "20.0"))

# ATR_FLOOR: ATR bu eşiğin altına düştüğünde minimum bu değer kullanılır.
#   → Piyasanın aşırı sıkıştığı dönemlerde lot hesabının patlamasını engeller.
#   → TECH/NAS100 için tipik ATR aralığı: 3.0 - 15.0 (15m tf)
ATR_FLOOR = float(os.getenv("STRESS_ATR_FLOOR", "0.0"))

# COST_BENEFIT_MAX_RATIO: (Komisyon + Spread) / TP Kazancı max oranı.
#   → Bu oranı aşan işlemler maliyet açısından verimsiz kabul edilir ve İPTAL edilir.
#   → NAS100/TECH için analiz: komisyon (~%0.02 * 18000 = 3.6 USD/lot/taraf) + spread
#     sabit bir maliyet/TP oranı yaratır (~%48 @ ATR=5, R:R=2). Bu yüzden
#     eşik değerin %60'ın ÜSTÜNDE tutulması normal işlemleri bloklamaz.
#   → Gerçek veto senaryosu: ATR aşırı dar + spread yüksek → oran %80-120%+ olur.
#   → 0.60 → %60 (normal trade geçer, absürd maliyet/TP oranları bloklanır)
COST_BENEFIT_MAX_RATIO = float(os.getenv("STRESS_COST_RATIO", "0.60"))

# ── İşlem Maliyetleri ─────────────────────────────────────────────────────────
SPREAD_PENALTY = float(os.getenv("STRESS_SPREAD_PENALTY", "0.05"))
COMMISSION_RATE = float(os.getenv("STRESS_COMMISSION_RATE", "0.0002"))

# ── MetaTrader 5 Giriş Bilgileri ──────────────────────────────────────────────
MT5_ACCOUNT = int(os.getenv("MT5_ACCOUNT", "5051110235"))
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "@iU2FcLc")
MT5_SERVER = os.getenv("MT5_SERVER", "MetaQuotes-Demo")
MT5_PATH = os.getenv("MT5_PATH", "")  # Boş ise varsayılan MT5 yolu otomatik aranır

# ── Kontrat Parametreleri ─────────────────────────────────────────────────────
CONTRACT_SIZE = float(os.getenv("STRESS_CONTRACT_SIZE", "1.0"))

# Dinamik lot tavanı: notional pozisyon büyüklüğü hesap bakiyesinin bu katını aşamaz.
# Broker seviyesinde ayrıca sembol min/max lot limitleri canlı emir öncesinde uygulanır.
MAX_NOTIONAL_LEVERAGE = float(os.getenv("STRESS_MAX_NOTIONAL_LEVERAGE", "20.0"))
