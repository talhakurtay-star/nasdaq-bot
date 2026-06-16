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

# CSV veri kaynağı ayarları
USE_CSV_DATA    = True
CSV_DIR         = os.path.join(str(BASE_DIR), "csv")
CSV_FILE_NAME   = "nasdaq_15m.csv"

# Dizinlerin varlığını kontrol et
os.makedirs(CACHE_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(CSV_DIR,   exist_ok=True)

# ── Zaman ve Sembol Ayarları ──────────────────────────────────────────────────
SYMBOL     = os.getenv("STRESS_SYMBOL",     "QQQ")
MT5_SYMBOL = os.getenv("STRESS_MT5_SYMBOL", "NAS100")

VIX_SYMBOL = os.getenv("STRESS_VIX_SYMBOL", "^VIX")
SPY_SYMBOL = os.getenv("STRESS_SPY_SYMBOL", "SPY")

# NASDAQ 100'ün en likit 15 hissesi (ICMarkets CFD isimleri)
# yfinance sembolü → MT5 CFD adı eşleşmesi
UNIVERSE: list[str] = [
    # ── Büyük Teknoloji (20) ──────────────────────────────────────────────
    "AAPL",  "MSFT",  "GOOGL", "META",  "ORCL",
    "CRM",   "ADBE",  "NOW",   "INTU",  "PANW",
    "ZS",    "CRWD",  "DDOG",  "NET",   "SNOW",
    "PLTR",  "FTNT",  "OKTA",  "MDB",   "SPLK",
    # ── Yarı İletken (15) ────────────────────────────────────────────────
    "NVDA",  "AMD",   "QCOM",  "AVGO",  "TXN",
    "AMAT",  "LRCX",  "KLAC",  "MU",   "MRVL",
    "ON",    "MPWR",  "SMCI",  "WOLF",  "SLAB",
    # ── Tüketici / E-ticaret (10) ────────────────────────────────────────
    "AMZN",  "TSLA",  "NFLX",  "COST",  "SBUX",
    "BKNG",  "ABNB",  "UBER",  "LYFT",  "DASH",
    # ── Sağlık / Biyoteknoloji (10) ──────────────────────────────────────
    "ISRG",  "VRTX",  "REGN",  "MRNA",  "ILMN",
    "IDXX",  "DXCM",  "PODD",  "HALO",  "ALGN",
    # ── Fintech / Finans (10) ────────────────────────────────────────────
    "PYPL",  "SQ",    "COIN",  "V",     "MA",
    "AXP",   "SOFI",  "AFRM",  "HOOD",  "BILL",
    # ── İletişim / Medya (8) ─────────────────────────────────────────────
    "SPOT",  "PINS",  "SNAP",  "ROKU",  "TTD",
    "MGNI",  "PARA",  "WBD",
    # ── Sanayi / Savunma (8) ─────────────────────────────────────────────
    "HON",   "GE",    "CAT",   "DE",
    "RTX",   "LMT",   "NOC",   "BA",
    # ── Enerji (7) ───────────────────────────────────────────────────────
    "XOM",   "CVX",   "COP",   "SLB",
    "HAL",   "BKR",   "OXY",
    # ── Hammadde / Madencilik (5) ────────────────────────────────────────
    "FCX",   "AA",    "NEM",   "GOLD",  "CLF",
    # ── Defansif ETF (7) ─────────────────────────────────────────────────
    "QQQ",   "SPY",   "IWM",   "XLF",
    "XLE",   "XLV",   "GLD",
]

# Aynı anda maksimum açık pozisyon sayısı
MAX_OPEN_POSITIONS = int(os.getenv("STRESS_MAX_POSITIONS", "3"))

# Sektörel korelasyon koruması: aynı sektörden max pozisyon sayısı
MAX_SECTOR_POSITIONS = int(os.getenv("STRESS_MAX_SECTOR_POS", "2"))

# Sektör → Sembol haritası (100 sembol, 10 sektör)
SECTOR_MAP: dict[str, str] = {
    # Büyük Teknoloji
    "AAPL": "TECH", "MSFT": "TECH", "GOOGL": "TECH", "META": "TECH", "ORCL": "TECH",
    "CRM":  "TECH", "ADBE": "TECH", "NOW":   "TECH", "INTU": "TECH", "PANW": "TECH",
    "ZS":   "TECH", "CRWD": "TECH", "DDOG":  "TECH", "NET":  "TECH", "SNOW": "TECH",
    "PLTR": "TECH", "FTNT": "TECH", "OKTA":  "TECH", "MDB":  "TECH", "SPLK": "TECH",
    # Yarı İletken
    "NVDA": "SEMI", "AMD":  "SEMI", "QCOM": "SEMI", "AVGO": "SEMI", "TXN":  "SEMI",
    "AMAT": "SEMI", "LRCX": "SEMI", "KLAC": "SEMI", "MU":   "SEMI", "MRVL": "SEMI",
    "ON":   "SEMI", "MPWR": "SEMI", "SMCI": "SEMI", "WOLF": "SEMI", "SLAB": "SEMI",
    # Tüketici / E-ticaret
    "AMZN": "CONSUMER", "TSLA": "CONSUMER", "NFLX": "CONSUMER", "COST": "CONSUMER",
    "SBUX": "CONSUMER", "BKNG": "CONSUMER", "ABNB": "CONSUMER", "UBER": "CONSUMER",
    "LYFT": "CONSUMER", "DASH": "CONSUMER",
    # Sağlık / Biyoteknoloji
    "ISRG": "HEALTH", "VRTX": "HEALTH", "REGN": "HEALTH", "MRNA": "HEALTH",
    "ILMN": "HEALTH", "IDXX": "HEALTH", "DXCM": "HEALTH", "PODD": "HEALTH",
    "HALO": "HEALTH", "ALGN": "HEALTH",
    # Fintech / Finans
    "PYPL": "FINTECH", "SQ":   "FINTECH", "COIN": "FINTECH", "V":    "FINTECH",
    "MA":   "FINTECH", "AXP":  "FINTECH", "SOFI": "FINTECH", "AFRM": "FINTECH",
    "HOOD": "FINTECH", "BILL": "FINTECH",
    # İletişim / Medya
    "SPOT": "MEDIA", "PINS": "MEDIA", "SNAP": "MEDIA", "ROKU": "MEDIA",
    "TTD":  "MEDIA", "MGNI": "MEDIA", "PARA": "MEDIA", "WBD":  "MEDIA",
    # Sanayi / Savunma
    "HON": "INDUSTRIAL", "GE":  "INDUSTRIAL", "CAT": "INDUSTRIAL", "DE":  "INDUSTRIAL",
    "RTX": "INDUSTRIAL", "LMT": "INDUSTRIAL", "NOC": "INDUSTRIAL", "BA":  "INDUSTRIAL",
    # Enerji
    "XOM": "ENERGY", "CVX": "ENERGY", "COP": "ENERGY", "SLB": "ENERGY",
    "HAL": "ENERGY", "BKR": "ENERGY", "OXY": "ENERGY",
    # Hammadde / Madencilik
    "FCX": "MATERIALS", "AA": "MATERIALS", "NEM": "MATERIALS",
    "GOLD": "MATERIALS", "CLF": "MATERIALS",
    # Defansif ETF
    "QQQ": "ETF_TECH", "SPY": "ETF_BROAD", "IWM": "ETF_SMALL",
    "XLF": "ETF_FIN",  "XLE": "ETF_ENERGY","XLV": "ETF_HEALTH",
    "GLD": "ETF_GOLD",
}

WATCHLIST = UNIVERSE + [SPY_SYMBOL, VIX_SYMBOL]

# TIMEFRAME: Grafik periyodu ("15m", "1h", "5m")
TIMEFRAME = "15m"

# BACKTEST_DAYS: Geriye dönük çekilecek veri gün sayısı
BACKTEST_DAYS = int(os.getenv("BACKTEST_DAYS", "360"))

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
XGB_PROBABILITY_THRESHOLD = float(os.getenv("STRESS_XGB_THRESHOLD", "0.0"))   # 0.0: XGB devre dışı — model prob < eşik → tüm trade veto olur

# ── Prop Firm Profit-Lock ──────────────────────────────────────────────────────
# Challenge hedefine yaklaşıldığında risk otomatik azalır — challenge pasını garantiler.
# Sermayeli botta KAPALI (False), prop firm botta AÇIK (True).
PROFIT_LOCK_ENABLED     = os.getenv("STRESS_PROFIT_LOCK", "False").lower() in ("true", "1", "yes")
PROFIT_LOCK_WARN_PCT    = float(os.getenv("STRESS_PROFIT_LOCK_WARN",  "0.06"))   # %6 kardan itibaren 0.75x
PROFIT_LOCK_LOCK_PCT    = float(os.getenv("STRESS_PROFIT_LOCK_LOCK",  "0.075"))  # %7.5 kardan itibaren 0.40x

# ── Funded Hesap Modu ─────────────────────────────────────────────────────────
# Challenge geçtikten sonra funded hesapta çalışırken True yap.
# Funded modda profit-lock devre dışı, risk tam güçte, sadece DD koruma aktif.
# Kullanım: settings.py'de FUNDED_MODE = True, veya ortam değişkeni set et.
FUNDED_MODE = os.getenv("STRESS_FUNDED_MODE", "False").lower() in ("true", "1", "yes")

# ── Portföy ve Sermaye Ayarları ──────────────────────────────────────────────
INITIAL_BALANCE = float(os.getenv("STRESS_INITIAL_BALANCE", "100000.0"))
REWARD_RISK_RATIO     = float(os.getenv("STRESS_RR",          "3.7"))
PARTIAL_CLOSE_R       = float(os.getenv("STRESS_PARTIAL_R",   "1.0"))  # 1R'da kısmi kâr al
PARTIAL_CLOSE_FRAC    = float(os.getenv("STRESS_PARTIAL_FRAC", "0.25")) # kaç % kapatılacak (0.25 = %25 → kalan %75 TP'ye koşar)
TRAILING_ACTIVATION_R = float(os.getenv("STRESS_TRAIL_R",     "1.0"))  # 1R sonrası trailing başlat
TRAILING_ATR_MULT     = float(os.getenv("STRESS_TRAIL_ATR",   "3.5"))  # > ATR_MULT=2.5 → breakeven'den sonra kalan %50 TP'ye kadar serbest çalışır
ATR_MULTIPLIER        = float(os.getenv("STRESS_ATR_MULT",    "2.5"))  # Geniş SL: erken tetiklenmeyi azaltır, WR artar

# ── Risk Yönetimi ve Prop Firm Limitleri ──────────────────────────────────────
# RISK_PER_TRADE: İşlem başına risk yüzdesi (Örn: 1.00 -> %1)
RISK_PER_TRADE_PCT = float(os.getenv("STRESS_RISK_PCT", "3.0"))   # 3.0%: MaxDD=%22.5, +%195/yıl (prop firm dışı kullanım)
RISK_PER_TRADE = RISK_PER_TRADE_PCT / 100.0                     # Lojik işlemlerde kullanılan decimal değer

# DAILY_DRAWDOWN_LIMIT: Günlük maksimum kayıp limiti (% cinsinden)
DAILY_DRAWDOWN_LIMIT = float(os.getenv("STRESS_DAILY_DD", "4.0"))  # %4.0: Funding Pips %5 daily limitinin altında güvenli tampon
MAX_DAILY_DRAWDOWN_PCT = DAILY_DRAWDOWN_LIMIT / 100.0

# TOTAL_DRAWDOWN_LIMIT: Hesap genelinde maksimum kayıp limiti (% cinsinden)
TOTAL_DRAWDOWN_LIMIT = float(os.getenv("STRESS_TOTAL_DD", "9.0"))  # %9.0: Funding Pips %10 max loss altında güvenli tampon
MAX_TOTAL_DRAWDOWN_PCT = TOTAL_DRAWDOWN_LIMIT / 100.0

# ── Zaman Filtreleri ──────────────────────────────────────────────────────────
TRADE_START_HOUR = int(os.getenv("STRESS_START_HOUR", "1"))      # UTC: tüm gün (NAS100 CFD 24h)
TRADE_END_HOUR   = int(os.getenv("STRESS_END_HOUR",   "23"))     # UTC: kapanış öncesi yeni işlem yok
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
MAX_LOT_LIMIT = float(os.getenv("STRESS_MAX_LOT", "1000.0"))  # Raised: 2% risk @ 21000 NAS100 ATR=30 → ~33 lots

# ATR_FLOOR: ATR bu eşiğin altına düştüğünde minimum bu değer kullanılır.
#   → Piyasanın aşırı sıkıştığı dönemlerde lot hesabının patlamasını engeller.
ATR_FLOOR = float(os.getenv("STRESS_ATR_FLOOR", "0.0"))

# ATR_MIN_ENTRY: Giriş için minimum ATR. Bu değerin altındaysa trade bloklama.
# NAS100 15m için tipik ATR: 20-80 puan. Çok düşük ATR = sıkışık piyasa = yüksek slippage.
ATR_MIN_ENTRY = float(os.getenv("STRESS_ATR_MIN_ENTRY", "20.0"))
ATR_MIN_PCT   = float(os.getenv("STRESS_ATR_MIN_PCT",   "0.0005"))  # 0.05% of price

# COST_BENEFIT_MAX_RATIO: (Komisyon + Spread) / TP Kazancı max oranı.
#   → Bu oranı aşan işlemler maliyet açısından verimsiz kabul edilir ve İPTAL edilir.
#   → NAS100/TECH için analiz: komisyon (~%0.02 * 18000 = 3.6 USD/lot/taraf) + spread
#     sabit bir maliyet/TP oranı yaratır (~%48 @ ATR=5, R:R=2). Bu yüzden
#     eşik değerin %60'ın ÜSTÜNDE tutulması normal işlemleri bloklamaz.
#   → Gerçek veto senaryosu: ATR aşırı dar + spread yüksek → oran %80-120%+ olur.
#   → 0.60 → %60 (normal trade geçer, absürd maliyet/TP oranları bloklanır)
COST_BENEFIT_MAX_RATIO = float(os.getenv("STRESS_COST_RATIO", "0.75"))

# ── İşlem Maliyetleri ─────────────────────────────────────────────────────────
# NAS100 gerçek maliyet modeli:
# ICMarkets raw spread: USD 1.50/100k notional per side → 0.000015 per side
# Spread: NAS100 ortalama 1.5 puan (index CFD tipik)
SPREAD_PENALTY  = float(os.getenv("STRESS_SPREAD_PENALTY",   "1.5"))     # NAS100 puan cinsinden spread
COMMISSION_RATE = float(os.getenv("STRESS_COMMISSION_RATE",  "0.000015")) # ICMarkets raw rate

# ── MetaTrader 5 Giriş Bilgileri ──────────────────────────────────────────────
# GÜVENLİK: Varsayılan değer yok. Ortam değişkeni set edilmezse başlangıçta hata verir.
# Kullanım: export MT5_ACCOUNT=12345678 MT5_PASSWORD=xxx MT5_SERVER=ICMarkets-Demo
_mt5_account_str = os.getenv("MT5_ACCOUNT", "")
MT5_ACCOUNT  = int(_mt5_account_str) if _mt5_account_str else 0
MT5_PASSWORD = os.getenv("MT5_PASSWORD", "")
MT5_SERVER   = os.getenv("MT5_SERVER",   "MetaQuotes-Demo")
MT5_PATH     = os.getenv("MT5_PATH",     "")

if not MT5_PASSWORD and os.getenv("STRESS_TEST_MODE", "1") != "1":
    import warnings
    warnings.warn(
        "MT5_PASSWORD ortam değişkeni set edilmemiş. "
        "Canlı trading için: export MT5_PASSWORD=sifreniz",
        stacklevel=2,
    )

# ── Kontrat Parametreleri ─────────────────────────────────────────────────────
CONTRACT_SIZE = float(os.getenv("STRESS_CONTRACT_SIZE", "1.0"))

# Dinamik lot tavanı: notional pozisyon büyüklüğü hesap bakiyesinin bu katını aşamaz.
# Broker seviyesinde ayrıca sembol min/max lot limitleri canlı emir öncesinde uygulanır.
MAX_NOTIONAL_LEVERAGE = float(os.getenv("STRESS_MAX_NOTIONAL_LEVERAGE", "20.0"))
