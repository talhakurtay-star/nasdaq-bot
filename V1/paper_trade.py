"""
paper_trade.py — NASDAQ BOT V2 · Kağıt Trading (Paper Trade)
─────────────────────────────────────────────────────────────
MT5 bağlantısı OLMADAN çalışır. Yahoo Finance'dan canlı QQQ/NAS100 verisi
çeker, her 15 dakikada sinyal üretir, kağıt üstünde işlem açıp kapatır.

Amaç: Challenge'a girmeden önce botun gerçek piyasada ne yaptığını 1 ay
      boyunca izlemek. Log dosyası ve konsol çıktısıyla takip edilir.

Çalıştırma (Google Colab veya terminal):
    python paper_trade.py

Colab'da 7/24 çalıştırmak için:
    - Runtime > Change runtime type > GPU (kapanmayı geciktirir)
    - Aşağıdaki JS trick'i tarayıcı konsoluna yapıştır:
      setInterval(() => document.querySelector('#ok').click(), 60000)
"""

from __future__ import annotations

import json
import logging
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# ── Log kurulumu ──────────────────────────────────────────────────────────────
LOG_DIR = Path("logs")
LOG_DIR.mkdir(exist_ok=True)
LOG_FILE = LOG_DIR / f"paper_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

_fmt = logging.Formatter("%(asctime)s [%(levelname)-8s] %(message)s", "%Y-%m-%d %H:%M:%S")
_ch = logging.StreamHandler(sys.stdout)
_ch.setFormatter(_fmt)
_ch.setLevel(logging.INFO)
_fh = logging.FileHandler(LOG_FILE, encoding="utf-8")
_fh.setFormatter(_fmt)
_fh.setLevel(logging.DEBUG)
logging.basicConfig(level=logging.DEBUG, handlers=[_ch, _fh])
logger = logging.getLogger("PaperTrade")

# ── Proje modülleri ───────────────────────────────────────────────────────────
try:
    import config.settings as cfg
    from data.data_feed    import DataFeed
    from ml.features       import FeatureEngine
    from risk.guardrails   import RiskGuardrails
    from risk.portfolio    import PortfolioManager
    from strategy.engine   import StrategyEngine
    from strategy.voting   import VotingMechanism
except ImportError as e:
    logger.critical("Modül yüklenemedi: %s — V1/ klasöründen çalıştır.", e)
    sys.exit(1)

# ── Sabitler ──────────────────────────────────────────────────────────────────
BAR_DAKIKALARI = frozenset({0, 15, 30, 45})   # Her saat başı 15m barları
DONGU_UYKU     = 20        # saniye — CPU boş tutmak için
GUNLUK_SIFIR   = 0         # UTC saat — günlük DD sıfırlama
PAPER_LOG_FILE = Path("logs") / "paper_trades.json"

# ── Paper Portfolio ───────────────────────────────────────────────────────────

class PaperPortfolio:
    """Gerçek para olmadan işlem takibi."""

    def __init__(self, baslangic: float = 100_000.0) -> None:
        self.baslangic    = baslangic
        self.bakiye       = baslangic
        self.acik_islem   = None   # dict veya None
        self.islem_gecmisi: list  = []
        self.daily_peak   = baslangic
        self.equity       = baslangic
        self.open_position = None  # PortfolioManager uyumu için

    @property
    def daily_peak_equity(self) -> float:
        return self.daily_peak

    def reset_daily_peak(self) -> None:
        self.daily_peak = self.equity

    def update_equity(self, fiyat: float) -> None:
        if self.acik_islem:
            pos = self.acik_islem
            if pos["yon"] == "STRONG_LONG":
                kar = (fiyat - pos["giris"]) * pos["lot"]
            else:
                kar = (pos["giris"] - fiyat) * pos["lot"]
            self.equity = self.bakiye + kar
        else:
            self.equity = self.bakiye
        if self.equity > self.daily_peak:
            self.daily_peak = self.equity

    def ac(self, yon: str, fiyat: float, sl: float, tp: float, lot: float, zaman) -> None:
        self.acik_islem = {
            "yon": yon, "giris": fiyat, "sl": sl, "tp": tp,
            "lot": lot, "acilis": str(zaman)
        }
        self.open_position = True
        logger.info("📋 PAPER İŞLEM AÇILDI | %s @ %.2f | SL=%.2f | TP=%.2f | Lot=%.2f",
                    yon, fiyat, sl, tp, lot)

    def kapat(self, fiyat: float, sebep: str, zaman) -> float:
        if not self.acik_islem:
            return 0.0
        pos = self.acik_islem
        if pos["yon"] == "STRONG_LONG":
            kar = (fiyat - pos["giris"]) * pos["lot"]
        else:
            kar = (pos["giris"] - fiyat) * pos["lot"]
        self.bakiye += kar
        self.equity  = self.bakiye
        kayit = {**pos, "cikis": fiyat, "kar": round(kar, 2),
                 "sebep": sebep, "kapanis": str(zaman)}
        self.islem_gecmisi.append(kayit)
        self.acik_islem  = None
        self.open_position = None
        durum = "✅ KAR" if kar > 0 else "❌ ZARAR"
        logger.info("📋 PAPER İŞLEM KAPANDI [%s] | %s @ %.2f → %.2f | PnL: %+.2f | Bakiye: %.2f",
                    sebep, pos["yon"], pos["giris"], fiyat, kar, self.bakiye)
        self._kaydet()
        return kar

    def kontrol_sl_tp(self, high: float, low: float, zaman) -> str | None:
        """Her bar high/low ile SL/TP tetiklenip tetiklenmediğini kontrol eder."""
        if not self.acik_islem:
            return None
        pos = self.acik_islem
        if pos["yon"] == "STRONG_LONG":
            if low <= pos["sl"]:
                self.kapat(pos["sl"], "SL", zaman)
                return "SL"
            if high >= pos["tp"]:
                self.kapat(pos["tp"], "TP", zaman)
                return "TP"
        else:
            if high >= pos["sl"]:
                self.kapat(pos["sl"], "SL", zaman)
                return "SL"
            if low <= pos["tp"]:
                self.kapat(pos["tp"], "TP", zaman)
                return "TP"
        return None

    def ozet(self) -> dict:
        toplam = len(self.islem_gecmisi)
        kazanan = sum(1 for t in self.islem_gecmisi if t["kar"] > 0)
        toplam_kar = self.bakiye - self.baslangic
        return {
            "bakiye": round(self.bakiye, 2),
            "toplam_kar": round(toplam_kar, 2),
            "toplam_kar_pct": round(toplam_kar / self.baslangic * 100, 2),
            "toplam_islem": toplam,
            "kazanan": kazanan,
            "win_rate": round(kazanan / toplam * 100, 1) if toplam else 0,
        }

    def _kaydet(self) -> None:
        try:
            PAPER_LOG_FILE.write_text(
                json.dumps(self.islem_gecmisi, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
        except Exception:
            pass


# ── Ana Motor ─────────────────────────────────────────────────────────────────

class PaperTradingEngine:

    def __init__(self) -> None:
        self.data_feed       = DataFeed()
        self.feature_engine  = FeatureEngine()
        self.strategy_engine = StrategyEngine()
        self.voting          = VotingMechanism()
        self.guardrails      = RiskGuardrails()
        self.paper           = PaperPortfolio(
            baslangic=float(getattr(cfg, "INITIAL_BALANCE", 100_000.0))
        )
        self._son_bar_dakika = -1
        self._gunluk_gun     = -1
        self._dongu_sayaci   = 0

    def calistir(self) -> None:
        logger.info("═" * 60)
        logger.info("  NASDAQ BOT V2 — PAPER TRADE MODU")
        logger.info("  Başlangıç: %s UTC", datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M"))
        logger.info("  Başlangıç Bakiyesi: $%.2f", self.paper.bakiye)
        logger.info("  Log: %s", LOG_FILE)
        logger.info("═" * 60)

        while True:
            try:
                simdi  = datetime.now(tz=timezone.utc)
                dakika = simdi.minute
                gun    = simdi.day

                # Günlük sıfırlama
                if simdi.hour == GUNLUK_SIFIR and gun != self._gunluk_gun:
                    self.guardrails.reset_daily_drawdown(self.paper)
                    self._gunluk_gun = gun
                    oz = self.paper.ozet()
                    logger.info("── GÜNLÜK ÖZET ──────────────────────────")
                    logger.info("  Bakiye: $%.2f | Kar: %+.2f (%+.2f%%)",
                                oz["bakiye"], oz["toplam_kar"], oz["toplam_kar_pct"])
                    logger.info("  İşlem: %d | Kazanan: %d | WR: %.1f%%",
                                oz["toplam_islem"], oz["kazanan"], oz["win_rate"])
                    logger.info("─────────────────────────────────────────")

                # 15 dakikalık bar tespiti
                if dakika % 15 == 0 and dakika != self._son_bar_dakika:
                    self._son_bar_dakika = dakika
                    self._dongu_sayaci  += 1
                    time.sleep(5)  # Bar kapanmasının tamamlanması için bekle
                    self._bar_isle(simdi)

                time.sleep(DONGU_UYKU)

            except KeyboardInterrupt:
                logger.info("Kullanıcı durdurdu (Ctrl+C).")
                break
            except Exception as e:
                logger.error("Ana döngü hatası: %s", e)
                time.sleep(30)

        # Kapanış özeti
        oz = self.paper.ozet()
        logger.info("═" * 60)
        logger.info("  PAPER TRADE TAMAMLANDI")
        logger.info("  Toplam Döngü    : %d", self._dongu_sayaci)
        logger.info("  Toplam İşlem    : %d", oz["toplam_islem"])
        logger.info("  Bitiş Bakiyesi  : $%.2f", oz["bakiye"])
        logger.info("  Toplam Kar/Zarar: %+.2f (%+.2f%%)", oz["toplam_kar"], oz["toplam_kar_pct"])
        logger.info("  Win Rate        : %.1f%%", oz["win_rate"])
        logger.info("  İşlem geçmişi   : %s", PAPER_LOG_FILE)
        logger.info("═" * 60)

    def _bar_isle(self, simdi: datetime) -> None:
        logger.info("── BAR [%s UTC] (#%d) ──────────────────",
                    simdi.strftime("%H:%M"), self._dongu_sayaci)

        # 1. Canlı veri çek
        try:
            df = self.data_feed.download_historical_data(force_refresh=True)
            if df is None or len(df) < 50:
                logger.warning("Yetersiz veri — atlanıyor.")
                return
        except Exception as e:
            logger.error("Veri indirme hatası: %s", e)
            return

        # 2. Açık işlem SL/TP kontrolü
        if self.paper.acik_islem:
            son_bar = df.iloc[-1]
            self.paper.update_equity(float(son_bar.get("Close", son_bar.iloc[-1])))
            tetik = self.paper.kontrol_sl_tp(
                float(son_bar.get("High", son_bar.iloc[1])),
                float(son_bar.get("Low",  son_bar.iloc[2])),
                simdi,
            )
            if tetik:
                return  # İşlem kapandı, bu barda yeni işlem açma

        # 3. İndikatörler hesapla
        try:
            df_ind = self.feature_engine.calculate_indicators(df)
        except Exception as e:
            logger.error("İndikatör hatası: %s", e)
            return

        # 4. Equity güncelle + guardrail
        son_fiyat = float(df_ind["Close"].iloc[-1])
        self.paper.update_equity(son_fiyat)

        if self.guardrails.check_drawdown_limits(self.paper):
            logger.warning("🚨 GUARDRAIL — trading durdu.")
            return

        # 5. Açık işlem varsa yeni sinyal arama
        if self.paper.acik_islem:
            logger.debug("Açık işlem var — yeni sinyal yok.")
            return

        # 6. Zaman filtresi
        if not self.guardrails.is_trading_allowed(self.paper, simdi):
            logger.debug("İşlem saati dışı veya haber filtresi.")
            return

        # 7. Temel sinyal
        try:
            current_idx = len(df_ind) - 1
            base_signal = self.strategy_engine.generate_base_signal(df_ind, current_idx)
        except Exception as e:
            logger.error("Strateji hatası: %s", e)
            return

        logger.info("Temel sinyal: %s", base_signal)
        if base_signal == "HOLD":
            return

        # 8. XGBoost vizesi
        try:
            final_signal = self.voting.decide_trade(
                df=df_ind,
                current_index=current_idx,
                base_signal=base_signal,
                feature_engine=self.feature_engine,
            )
        except Exception as e:
            logger.error("XGBoost hatası: %s", e)
            return

        logger.info("Final sinyal: %s", final_signal)
        if final_signal == "HOLD":
            return

        # 9. ATR lot/SL/TP hesapla ve kağıt işlem aç
        try:
            atr_col = next((c for c in ("ATR", "atr", "ATR_14") if c in df_ind.columns), None)
            if not atr_col:
                logger.warning("ATR kolonu bulunamadı.")
                return

            atr   = float(df_ind[atr_col].iloc[-1])
            sl_d  = atr * float(getattr(cfg, "ATR_MULTIPLIER", 2.5))
            risk  = self.paper.bakiye * float(getattr(cfg, "RISK_PER_TRADE", 0.012))
            lot   = risk / sl_d if sl_d > 0 else 0
            rr    = float(getattr(cfg, "REWARD_RISK_RATIO", 2.5))

            if final_signal == "STRONG_LONG":
                sl = son_fiyat - sl_d
                tp = son_fiyat + sl_d * rr
            else:
                sl = son_fiyat + sl_d
                tp = son_fiyat - sl_d * rr

            self.paper.ac(final_signal, son_fiyat, sl, tp, lot, simdi)

        except Exception as e:
            logger.error("İşlem açma hatası: %s", e)


# ── Çalıştır ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    engine = PaperTradingEngine()
    engine.calistir()
