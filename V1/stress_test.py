"""
╔══════════════════════════════════════════════════════════════════════════════╗
║        NASDAQ BOT V2 — KANTİTATİF STRES TESTİ ODASI  v2.0                 ║
║        Chief Risk Officer · Quant Research Division                          ║
║        Mimari: Fiziksel Dosya Yamama + İzole Subprocess Yürütme             ║
║        PEP8 Uyumlu  |  Look-Ahead Bias: SIFIR  |  Modül Kirlenmesi: SIFIR  ║
╚══════════════════════════════════════════════════════════════════════════════╝

ÇALIŞMA PRENSİBİ
────────────────
Klasik importlib.reload() yaklaşımının "from X import Y" kirlenmesini çözmek
için bu betik tamamen izole bir subprocess mimarisi kullanır:

  1. config/settings.py dosyasını YEDEKLE  (settings.py.stress_bak)
  2. Her senaryo için:
     a. settings.py içindeki parametreleri fiziksel olarak YAMA (regex + File I/O)
     b. subprocess.run("python ml/trainer.py")  → temiz Python sürecinde eğit
     c. subprocess.run("python backtest.py")    → temiz Python sürecinde simüle
     d. backtest.py stdout çıktısını regex ile ayrıştır → temp_metrics.json yaz
     e. temp_metrics.json oku → senaryo hanesine kaydet
  3. settings.py'yi YEDEKTEN GERİ YÜKLE
  4. ANSI renkli KANTİTATİF STRES TESTİ MATRİSİ'ni bas

Her subprocess kendi temiz Python yorumlayıcısında başlar; önceki senaryonun
import durumu veya önbelleğe alınmış değerleri bir sonrakini asla kirletemez.

ÇALIŞTIRMA
──────────
    python stres_testi_odasi.py

GEREKLİLİK
──────────
    - Projenin kök dizininden çalıştırılmalıdır (backtest.py ile aynı seviye).
    - config/settings.py dosyasına yazma izni gereklidir.
    - ml/trainer.py ve backtest.py doğrudan `python <dosya>` ile çalışabilmeli.
"""

from __future__ import annotations

import json
import logging
import math
import os
import re
import shutil
import subprocess
import sys
sys.stdout.reconfigure(encoding='utf-8')
sys.stderr.reconfigure(encoding='utf-8')
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

# ──────────────────────────────────────────────────────────────────────────────
# LOGLAMA — Renkli, saat damgalı kurumsal format
# ──────────────────────────────────────────────────────────────────────────────

_ANSI_RESET  = "\033[0m"
_ANSI_BOLD   = "\033[1m"
_ANSI_CYAN   = "\033[96m"
_ANSI_YELLOW = "\033[93m"
_ANSI_GREEN  = "\033[92m"
_ANSI_RED    = "\033[91m"
_ANSI_GREY   = "\033[90m"
_ANSI_WHITE  = "\033[97m"
_ANSI_BLUE   = "\033[94m"


class _RenkliFormatter(logging.Formatter):
    """Seviyeye göre renklendirilmiş konsol log biçimleyicisi."""

    _SEVIYE_RENK: dict[int, str] = {
        logging.DEBUG:    _ANSI_GREY,
        logging.INFO:     _ANSI_CYAN,
        logging.WARNING:  _ANSI_YELLOW,
        logging.ERROR:    _ANSI_RED,
        logging.CRITICAL: _ANSI_RED + _ANSI_BOLD,
    }

    def format(self, record: logging.LogRecord) -> str:
        renk = self._SEVIYE_RENK.get(record.levelno, "")
        seviye = f"{renk}{record.levelname:<8}{_ANSI_RESET}"
        zaman  = f"{_ANSI_GREY}{self.formatTime(record, '%H:%M:%S')}{_ANSI_RESET}"
        mesaj  = record.getMessage()
        return f"{zaman}  [{seviye}]  {mesaj}"


def _logger_kur(ad: str) -> logging.Logger:
    lg = logging.getLogger(ad)
    if not lg.handlers:
        sh = logging.StreamHandler(sys.stdout)
        sh.setFormatter(_RenkliFormatter())
        lg.addHandler(sh)
        lg.propagate = False
    lg.setLevel(logging.DEBUG)
    return lg


logger = _logger_kur("StresOdasi")

# ──────────────────────────────────────────────────────────────────────────────
# PROJE YOL SABİTLERİ
# ──────────────────────────────────────────────────────────────────────────────

PROJE_KOK       = Path(__file__).resolve().parent
SETTINGS_DOSYA  = PROJE_KOK / "config" / "settings.py"
YEDEK_DOSYA     = PROJE_KOK / "config" / "settings.py.stress_bak"
TEMP_METRIK     = PROJE_KOK / "temp_metrics.json"
TRAINER_BETIK   = PROJE_KOK / "ml" / "trainer.py"
BACKTEST_BETIK  = PROJE_KOK / "backtest.py"

# Subprocess timeout sınırları (saniye)
TRAINER_TIMEOUT  = 60 * 30   # 30 dakika — uzun backtest dönemleri için
BACKTEST_TIMEOUT = 60 * 60   # 60 dakika — 700 günlük 1h bar yükü için

# ──────────────────────────────────────────────────────────────────────────────
# SENARYO VERİ YAPISI
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class Senaryo:
    """Tek bir stres testi senaryosunu temsil eden değer nesnesi."""

    ad:                        str
    etiket:                    str    # Rapor tablosundaki kısa ad
    timeframe:                 str
    backtest_days:             int
    xgb_probability_threshold: float
    reward_risk_ratio:         float
    aciklama:                  str = ""

    # ── Senaryo tamamlanınca doldurulur ──
    sonuclar:    dict[str, Any] = field(default_factory=dict)
    hata:        str | None     = None
    sure_saniye: float          = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# SENARYO TANIMLAMALARı
# ──────────────────────────────────────────────────────────────────────────────

SENARYOLAR: list[Senaryo] = [
    Senaryo(
        ad="SENARYO 1: 2 Yıllık Makro Stres",
        etiket="MAKRO",
        timeframe="1h",
        backtest_days=700,
        xgb_probability_threshold=0.65,
        reward_risk_ratio=2.0,
        aciklama="Uzun vadeli piyasa döngüsü — boğa/ayı geçişleri dahil",
    ),
    Senaryo(
        ad="SENARYO 2: Keskin Nişancı Scalp",
        etiket="SCALP",
        timeframe="15m",
        backtest_days=60,
        xgb_probability_threshold=0.70,
        reward_risk_ratio=3.0,
        aciklama="Yüksek eşikli, yüksek ödüllü — düşük frekans scalp",
    ),
    Senaryo(
        ad="SENARYO 3: Testere Piyasası Tuzağı",
        etiket="TESTERE",
        timeframe="15m",
        backtest_days=60,
        xgb_probability_threshold=0.60,
        reward_risk_ratio=1.5,
        aciklama="Whipsaw ortamı — düşük eşik, dar RR — sistemi sınırla",
    ),
]

# ──────────────────────────────────────────────────────────────────────────────
# FİZİKSEL DOSYA YAMAMA MOToru
# ──────────────────────────────────────────────────────────────────────────────

class DosyaYamaci:
    """
    config/settings.py dosyasını fiziksel olarak yamas — import kirlenmesi SIFIR.

    Desteklenen parametre şablonları
    ─────────────────────────────────
      TIMEFRAME                 = "1h"        → string (çift/tek tırnak)
      BACKTEST_DAYS             = 700         → int
      XGB_PROBABILITY_THRESHOLD = 0.65        → float
      REWARD_RISK_RATIO         = 2.0         → float

    Regex satırın geri kalanını (yorum dahil) korur;
    yalnızca değer kısmı güncellenir.
    """

    # (parametre_adi, değer_tipi)
    _PARAMETRE_TIPLERI: dict[str, str] = {
        "TIMEFRAME":                 "str",
        "BACKTEST_DAYS":             "int",
        "XGB_PROBABILITY_THRESHOLD": "float",
        "REWARD_RISK_RATIO":         "float",
    }

    @staticmethod
    def _deger_str(tip: str, deger: Any) -> str:
        """Python kaynak koduna yazılacak değer gösterimini üretir."""
        if tip == "str":
            return f'"{deger}"'
        if tip == "int":
            return str(int(deger))
        # float → en az bir ondalık basamak garanti et
        return f"{float(deger):.4f}".rstrip("0").rstrip(".") + (
            "" if "." in f"{float(deger):.4f}".rstrip("0") else ".0"
        )

    @classmethod
    def yama_uygula(cls, senaryo: Senaryo) -> None:
        """
        settings.py dosyasını senaryo parametrelerine göre fiziksel olarak günceller.
        Hata durumunda orijinal içerik korunur (atomic write pattern).
        """
        logger.info("📝 settings.py fiziksel yamalaması başlıyor…")

        if not SETTINGS_DOSYA.exists():
            raise FileNotFoundError(
                f"settings.py bulunamadı: {SETTINGS_DOSYA}\n"
                "Betiği projenin kök dizininden çalıştırdığınızdan emin olun."
            )

        # Yeni değerler → dict
        yeni_degerler: dict[str, Any] = {
            "TIMEFRAME":                 senaryo.timeframe,
            "BACKTEST_DAYS":             senaryo.backtest_days,
            "XGB_PROBABILITY_THRESHOLD": senaryo.xgb_probability_threshold,
            "REWARD_RISK_RATIO":         senaryo.reward_risk_ratio,
        }

        icerik = SETTINGS_DOSYA.read_text(encoding="utf-8")
        yeni_icerik = icerik

        for param, deger in yeni_degerler.items():
            tip = cls._PARAMETRE_TIPLERI[param]
            deger_goster = cls._deger_str(tip, deger)

            if tip == "str":
                # String: PARAM = "eski"  ya da  PARAM = 'eski'
                desen = rf"^({re.escape(param)}\s*=\s*)[\"'][^\"']*[\"']"
                yeni  = rf"\g<1>{deger_goster}"
            else:
                # Sayısal: PARAM = 123  ya da  PARAM = 1.23
                desen = rf"^({re.escape(param)}\s*=\s*)[\d.+-]+"
                yeni  = rf"\g<1>{deger_goster}"

            yeni_icerik_deneme = re.sub(
                desen, yeni, yeni_icerik, flags=re.MULTILINE
            )

            if yeni_icerik_deneme == yeni_icerik:
                logger.warning(
                    "   ⚠ Parametre bulunamadı veya değişmedi: %s "
                    "(settings.py'de bu isimde bir satır var mı?)", param
                )
            else:
                logger.debug(
                    "   ✎ %-30s → %s", param, deger_goster
                )

            yeni_icerik = yeni_icerik_deneme

        # Atomic write: geçici dosya → rename
        tmp_yol = SETTINGS_DOSYA.with_suffix(".py.tmp_stress")
        try:
            tmp_yol.write_text(yeni_icerik, encoding="utf-8")
            tmp_yol.replace(SETTINGS_DOSYA)
        except Exception:
            tmp_yol.unlink(missing_ok=True)
            raise

        logger.info(
            "   ✔ TIMEFRAME=%-4s | DAYS=%-4d | XGB_THR=%.2f | RR=%.1f",
            senaryo.timeframe,
            senaryo.backtest_days,
            senaryo.xgb_probability_threshold,
            senaryo.reward_risk_ratio,
        )

    @staticmethod
    def yedek_al() -> None:
        """settings.py'nin yedeğini alır. Yedek zaten varsa üzerine yazar."""
        shutil.copy2(SETTINGS_DOSYA, YEDEK_DOSYA)
        logger.info("💾 settings.py yedeği alındı → %s", YEDEK_DOSYA.name)

    @staticmethod
    def yedekten_geri_yukle() -> None:
        """settings.py'yi yedekten geri yükler ve yedeği siler."""
        if YEDEK_DOSYA.exists():
            shutil.copy2(YEDEK_DOSYA, SETTINGS_DOSYA)
            YEDEK_DOSYA.unlink(missing_ok=True)
            logger.info("♻️  settings.py orijinal haline geri yüklendi.")
        else:
            logger.warning("Yedek dosya bulunamadı: %s", YEDEK_DOSYA)

# ──────────────────────────────────────────────────────────────────────────────
# SUBPROCESS YÜRÜTÜCÜsü
# ──────────────────────────────────────────────────────────────────────────────

class SubprocessYurutucu:
    """
    ml/trainer.py ve backtest.py'yi temiz, izole Python süreçleri olarak çalıştırır.
    stdout + stderr yakalanır; timeout ve hata yönetimi eksiksizdir.
    """

    def __init__(self, python_yorumlayici: str = sys.executable) -> None:
        self._py = python_yorumlayici

    def calistir(
        self,
        betik: Path,
        aciklama: str,
        timeout: int,
        ekstra_env: dict[str, str] | None = None,
    ) -> tuple[str, str, int]:
        """
        Verilen Python betiğini subprocess olarak yürütür.

        Dönüş
        ──────
        (stdout_metni, stderr_metni, return_code) üçlüsü.
        """
        logger.info("🚀 Subprocess başlatılıyor: %s", aciklama)
        logger.debug("   Komut: %s %s", self._py, betik)

        env = os.environ.copy()
        env["STRESS_TEST_MODE"] = "1"   # MT5 şifre uyarısını bastır
        if ekstra_env:
            env.update(ekstra_env)

        # PYTHONPATH'e proje kökünü ekle — modül import'ları için kritik
        mevcut_pythonpath = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = (
            f"{PROJE_KOK}{os.pathsep}{mevcut_pythonpath}"
            if mevcut_pythonpath else str(PROJE_KOK)
        )

        try:
            sonuc = subprocess.run(
                [self._py, str(betik)],
                cwd=str(PROJE_KOK),
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                timeout=timeout,
                env=env,
            )
        except subprocess.TimeoutExpired:
            hata_msg = (
                f"Subprocess zaman aşımına uğradı ({timeout}s): {betik.name}"
            )
            logger.error("⏱ %s", hata_msg)
            raise RuntimeError(hata_msg)
        except Exception as exc:
            logger.error("❌ Subprocess başlatma hatası: %s", exc)
            raise

        # Çıktıyı loglara aktar (ilk 50 satır — debug seviyesinde)
        for satir in sonuc.stdout.splitlines()[:50]:
            logger.debug("   [stdout] %s", satir)
        if sonuc.stderr.strip():
            for satir in sonuc.stderr.splitlines()[:20]:
                logger.debug("   [stderr] %s", satir)

        if sonuc.returncode != 0:
            logger.warning(
                "   ⚠ Subprocess sıfır dışı çıkış kodu: %d  (%s)",
                sonuc.returncode,
                betik.name,
            )
        else:
            logger.info("   ✔ %s tamamlandı (kod: 0).", aciklama)

        return sonuc.stdout, sonuc.stderr, sonuc.returncode

# ──────────────────────────────────────────────────────────────────────────────
# METRİK AYRIŞTIRICISI
# ──────────────────────────────────────────────────────────────────────────────

class MetrikAyristirici:
    """
    backtest.py stdout çıktısını veya temp_metrics.json dosyasını okuyarak
    standart quant metriklerini çıkarır.

    Öncelik sırası:
      1. temp_metrics.json dosyası (backtest.py tarafından yazılmışsa)
      2. stdout regex ayrıştırması (genel — birden fazla isimlendirme kalıbını karşılar)
      3. Sıfır değerleri ile boş dict (son çare)
    """

    # ── Regex kalıpları: (metrik_anahtarı, desen_listesi) ─────────────────────
    # Her desen, yakalanabilir tek bir sayısal gruba sahip olmalıdır.
    _DESENLER: dict[str, list[str]] = {
        "net_pnl": [
            r"[Nn]et\s*[Pp][Nn][Ll]\s*[:\$=]\s*\$?\s*([-+]?[\d,]+\.?\d*)",
            r"[Tt]otal\s*[Pp]rofit\s*[:\$=]\s*\$?\s*([-+]?[\d,]+\.?\d*)",
            r"[Ff]inal\s*[Pp][Nn][Ll]\s*[:\$=]\s*\$?\s*([-+]?[\d,]+\.?\d*)",
            r"[Gg]elir\s*[:\$=]\s*\$?\s*([-+]?[\d,]+\.?\d*)",
            r"[Kk]ar\s*/\s*[Zz]arar\s*[:\$=]\s*\$?\s*([-+]?[\d,]+\.?\d*)",
        ],
        "net_pnl_pct": [
            r"[Nn]et\s*[Pp][Nn][Ll]\s*%\s*[:\$=]\s*([-+]?[\d.]+)\s*%",
            r"[Rr]eturn\s*%\s*[:\$=]\s*([-+]?[\d.]+)\s*%",
            r"[Tt]oplam\s*[Gg]etiri\s*[:\$=]\s*([-+]?[\d.]+)\s*%",
        ],
        "maks_drawdown_pct": [
            r"[Mm]ax(?:imum)?\s*[Dd]rawdown\s*[:\$=]\s*([-+]?[\d.]+)\s*%",
            r"[Mm][Aa][Xx]\s*DD\s*[:\$=]\s*([-+]?[\d.]+)\s*%",
            r"[Mm]aks(?:imum)?\s*[Dd]rawdown\s*[:\$=]\s*([-+]?[\d.]+)\s*%",
            r"[Dd]rawdown\s*[:\$=]\s*([-+]?[\d.]+)\s*%",
        ],
        "win_rate_pct": [
            r"[Ww]in\s*[Rr]ate\s*[:\$=]\s*([\d.]+)\s*%",
            r"[Ww]in\s*[Rr]atio\s*[:\$=]\s*([\d.]+)\s*%",
            r"[Kk]azanma\s*[Oo]ranı\s*[:\$=]\s*([\d.]+)\s*%",
            r"[Ww]inrate\s*[:\$=]\s*([\d.]+)\s*%",
        ],
        "sharpe": [
            r"[Ss]harpe\s*(?:[Rr]atio\s*)?[:\$=]\s*([-+]?[\d.]+)",
            r"[Ss]harpe\s*[Oo]ranı\s*[:\$=]\s*([-+]?[\d.]+)",
        ],
        "toplam_islem": [
            r"[Tt]otal\s*[Tt]rades?\s*[:\$=]\s*(\d+)",
            r"[Tt]rade\s*[Cc]ount\s*[:\$=]\s*(\d+)",
            r"[Tt]oplam\s*[İI]şlem\s*[:\$=]\s*(\d+)",
            r"[İI]şlem\s*[Ss]ayısı\s*[:\$=]\s*(\d+)",
            r"[Nn]um(?:ber)?\s*[Oo]f\s*[Tt]rades?\s*[:\$=]\s*(\d+)",
        ],
        "kazanan": [
            r"[Ww]inning\s*[Tt]rades?\s*[:\$=]\s*(\d+)",
            r"[Ww]ins?\s*[:\$=]\s*(\d+)",
            r"[Kk]azanan\s*[:\$=]\s*(\d+)",
        ],
        "kaybeden": [
            r"[Ll]osing\s*[Tt]rades?\s*[:\$=]\s*(\d+)",
            r"[Ll]oss(?:es)?\s*[:\$=]\s*(\d+)",
            r"[Kk]aybeden\s*[:\$=]\s*(\d+)",
        ],
    }

    @classmethod
    def _json_oku(cls) -> dict[str, Any] | None:
        """temp_metrics.json varsa okur; yoksa None döner."""
        if not TEMP_METRIK.exists():
            return None
        try:
            ham = TEMP_METRIK.read_text(encoding="utf-8")
            veri = json.loads(ham)
            logger.info("   📂 temp_metrics.json okundu.")
            return veri
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("   temp_metrics.json okunamadı: %s", exc)
            return None

    @classmethod
    def _stdout_ayristir(cls, stdout: str) -> dict[str, Any]:
        """stdout metnini regex ile ayrıştırır."""
        sonuc: dict[str, Any] = {}
        for anahtar, desenler in cls._DESENLER.items():
            for desen in desenler:
                eslesme = re.search(desen, stdout, re.MULTILINE)
                if eslesme:
                    ham_deger = eslesme.group(1).replace(",", "")
                    try:
                        if anahtar in ("toplam_islem", "kazanan", "kaybeden"):
                            sonuc[anahtar] = int(ham_deger)
                        else:
                            sonuc[anahtar] = float(ham_deger)
                        logger.debug(
                            "   ✎ Regex buldu — %-25s = %s", anahtar, ham_deger
                        )
                    except ValueError:
                        pass
                    break   # Anahtar eşleşti; sonraki anahtar
        return sonuc

    @classmethod
    def _bos_metrik(cls) -> dict[str, Any]:
        return {
            "net_pnl":           0.0,
            "net_pnl_pct":       0.0,
            "maks_drawdown_pct": 0.0,
            "win_rate_pct":      0.0,
            "sharpe":            0.0,
            "toplam_islem":      0,
            "kazanan":           0,
            "kaybeden":          0,
        }

    @classmethod
    def ayristir(cls, stdout: str, stderr: str) -> dict[str, Any]:
        """
        Metrikleri çıkarır; öncelik: JSON dosyası → stdout regex → sıfırlar.
        """
        # 1. temp_metrics.json kontrolü (backtest.py yazdıysa)
        json_metrik = cls._json_oku()
        if json_metrik and isinstance(json_metrik, dict):
            bos = cls._bos_metrik()
            bos.update(json_metrik)   # bilinen anahtarları doldur
            # Eksik anahtarları stdout'dan tamamla
            eksik = {k: v for k, v in bos.items() if bos[k] == 0 and k not in json_metrik}
            if eksik:
                stdout_ek = cls._stdout_ayristir(stdout)
                for k in eksik:
                    if k in stdout_ek:
                        bos[k] = stdout_ek[k]
            return bos

        # 2. Saf stdout ayrıştırması
        stdout_metrik = cls._stdout_ayristir(stdout)
        if stdout_metrik:
            logger.info(
                "   📊 stdout'dan %d metrik çıkarıldı.", len(stdout_metrik)
            )
            bos = cls._bos_metrik()
            bos.update(stdout_metrik)
            return bos

        # 3. Son çare: tüm sıfırlar
        logger.warning(
            "   ⚠ Hiçbir metrik ayrıştırılamadı. "
            "backtest.py çıktısının bilinen kalıplarla eşleştiğini "
            "veya temp_metrics.json yazdığını doğrulayın."
        )
        return cls._bos_metrik()

    @staticmethod
    def json_temizle() -> None:
        """temp_metrics.json varsa siler (senaryo arası kirlenmesini önler)."""
        TEMP_METRIK.unlink(missing_ok=True)

    @staticmethod
    def json_yaz(metrikler: dict[str, Any]) -> None:
        """
        Metrikleri temp_metrics.json olarak kök dizine yazar.
        backtest.py kendi içinden bu fonksiyonu çağırıyorsa kullanmak için
        dışarıdan da import edilebilir.
        """
        TEMP_METRIK.write_text(
            json.dumps(metrikler, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

# ──────────────────────────────────────────────────────────────────────────────
# QUANT METRİK HESAPLAYICILARI (bağımsız — backtest.py'den import gerekmez)
# ──────────────────────────────────────────────────────────────────────────────

def sharpe_orani_hesapla(
    getiri_serisi: list[float],
    risk_free_rate: float = 0.0,
    yillik_faktor: float  = 252.0,
) -> float:
    """Günlük getiri serisinden yıllıklandırılmış Sharpe oranı."""
    if len(getiri_serisi) < 2:
        return 0.0
    import statistics
    ort = statistics.mean(getiri_serisi)
    std = statistics.stdev(getiri_serisi)
    if std < 1e-10:
        return 0.0
    gunluk_rf = risk_free_rate / yillik_faktor
    return ((ort - gunluk_rf) / std) * math.sqrt(yillik_faktor)


def maks_drawdown_hesapla(equity_serisi: list[float]) -> float:
    """Equity serisinden maksimum drawdown (negatif yüzde)."""
    if len(equity_serisi) < 2:
        return 0.0
    tepe = equity_serisi[0]
    mdd  = 0.0
    for deger in equity_serisi[1:]:
        if deger > tepe:
            tepe = deger
        if tepe > 1e-10:
            dd = (deger - tepe) / tepe * 100.0
            if dd < mdd:
                mdd = dd
    return mdd

# ──────────────────────────────────────────────────────────────────────────────
# ANA SENARYO KOŞUCU
# ──────────────────────────────────────────────────────────────────────────────

def senaryoyu_koştur(
    senaryo:    Senaryo,
    yamaci:     DosyaYamaci,
    yurutucu:   SubprocessYurutucu,
) -> None:
    """
    Tek bir senaryoyu uçtan uca yürütür:
      1. settings.py'yi fiziksel yama
      2. ml/trainer.py subprocess → XGBoost sıfırdan eğit
      3. backtest.py subprocess → simülasyon
      4. Metrikleri ayrıştır → senaryo.sonuclar
    """
    baslangic = time.perf_counter()

    _ayrac = "═" * 72
    logger.info("")
    logger.info(_ayrac)
    logger.info("▶  %s%s%s", _ANSI_BOLD, senaryo.ad.upper(), _ANSI_RESET)
    logger.info("   %s%s%s", _ANSI_GREY, senaryo.aciklama, _ANSI_RESET)
    logger.info(_ayrac)

    try:
        # ── 1. Fiziksel dosya yamalaması ────────────────────────────────────
        yamaci.yama_uygula(senaryo)

        # ── 2. Trainer subprocess ───────────────────────────────────────────
        if TRAINER_BETIK.exists():
            t_stdout, t_stderr, t_kod = yurutucu.calistir(
                betik=TRAINER_BETIK,
                aciklama="XGBoost Eğitici (ml/trainer.py)",
                timeout=TRAINER_TIMEOUT,
            )
            if t_kod != 0:
                logger.warning(
                    "   ml/trainer.py sıfır dışı çıkış kodu (%d). "
                    "Model önceki checkpoint'ten yüklenecek.", t_kod
                )
        else:
            logger.warning(
                "   ml/trainer.py bulunamadı (%s). "
                "Mevcut model kullanılacak.", TRAINER_BETIK
            )

        # ── 3. Backtest subprocess — her çalışmadan önce JSON temizle ───────
        MetrikAyristirici.json_temizle()

        bt_stdout, bt_stderr, bt_kod = yurutucu.calistir(
            betik=BACKTEST_BETIK,
            aciklama="Bar-by-Bar Simülasyon (backtest.py)",
            timeout=BACKTEST_TIMEOUT,
            ekstra_env={"STRESS_SCENARIO_NAME": senaryo.etiket},
        )


        if bt_kod != 0:
            logger.warning(
                "   backtest.py sıfır dışı çıkış kodu (%d). "
                "Metrikler kısmi olabilir.", bt_kod
            )

        # ── 4. Metrik ayrıştırması ──────────────────────────────────────────
        metrikler = MetrikAyristirici.ayristir(bt_stdout, bt_stderr)
        senaryo.sonuclar = metrikler

        sure = time.perf_counter() - baslangic
        senaryo.sure_saniye = sure

        logger.info(
            "%s✅ Senaryo tamamlandı%s  │  Süre: %.1fs  │  "
            "Net PnL: $%.2f  │  Win Rate: %.1f%%  │  Sharpe: %.3f",
            _ANSI_GREEN, _ANSI_RESET,
            sure,
            metrikler.get("net_pnl", 0.0),
            metrikler.get("win_rate_pct", 0.0),
            metrikler.get("sharpe", 0.0),
        )

    except Exception as exc:
        sure = time.perf_counter() - baslangic
        senaryo.sure_saniye = sure
        senaryo.hata = str(exc)
        logger.error(
            "%s❌ Senaryo HATA ile sonlandı%s: %s",
            _ANSI_RED, _ANSI_RESET, exc,
        )
        logger.debug(traceback.format_exc())

# ──────────────────────────────────────────────────────────────────────────────
# KANTİTATİF STRES TESTİ MATRİSİ RAPORU
# ──────────────────────────────────────────────────────────────────────────────

# Sütun genişlikleri (padding hariç)
_SUTUN_EN: dict[str, int] = {
    "senaryo":  22,
    "tf":        5,
    "gun":       5,
    "thr":       6,
    "rr":        5,
    "pnl":      13,
    "pnl_pct":   9,
    "mdd":      12,
    "wr":        9,
    "sharpe":    8,
    "islem":     7,
    "sure":      8,
}


def _sutun(metin: str, en: int, hiza: str = "<") -> str:
    return f" {metin:{hiza}{en}} "


def _renkli_sayi(
    deger:     float,
    iyi_esik:  float,
    kotu_esik: float,
    format_str: str = ".2f",
    ters: bool = False,   # True → küçük değer iyidir (drawdown)
) -> str:
    """Eşiğe göre yeşil/sarı/kırmızı ANSI renkli sayı döner."""
    if ters:
        iyi  = deger >= iyi_esik
        kotu = deger < kotu_esik
    else:
        iyi  = deger >= iyi_esik
        kotu = deger < kotu_esik

    ham = f"{deger:{format_str}}"
    if iyi:
        return f"{_ANSI_GREEN}{ham}{_ANSI_RESET}"
    if kotu:
        return f"{_ANSI_RED}{ham}{_ANSI_RESET}"
    return f"{_ANSI_YELLOW}{ham}{_ANSI_RESET}"


def _drawdown_renkli(deger: float) -> str:
    """Drawdown için özel renklendirme (negatif değer, daha küçük = daha kötü)."""
    abs_dd = abs(deger)
    ham    = f"{deger:.2f}%"
    if abs_dd <= 5.0:
        return f"{_ANSI_GREEN}{ham}{_ANSI_RESET}"
    if abs_dd <= 15.0:
        return f"{_ANSI_YELLOW}{ham}{_ANSI_RESET}"
    return f"{_ANSI_RED}{ham}{_ANSI_RESET}"


def stres_matrisi_yazdir(senaryolar: list[Senaryo]) -> None:
    """Tüm senaryo sonuçlarını kurumsal ASCII tablo formatında terminale basar."""

    toplam_en = (
        sum(_SUTUN_EN.values()) + len(_SUTUN_EN) * 3 + 1
    )
    KALIN_AYRAC = "═" * toplam_en
    INCE_AYRAC  = "─" * toplam_en

    def baslik_kapat(metin: str) -> str:
        doldurma = toplam_en - len(metin) - 2
        sol  = doldurma // 2
        sag  = doldurma - sol
        return f"║{' ' * sol}{metin}{' ' * sag}║"

    # ── Üst banner ──────────────────────────────────────────────────────────
    print()
    print(f"{_ANSI_BLUE}{_ANSI_BOLD}{'╔' + '═' * (toplam_en - 2) + '╗'}{_ANSI_RESET}")
    for satir in [
        "KANTİTATİF STRES TESTİ MATRİSİ — NASDAQ BOT V2",
        f"Tarih: {datetime.now().strftime('%Y-%m-%d  %H:%M:%S')}",
        f"Senaryo Sayısı: {len(senaryolar)}  │  "
        f"Başarılı: {sum(1 for s in senaryolar if not s.hata)}  │  "
        f"Başarısız: {sum(1 for s in senaryolar if s.hata)}",
    ]:
        print(f"{_ANSI_BLUE}{_ANSI_BOLD}{baslik_kapat(satir)}{_ANSI_RESET}")
    print(f"{_ANSI_BLUE}{_ANSI_BOLD}{'╚' + '═' * (toplam_en - 2) + '╝'}{_ANSI_RESET}")
    print()

    # ── Sütun başlıkları ────────────────────────────────────────────────────
    print(KALIN_AYRAC)
    basliklar = (
        f"│{_sutun('SENARYO', _SUTUN_EN['senaryo'])}"
        f"│{_sutun('TF', _SUTUN_EN['tf'], '^')}"
        f"│{_sutun('GÜN', _SUTUN_EN['gun'], '^')}"
        f"│{_sutun('THR', _SUTUN_EN['thr'], '^')}"
        f"│{_sutun('R:R', _SUTUN_EN['rr'], '^')}"
        f"│{_sutun('NET PnL ($)', _SUTUN_EN['pnl'], '^')}"
        f"│{_sutun('PnL %', _SUTUN_EN['pnl_pct'], '^')}"
        f"│{_sutun('MAX DD (%)', _SUTUN_EN['mdd'], '^')}"
        f"│{_sutun('WIN RT%', _SUTUN_EN['wr'], '^')}"
        f"│{_sutun('SHARPE', _SUTUN_EN['sharpe'], '^')}"
        f"│{_sutun('İŞLEM', _SUTUN_EN['islem'], '^')}"
        f"│{_sutun('SÜRE(s)', _SUTUN_EN['sure'], '^')}"
        f"│"
    )
    print(f"{_ANSI_BOLD}{_ANSI_WHITE}{basliklar}{_ANSI_RESET}")
    print(KALIN_AYRAC)

    # ── Senaryo satırları ────────────────────────────────────────────────────
    for idx, s in enumerate(senaryolar):

        if s.hata:
            # Hatalı satır — tüm metrik hücrelerini "ERR" ile doldur
            kisalt = s.hata[:28] + "…" if len(s.hata) > 28 else s.hata
            print(
                f"│ {_ANSI_RED}{s.etiket:<{_SUTUN_EN['senaryo'] - 1}}{_ANSI_RESET}"
                f"│{s.timeframe:^{_SUTUN_EN['tf'] + 2}}"
                f"│{s.backtest_days:^{_SUTUN_EN['gun'] + 2}}"
                f"│{s.xgb_probability_threshold:^{_SUTUN_EN['thr'] + 2}.2f}"
                f"│{s.reward_risk_ratio:^{_SUTUN_EN['rr'] + 2}.1f}"
                f"│ {_ANSI_RED}{'HATA: ' + kisalt:<{_SUTUN_EN['pnl'] - 1}}{_ANSI_RESET}"
                f"│{'N/A':^{_SUTUN_EN['pnl_pct'] + 2}}"
                f"│{'N/A':^{_SUTUN_EN['mdd'] + 2}}"
                f"│{'N/A':^{_SUTUN_EN['wr'] + 2}}"
                f"│{'N/A':^{_SUTUN_EN['sharpe'] + 2}}"
                f"│{'0':^{_SUTUN_EN['islem'] + 2}}"
                f"│{s.sure_saniye:^{_SUTUN_EN['sure'] + 2}.1f}"
                f"│"
            )
        else:
            r          = s.sonuclar
            net_pnl    = r.get("net_pnl", 0.0)
            pnl_pct    = r.get("net_pnl_pct", 0.0)
            mdd        = r.get("maks_drawdown_pct", 0.0)
            wr         = r.get("win_rate_pct", 0.0)
            sharpe     = r.get("sharpe", 0.0)
            toplam     = r.get("toplam_islem", 0)

            # Renkli değer hücreleri (ANSI padding boşlukla)
            pnl_r    = _renkli_sayi(net_pnl,  0.0,  -500.0, ",.2f")
            pnl_pct_r= _renkli_sayi(pnl_pct,  0.0,  -5.0,   ".2f")
            wr_r     = _renkli_sayi(wr,       55.0,  40.0,  ".1f")
            sharpe_r = _renkli_sayi(sharpe,    1.0,   0.0,  ".3f")
            mdd_r    = _drawdown_renkli(mdd)

            print(
                f"│ {s.etiket:<{_SUTUN_EN['senaryo'] - 1}}"
                f"│{s.timeframe:^{_SUTUN_EN['tf'] + 2}}"
                f"│{s.backtest_days:^{_SUTUN_EN['gun'] + 2}}"
                f"│{s.xgb_probability_threshold:^{_SUTUN_EN['thr'] + 2}.2f}"
                f"│{s.reward_risk_ratio:^{_SUTUN_EN['rr'] + 2}.1f}"
                f"│ {pnl_r:<{_SUTUN_EN['pnl'] + 8}} "
                f"│ {pnl_pct_r}%"
                f"│ {mdd_r:<{_SUTUN_EN['mdd'] + 8}}"
                f"│ {wr_r}%"
                f"│ {sharpe_r}"
                f"│{toplam:^{_SUTUN_EN['islem'] + 2}}"
                f"│{s.sure_saniye:^{_SUTUN_EN['sure'] + 2}.1f}"
                f"│"
            )

        if idx < len(senaryolar) - 1:
            print(INCE_AYRAC)

    print(KALIN_AYRAC)

    # ── Özet panel ───────────────────────────────────────────────────────────
    basarili = [s for s in senaryolar if not s.hata]
    if len(basarili) >= 1:
        en_iyi_sharpe = max(basarili, key=lambda s: s.sonuclar.get("sharpe", -999))
        en_iyi_pnl    = max(basarili, key=lambda s: s.sonuclar.get("net_pnl", -999))
        en_az_dd      = min(
            basarili,
            key=lambda s: abs(s.sonuclar.get("maks_drawdown_pct", 0.0))
        )
        print()
        print(f"{_ANSI_BOLD}{INCE_AYRAC[:55]}{_ANSI_RESET}")
        print(f"{_ANSI_BOLD}  ÖZET BULGULAR{_ANSI_RESET}")
        print(INCE_AYRAC[:55])
        print(
            f"  🏆 En Yüksek Sharpe  → "
            f"{_ANSI_GREEN}{en_iyi_sharpe.etiket}{_ANSI_RESET}  "
            f"(Sharpe: {en_iyi_sharpe.sonuclar.get('sharpe', 0):.3f})"
        )
        print(
            f"  💰 En Yüksek PnL     → "
            f"{_ANSI_GREEN}{en_iyi_pnl.etiket}{_ANSI_RESET}  "
            f"(PnL: ${en_iyi_pnl.sonuclar.get('net_pnl', 0):,.2f})"
        )
        print(
            f"  🛡️  En Düşük Drawdown → "
            f"{_ANSI_GREEN}{en_az_dd.etiket}{_ANSI_RESET}  "
            f"(DD: {en_az_dd.sonuclar.get('maks_drawdown_pct', 0):.2f}%)"
        )
        print(INCE_AYRAC[:55])

    print()
    print(
        f"  Renk skalası:  "
        f"{_ANSI_GREEN}█ İYİ{_ANSI_RESET}  "
        f"{_ANSI_YELLOW}█ ORTA{_ANSI_RESET}  "
        f"{_ANSI_RED}█ KÖTÜ/HATA{_ANSI_RESET}"
    )
    print(
        "  Eşikler: PnL>0 ✓  │  Win Rate>55% ✓  │  "
        "Sharpe>1.0 ✓  │  MaxDD<5% ✓"
    )
    print()

# ──────────────────────────────────────────────────────────────────────────────
# ÖN KONTROLLER
# ──────────────────────────────────────────────────────────────────────────────

def on_kontroller() -> None:
    """Kritik dosya ve dizinlerin varlığını doğrular."""
    hatalar: list[str] = []

    if not SETTINGS_DOSYA.exists():
        hatalar.append(f"config/settings.py bulunamadı: {SETTINGS_DOSYA}")
    if not BACKTEST_BETIK.exists():
        hatalar.append(f"backtest.py bulunamadı: {BACKTEST_BETIK}")

    if hatalar:
        for h in hatalar:
            logger.critical("❌ %s", h)
        logger.critical(
            "Betiği projenin kök dizininden çalıştırdığınızdan emin olun."
        )
        sys.exit(2)

    if not TRAINER_BETIK.exists():
        logger.warning(
            "⚠ ml/trainer.py bulunamadı. "
            "Model eğitim adımı atlanacak, mevcut model kullanılacak."
        )

# ──────────────────────────────────────────────────────────────────────────────
# ENTRYPOINT
# ──────────────────────────────────────────────────────────────────────────────

def main() -> None:
    """
    Stres Testi Odası ana orkestratörü.

    Akış:
      ön kontrol → yedek → [her senaryo: yama + eğit + simüle] → geri yükle → rapor
    """
    logger.info("━" * 72)
    logger.info(
        "  %sNASDAQ BOT V2 — STRES TESTİ ODASI BAŞLIYOR%s",
        _ANSI_BOLD, _ANSI_RESET,
    )
    logger.info("  Mimari     : Fiziksel Dosya Yamama + İzole Subprocess")
    logger.info("  Python     : %s", sys.executable)
    logger.info("  Proje Kök  : %s", PROJE_KOK)
    logger.info("  Senaryo #  : %d", len(SENARYOLAR))
    logger.info("  Zaman      : %s", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    logger.info("━" * 72)

    # ── Ön kontroller ─────────────────────────────────────────────────────────
    on_kontroller()

    yamaci   = DosyaYamaci()
    yurutucu = SubprocessYurutucu()

    # ── Yedek al (kritik — tüm süreç boyunca bu dosya restore edilecek) ──────
    try:
        DosyaYamaci.yedek_al()
    except Exception as exc:
        logger.critical("Yedek alınamadı: %s  — DURDURULUYOR.", exc)
        sys.exit(3)

    # ── Senaryo döngüsü ───────────────────────────────────────────────────────
    try:
        for sira, senaryo in enumerate(SENARYOLAR, start=1):
            logger.info(
                "\n%s🧪 [%d/%d] %s başlıyor…%s",
                _ANSI_BOLD, sira, len(SENARYOLAR), senaryo.ad, _ANSI_RESET,
            )
            senaryoyu_koştur(senaryo, yamaci, yurutucu)

    finally:
        # ── Geri yükleme — hata ya da normal çıkış fark etmez ───────────────
        logger.info("")
        try:
            DosyaYamaci.yedekten_geri_yukle()
        except Exception as exc:
            logger.error(
                "❌ settings.py GERİ YÜKLENEMEDİ: %s\n"
                "   Yedeği manuel geri yükleyin: %s", exc, YEDEK_DOSYA
            )

        # temp_metrics.json temizle
        MetrikAyristirici.json_temizle()

    # ── Son rapor ─────────────────────────────────────────────────────────────
    stres_matrisi_yazdir(SENARYOLAR)

    toplam_sure  = sum(s.sure_saniye for s in SENARYOLAR)
    hata_sayisi  = sum(1 for s in SENARYOLAR if s.hata)
    basari_sayisi= len(SENARYOLAR) - hata_sayisi

    logger.info("━" * 72)
    logger.info("  STRES TESTİ TAMAMLANDI")
    logger.info("  Toplam süre      : %.1f saniye", toplam_sure)
    logger.info("  Başarılı senaryo : %d / %d", basari_sayisi, len(SENARYOLAR))
    if hata_sayisi:
        logger.warning("  Hatalı senaryo   : %d", hata_sayisi)
    logger.info("━" * 72)

    sys.exit(1 if hata_sayisi == len(SENARYOLAR) else 0)


if __name__ == "__main__":
    main()