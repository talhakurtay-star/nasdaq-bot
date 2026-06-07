"""
mt5_symbol_search.py
--------------------
MT5 terminalindeki mevcut sembolleri tarar ve NASDAQ/US100 adaylarını listeler.
MT5 terminali açık ve bağlı olmalıdır.
Kullanım: python mt5_symbol_search.py
"""
import MetaTrader5 as mt5
import sys

if not mt5.initialize():
    print(f"❌ MT5 başlatılamadı: {mt5.last_error()}")
    sys.exit(1)

info = mt5.terminal_info()
print(f"✅ MT5 bağlandı — Terminal: {info.name} build={info.build}")
print()

# Tüm sembolleri al
all_symbols = mt5.symbols_get()
if all_symbols is None:
    print("❌ Sembol listesi alınamadı.")
    mt5.shutdown()
    sys.exit(1)

print(f"Toplam sembol sayısı: {len(all_symbols)}")
print()

# NASDAQ / US100 / NAS100 ile ilgili anahtar kelimeler
keywords = ["nas", "us100", "nq", "ustec", "ndx", "nasdaq", "tech", "us10"]

print("=" * 55)
print("  NASDAQ / US100 ADAY SEMBOLLERİ")
print("=" * 55)
adaylar = []
for s in all_symbols:
    isim = s.name.lower()
    if any(k in isim for k in keywords):
        adaylar.append(s)
        print(f"  → {s.name:20s}  spread={s.spread:6.1f}  "
              f"digits={s.digits}  "
              f"{'[İşlem mevcut]' if s.trade_mode > 0 else '[Salt okunur]'}")

if not adaylar:
    print("  (Spesifik anahtar kelimelerle bulunamadı — geniş tarama yapılıyor…)")
    print()
    print("  Tüm Index / CFD sembolleri:")
    for s in all_symbols:
        if s.path and ("index" in s.path.lower() or "cfd" in s.path.lower()):
            print(f"  → {s.name:20s}  {s.path}")

print()
print("=" * 55)
print("  Bulunan sembolü settings.py'ye girin:")
print('  MT5_SYMBOL = "US100"   ← örnek, yukarıdan doğru olanı seçin')
print("=" * 55)

mt5.shutdown()