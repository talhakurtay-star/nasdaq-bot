import os
import pandas as pd
import MetaTrader5 as mt5
from datetime import datetime, timedelta

try:
    from config.settings import (
        SYMBOL, MT5_SYMBOL, VIX_SYMBOL, SPY_SYMBOL, TIMEFRAME, 
        BACKTEST_DAYS, CACHE_DIR
    )
except ImportError:
    # Geliştirme/test aşamasında config bulunamazsa varsayılan değerler
    SYMBOL = "QQQ"
    MT5_SYMBOL = "TECH"    # Az önce settings.py'de seçtiğimiz çalışan sembol
    VIX_SYMBOL = "VIX"     # MT5 sunucunuzdaki VIX karşılığı (Yoksa otomatik bypass edilecek)
    SPY_SYMBOL = "US500"   # MT5 sunucunuzdaki S&P500 karşılığı (Yoksa otomatik bypass edilecek)
    TIMEFRAME = "15m"
    BACKTEST_DAYS = 90     # MT5 sayesinde artık 60 gün sınırımız yok! İstediğin kadar artırabilirsin.
    CACHE_DIR = "cache"


class DataFeed:
    def __init__(self):
        """
        MT5 Destekli Güvenli Veri Besleyici başlatıcı. 
        Önbellek dizinini ve MT5 bağlantısını kontrol eder.
        """
        if not os.path.exists(CACHE_DIR):
            os.makedirs(CACHE_DIR)
            
        # MT5 terminal bağlantısının aktif olduğundan emin oluyoruz
        if not mt5.initialize():
            print(f"[CRITICAL] DataFeed | MT5 Başlatılamadı! Son Hata: {mt5.last_error()}")
            
    def _get_mt5_timeframe(self, tf_str: str):
        """
        String formatındaki zaman dilimini MT5 sabitlerine haritalandırır.
        """
        tf_mapping = {
            "1m": mt5.TIMEFRAME_M1,
            "5m": mt5.TIMEFRAME_M5,
            "15m": mt5.TIMEFRAME_M15,
            "1h": mt5.TIMEFRAME_H1,
            "1d": mt5.TIMEFRAME_D1
        }
        return tf_mapping.get(tf_str, mt5.TIMEFRAME_M15)

    def _fetch_from_mt5(self, symbol: str, mt5_tf, start_date: datetime, end_date: datetime) -> pd.DataFrame:
        """
        MT5 sunucusundan izole ve sızıntısız bar verisi çeker.
        Sunucudan veri indirilmesini tetikleyen ve senkronizasyon tamamlanana kadar
        bekleyen (retry loop) gelişmiş mekanizmaya sahiptir.
        """
        import time
        
        # Sembolü Market Watch (Piyasa Gözlemi) ekranında aktif et
        mt5.symbol_select(symbol, True)
        
        # Zaman dilimine göre tahmini beklenen bar sayısını hesapla
        duration_days = (end_date - start_date).days
        bars_per_hour = 4  # varsayılan 15m
        if mt5_tf == mt5.TIMEFRAME_M1:
            bars_per_hour = 60
        elif mt5_tf == mt5.TIMEFRAME_M5:
            bars_per_hour = 12
        elif mt5_tf == mt5.TIMEFRAME_M15:
            bars_per_hour = 4
        elif mt5_tf == mt5.TIMEFRAME_H1:
            bars_per_hour = 1
        elif mt5_tf == mt5.TIMEFRAME_D1:
            bars_per_hour = 1.0 / 24.0
            
        # Hafta sonlarını düşerek yaklaşık beklenen bar sayısını hesapla (yaklaşık %70 verimlilik)
        expected_bars = int(duration_days * 24 * bars_per_hour * 0.7)
        expected_bars = max(expected_bars, 100)  # Güvenli taban limit
        
        rates = None
        # MT5 arka planda veriyi çekene kadar maksimum 5 kez deniyoruz (her denemede 1.5s bekler)
        for attempt in range(1, 6):
            rates = mt5.copy_rates_range(symbol, mt5_tf, start_date, end_date)
            
            # Eğer beklenen bar sayısının en az %80'ine ulaştıysak başarılı kabul et
            if rates is not None and len(rates) >= int(expected_bars * 0.8):
                print(f"[OK] {symbol} verileri senkronize edildi. Bar sayısı: {len(rates)} (Deneme {attempt})")
                break
                
            actual_len = len(rates) if rates is not None else 0
            print(f"[WAIT] {symbol} yetersiz bar sayısı: {actual_len}/{expected_bars}. "
                  f"Arka planda geçmiş veri indiriliyor... (Deneme {attempt}/5)")
            
            # copy_rates_from_pos çağrısı yaparak MT5 terminalinin sunucudan veri indirmesini tetikliyoruz
            mt5.copy_rates_from_pos(symbol, mt5_tf, 0, expected_bars)
            time.sleep(1.5)
            
        if rates is None or len(rates) == 0:
            return pd.DataFrame()
            
        df = pd.DataFrame(rates)
        # Unix timestamp'i datetime nesnesine dönüştür
        df['time'] = pd.to_datetime(df['time'], unit='s')
        df.set_index('time', inplace=True)
        return df


    def download_historical_data(self, force_refresh: bool = False) -> pd.DataFrame:
        """
        Ana parite, SPY ve VIX verilerini doğrudan MT5 üzerinden indirir.
        Tüm serileri zaman damgasına (timestamp) göre sızıntısız hizalar.
        """
        cache_path = os.path.join(CACHE_DIR, f"cache_{TIMEFRAME}_{BACKTEST_DAYS}d.csv")

        # 1. Önbellek (Cache) Kontrolü — force_refresh=True ise atla
        if not force_refresh and os.path.exists(cache_path):
            print(f"Veri yerel önbellekten ({cache_path}) okunuyor...")
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            return df


        if force_refresh:
            print("Canlı mod: önbellek atlanıyor, MT5 API'den taze veri enjekte ediliyor...")
        else:
            print("MT5 API üzerinden güncel veri havuzu indiriliyor...")
            
        end_date = datetime.now()
        start_date = end_date - timedelta(days=BACKTEST_DAYS)
        
        mt5_tf = self._get_mt5_timeframe(TIMEFRAME)
        
        # 2. Verilerin MT5'ten Çekilmesi
        main_data = self._fetch_from_mt5(MT5_SYMBOL, mt5_tf, start_date, end_date)
        if main_data.empty:
            raise ValueError(f"[CRITICAL] Ana sembol ({MT5_SYMBOL}) için MT5'ten veri alınamadı! "
                             f"Sembol adını veya MT5 bağlantısını kontrol edin.")

            
        spy_data = self._fetch_from_mt5(SPY_SYMBOL, mt5_tf, start_date, end_date)
        vix_data = self._fetch_from_mt5(VIX_SYMBOL, mt5_tf, start_date, end_date)
        
        # 3. Ana DataFrame'in Oluşturulması ve Hizalama
        # MT5'ten gelen sütun isimleri küçük harflidir ('open', 'high' vb.), bunları orijinal yapıya eşitliyoruz
        main_df = pd.DataFrame({
            'Open': main_data['open'],
            'High': main_data['high'],
            'Low': main_data['low'],
            'Close': main_data['close'],
            'Volume': main_data['tick_volume']
        }, index=main_data.index)
        
        # 4. Korelasyon ve Volatilite Verilerinin Korumalı Entegrasyonu
        if not spy_data.empty:
            main_df['SPY_Close'] = spy_data['close']
        else:
            print(f"[WARNING] {SPY_SYMBOL} MT5'te bulunamadı. Korelasyon için ana fiyat kopyalanıyor.")
            main_df['SPY_Close'] = main_df['Close']  # Çökmemesi için fallback
            
        if not vix_data.empty:
            main_df['VIX_Close'] = vix_data['close']
        else:
            print(f"[WARNING] {VIX_SYMBOL} MT5'te bulunamadı. Sabit VIX taban puanı (15.0) atanıyor.")
            main_df['VIX_Close'] = 15.0  # Volatilite motorunun çökmemesi için güvenli taban değer
        
        # 5. Eksik Verilerin Doldurulması (Look-ahead bias koruması)
        main_df = main_df.ffill().dropna()
        
        # İndirilen veriyi cache'e kaydet — yalnızca backtest modunda (force_refresh=False)
        if not force_refresh:
            main_df.to_csv(cache_path)
            print(f"Veri başarıyla MT5'ten çekildi ve önbelleğe kaydedildi. Toplam: {len(main_df)} bar.")
        else:
            print(f"Canlı Motor | Taze veri enjekte edildi: {len(main_df)} bar.")
        
        return main_df