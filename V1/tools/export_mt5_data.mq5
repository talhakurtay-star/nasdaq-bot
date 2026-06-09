//+------------------------------------------------------------------+
//| export_mt5_data.mq5                                              |
//| MT5 Terminalinde çalıştır → V1/csv/symbols/ klasörüne veri aktar |
//|                                                                  |
//| Kullanım:                                                        |
//|   1. MetaEditor'da bu dosyayı aç                                 |
//|   2. Compile et (F7)                                             |
//|   3. MT5 → Tools → Scripts → ExportNasdaqData olarak çalıştır   |
//|   4. Çıktı: MT5/MQL5/Files/nasdaq_bot_export/ klasörü           |
//|   5. Bu klasördeki CSV'leri V1/csv/symbols/ klasörüne kopyala   |
//+------------------------------------------------------------------+
#property script_show_inputs

input string   ExportDir    = "nasdaq_bot_export\\";    // Çıktı klasörü
input int      BarsToExport = 100000;                   // Kaç bar (yaklaşık 2 yıl = 70k bar)
input ENUM_TIMEFRAMES TF    = PERIOD_M15;               // 15 dakika

// Export edilecek semboller (ICMarkets CFD isimleri)
// Farklı sektörlerden → korelasyon azaltır
string SYMBOLS[] = {
   // ABD Equity Index
   "NAS100",    // Nasdaq 100 (ana sembol)
   "US30",      // Dow Jones Industrial
   "SPX500",    // S&P 500
   // Emtia
   "XAUUSD",    // Altın (savunma/güvenli liman)
   "USOIL",     // Ham petrol (enerji sektörü)
   // Döviz (farklı ekonomik döngü)
   "EURUSD",    // EUR/USD
   "GBPUSD",    // GBP/USD
   "USDJPY",    // USD/JPY (ters korelasyon aracı)
   // Kripto (yüksek volatilite)
   "BTCUSD",    // Bitcoin
};

//+------------------------------------------------------------------+
void OnStart()
{
   Print("=== MT5 Data Export Başladı ===");
   Print("Timeframe: M15 | Bars: ", BarsToExport);

   int exported = 0;

   for(int s = 0; s < ArraySize(SYMBOLS); s++)
   {
      string symbol = SYMBOLS[s];

      // Sembol mevcut mu kontrol et
      if(!SymbolSelect(symbol, true))
      {
         Print("ATLANDI: ", symbol, " — Market Watch'ta bulunamadı");
         continue;
      }

      // Veriyi yükle
      datetime rates_time[];
      double   rates_open[], rates_high[], rates_low[], rates_close[], rates_vol[];

      int total = CopyTime  (symbol, TF, 0, BarsToExport, rates_time);
                  CopyOpen  (symbol, TF, 0, BarsToExport, rates_open);
                  CopyHigh  (symbol, TF, 0, BarsToExport, rates_high);
                  CopyLow   (symbol, TF, 0, BarsToExport, rates_low);
                  CopyClose (symbol, TF, 0, BarsToExport, rates_close);
                  CopyTickVolume(symbol, TF, 0, BarsToExport, rates_vol);

      if(total <= 0)
      {
         Print("HATA: ", symbol, " için veri alınamadı");
         continue;
      }

      // CSV dosyası oluştur
      string filename = ExportDir + symbol + "_15m.csv";
      int fh = FileOpen(filename, FILE_WRITE | FILE_CSV | FILE_ANSI, ',');

      if(fh == INVALID_HANDLE)
      {
         Print("HATA: Dosya açılamadı: ", filename);
         continue;
      }

      // Başlık satırı (Python DataFeed ile uyumlu format)
      FileWrite(fh, "Datetime", "Open", "High", "Low", "Close", "Volume");

      // Verileri eski→yeni sırayla yaz (index[0] en eski)
      for(int i = total - 1; i >= 0; i--)
      {
         string dt = TimeToString(rates_time[i], TIME_DATE | TIME_MINUTES);
         // Format: 2024.01.15 09:30 → 2024-01-15 09:30:00
         StringReplace(dt, ".", "-");
         dt += ":00";

         FileWrite(fh,
            dt,
            DoubleToString(rates_open[i],  5),
            DoubleToString(rates_high[i],  5),
            DoubleToString(rates_low[i],   5),
            DoubleToString(rates_close[i], 5),
            DoubleToString(rates_vol[i],   0)
         );
      }

      FileClose(fh);
      exported++;
      Print("OK: ", symbol, " → ", filename, " (", total, " bar)");
   }

   Print("=== Export Tamamlandı: ", exported, "/", ArraySize(SYMBOLS), " sembol ===");
   Print("Dosyalar: MT5/MQL5/Files/", ExportDir);
   Print("Bu dosyaları kopyala → nasdaq-bot/V1/csv/symbols/");
}
//+------------------------------------------------------------------+
