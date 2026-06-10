//+------------------------------------------------------------------+
//| export_mt5_data.mq5                                              |
//| MT5 Terminalinde çalıştır → V1/csv/symbols/ klasörüne veri aktar |
//|                                                                  |
//| Kullanım:                                                        |
//|   1. MetaEditor'da bu dosyayı aç                                 |
//|   2. Compile et (F7) — 0 error olmalı                            |
//|   3. MT5 Navigator → Scripts → export_mt5_data → grafige sürükle |
//|   4. Çıktı: MT5/MQL5/Files/nasdaq_bot_export/ klasörü           |
//|   5. Bu klasördeki CSV'leri V1/csv/symbols/ klasörüne kopyala   |
//+------------------------------------------------------------------+
#property script_show_inputs

input string         ExportDir    = "nasdaq_bot_export\\"; // Çıktı klasörü
input int            BarsToExport = 100000;                // ~2 yıl = 70k bar
input ENUM_TIMEFRAMES TF          = PERIOD_M15;            // 15 dakika

// Export edilecek semboller (ICMarkets CFD isimleri)
string SYMBOLS[] = {
   "NAS100",   // Nasdaq 100
   "US30",     // Dow Jones
   "SPX500",   // S&P 500
   "XAUUSD",   // Altın
   "USOIL",    // Ham Petrol
   "EURUSD",   // EUR/USD
   "GBPUSD",   // GBP/USD
   "USDJPY",   // USD/JPY
   "BTCUSD",   // Bitcoin
};

//+------------------------------------------------------------------+
void OnStart()
{
   Print("=== MT5 Data Export Başladı ===");
   Print("Timeframe: M15 | Bars: ", BarsToExport);

   // Çıktı klasörünü oluştur
   if(!FolderCreate(ExportDir, FILE_COMMON))
   {
      // Klasör zaten varsa hata vermez, devam et
   }

   int exported = 0;

   for(int s = 0; s < ArraySize(SYMBOLS); s++)
   {
      string symbol = SYMBOLS[s];

      if(!SymbolSelect(symbol, true))
      {
         Print("ATLANDI: ", symbol, " — Market Watch'ta bulunamadı");
         continue;
      }

      // OHLCV dizileri — CopyTickVolume long[] ister!
      datetime rates_time[];
      double   rates_open[], rates_high[], rates_low[], rates_close[];
      long     rates_vol[];   // <-- long (double değil)

      int total = CopyTime      (symbol, TF, 0, BarsToExport, rates_time);
      if(total <= 0) { Print("HATA: ", symbol, " zaman verisi yok"); continue; }

      CopyOpen      (symbol, TF, 0, BarsToExport, rates_open);
      CopyHigh      (symbol, TF, 0, BarsToExport, rates_high);
      CopyLow       (symbol, TF, 0, BarsToExport, rates_low);
      CopyClose     (symbol, TF, 0, BarsToExport, rates_close);
      CopyTickVolume(symbol, TF, 0, BarsToExport, rates_vol); // long[] ✓

      // CSV dosyası — Python DataFeed ile uyumlu format:
      // DATE<TAB>TIME<TAB>OPEN<TAB>HIGH<TAB>LOW<TAB>CLOSE<TAB>TICKVOL<TAB>VOL<TAB>SPREAD
      string filename = ExportDir + symbol + "_15m.csv";
      int fh = FileOpen(filename, FILE_WRITE | FILE_ANSI | FILE_COMMON);

      if(fh == INVALID_HANDLE)
      {
         Print("HATA: Dosya açılamadı: ", filename);
         continue;
      }

      // Başlık
      FileWriteString(fh, "DATE\tTIME\tOPEN\tHIGH\tLOW\tCLOSE\tTICKVOL\tVOL\tSPREAD\n");

      // Eski → Yeni sırayla yaz (index[total-1] en eski bar)
      for(int i = total - 1; i >= 0; i--)
      {
         string date_str = TimeToString(rates_time[i], TIME_DATE);
         string time_str = TimeToString(rates_time[i], TIME_MINUTES);
         // date: 2024.01.15 → nokta kullan (data_feed beklentisi)
         string line = date_str + "\t" + time_str +
                       "\t" + DoubleToString(rates_open[i],  5) +
                       "\t" + DoubleToString(rates_high[i],  5) +
                       "\t" + DoubleToString(rates_low[i],   5) +
                       "\t" + DoubleToString(rates_close[i], 5) +
                       "\t" + IntegerToString(rates_vol[i]) +
                       "\t0\t0\n";
         FileWriteString(fh, line);
      }

      FileClose(fh);
      exported++;
      Print("OK: ", symbol, " → ", filename, " (", total, " bar)");
   }

   Print("=== Export Tamamlandı: ", exported, "/", ArraySize(SYMBOLS), " sembol ===");
   Print("Dosyalar burda: MT5/MQL5/Files/Common/ klasörü → ", ExportDir);
   Print("Kopyala → nasdaq-bot/V1/csv/symbols/");
}
//+------------------------------------------------------------------+
