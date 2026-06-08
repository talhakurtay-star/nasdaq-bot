"""
ml/features.py — Overfitting-proof feature engine.

Kural: Hiçbir mutlak fiyat değeri (Close, EMA_9, EMA_200 vs.) feature olarak
       giremez. Model belirli fiyat seviyelerini ezberleyemez. Yalnızca
       normalize/göreceli/bounded değerler kullanılır.
"""
import numpy as np
import pandas as pd

try:
    import talib
    _TALIB = True
except ImportError:
    _TALIB = False

try:
    import pandas_ta as _pta
    _PTA = True
except ImportError:
    _PTA = False

if not _TALIB and not _PTA:
    raise ImportError("talib veya pandas_ta kurulu olmalı.")

try:
    from config.settings import (
        EMA_FAST, EMA_SLOW, RSI_PERIOD, ATR_PERIOD
    )
except ImportError:
    EMA_FAST = 9
    EMA_SLOW = 21
    RSI_PERIOD = 14
    ATR_PERIOD = 14

EMA_50       = 50
EMA_200      = 200
ADX_PERIOD   = 14
MACD_FAST    = 12
MACD_SLOW    = 26
MACD_SIGNAL  = 9
BB_PERIOD    = 20
BB_STD       = 2.0
STOCHRSI_K   = 5
STOCHRSI_D   = 3
ROC_PERIOD   = 10
WARMUP_BARS  = 210


# ── Teknik indikatör hesaplayıcılar ──────────────────────────────────────────

def _ema(s, p):
    if _TALIB:
        return pd.Series(talib.EMA(s.values.astype(float), timeperiod=p), index=s.index)
    return s.ewm(span=p, adjust=False).mean()

def _rsi(s, p):
    if _TALIB:
        return pd.Series(talib.RSI(s.values.astype(float), timeperiod=p), index=s.index)
    d = s.diff()
    g = d.clip(lower=0).ewm(com=p-1, adjust=False).mean()
    l = (-d.clip(upper=0)).ewm(com=p-1, adjust=False).mean()
    return 100 - (100 / (1 + g / l.replace(0, np.nan)))

def _atr(h, l, c, p):
    if _TALIB:
        return pd.Series(talib.ATR(h.values.astype(float), l.values.astype(float),
                                   c.values.astype(float), timeperiod=p), index=c.index)
    tr = pd.concat([h-l, (h-c.shift()).abs(), (l-c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(com=p-1, adjust=False).mean()

def _adx(h, l, c, p):
    if _TALIB:
        adx = pd.Series(talib.ADX(h.values.astype(float), l.values.astype(float),
                                   c.values.astype(float), timeperiod=p), index=c.index)
        dmp = pd.Series(talib.PLUS_DI(h.values.astype(float), l.values.astype(float),
                                       c.values.astype(float), timeperiod=p), index=c.index)
        dmn = pd.Series(talib.MINUS_DI(h.values.astype(float), l.values.astype(float),
                                        c.values.astype(float), timeperiod=p), index=c.index)
        return adx, dmp, dmn
    
    if _PTA:
        try:
            import pandas_ta as pta
            adx_df = pta.adx(h, l, c, length=p)
            if adx_df is not None and not adx_df.empty:
                adx_col = [col for col in adx_df.columns if col.startswith("ADX")][0]
                dmp_col = [col for col in adx_df.columns if col.startswith("DMP")][0]
                dmn_col = [col for col in adx_df.columns if col.startswith("DMN")][0]
                return adx_df[adx_col], adx_df[dmp_col], adx_df[dmn_col]
        except Exception:
            pass

    # Pure pandas manual calculation of ADX, PLUS_DI, MINUS_DI
    up_move = h.diff()
    down_move = -l.diff()
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    
    tr = pd.concat([h - l, (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    
    tr_smoothed = tr.ewm(alpha=1.0/p, adjust=False).mean()
    plus_dm_smoothed = pd.Series(plus_dm, index=c.index).ewm(alpha=1.0/p, adjust=False).mean()
    minus_dm_smoothed = pd.Series(minus_dm, index=c.index).ewm(alpha=1.0/p, adjust=False).mean()
    
    tr_smoothed_safe = tr_smoothed.replace(0, np.nan)
    
    plus_di = 100.0 * plus_dm_smoothed / tr_smoothed_safe
    minus_di = 100.0 * minus_dm_smoothed / tr_smoothed_safe
    
    di_sum = plus_di + minus_di
    di_sum_safe = di_sum.replace(0, np.nan)
    
    dx = 100.0 * (plus_di - minus_di).abs() / di_sum_safe
    adx = dx.ewm(alpha=1.0/p, adjust=False).mean()
    
    return adx, plus_di, minus_di

def _macd(s, fast, slow, signal):
    if _TALIB:
        m, sig, h = talib.MACD(s.values.astype(float), fastperiod=fast,
                                slowperiod=slow, signalperiod=signal)
        idx = s.index
        return pd.Series(m, idx), pd.Series(sig, idx), pd.Series(h, idx)
    ef = s.ewm(span=fast, adjust=False).mean()
    es = s.ewm(span=slow, adjust=False).mean()
    mac = ef - es
    sig = mac.ewm(span=signal, adjust=False).mean()
    return mac, sig, mac - sig

def _bbands(s, p, std_mult):
    mid = s.rolling(p).mean()
    std = s.rolling(p).std(ddof=0)
    up  = mid + std_mult * std
    lo  = mid - std_mult * std
    rng = (up - lo).replace(0, np.nan)
    pct = (s - lo) / rng
    return lo, mid, up, pct

def _stochrsi(s, rsi_p, k, d):
    r = _rsi(s, rsi_p)
    rmin = r.rolling(rsi_p).min()
    rmax = r.rolling(rsi_p).max()
    stoch = (r - rmin) / (rmax - rmin).replace(0, np.nan) * 100
    kl = stoch.rolling(k).mean()
    dl = kl.rolling(d).mean()
    return kl, dl

def _roc(s, p):
    return s.pct_change(p) * 100


class FeatureEngine:
    def __init__(self):
        pass

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        c, h, l = df['Close'], df['High'], df['Low']

        df[f'EMA_{EMA_FAST}']  = _ema(c, EMA_FAST)
        df[f'EMA_{EMA_SLOW}']  = _ema(c, EMA_SLOW)
        df[f'EMA_{EMA_50}']    = _ema(c, EMA_50)
        df[f'EMA_{EMA_200}']   = _ema(c, EMA_200)

        df['RSI']   = _rsi(c, RSI_PERIOD)
        df['RSI_7'] = _rsi(c, 7)
        df['ATR']   = _atr(h, l, c, ATR_PERIOD)

        adx, dmp, dmn = _adx(h, l, c, ADX_PERIOD)
        df['ADX'], df['DMP'], df['DMN'] = adx, dmp, dmn

        macd, macd_sig, macd_hist = _macd(c, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
        df['MACD']        = macd
        df['MACD_Signal'] = macd_sig
        df['MACD_Hist']   = macd_hist

        bb_l, bb_m, bb_u, bb_pct = _bbands(c, BB_PERIOD, BB_STD)
        df['BB_Lower'], df['BB_Mid'] = bb_l, bb_m
        df['BB_Upper'], df['BB_Pct'] = bb_u, bb_pct

        sk, sd = _stochrsi(c, RSI_PERIOD, STOCHRSI_K, STOCHRSI_D)
        df['StochRSI_K'], df['StochRSI_D'] = sk, sd

        df['ROC'] = _roc(c, ROC_PERIOD)

        # EMA hizalama süresi: kaç bardır ema_fast > ema_slow?
        aligned = (df[f'EMA_{EMA_FAST}'] > df[f'EMA_{EMA_SLOW}']).astype(int)
        # Arka arkaya sayım (grup değişiminde sıfırla)
        grp = (aligned != aligned.shift()).cumsum()
        df['EMA_Align_Bars'] = aligned.groupby(grp).cumcount() + 1
        df['EMA_Align_Bars'] = df['EMA_Align_Bars'] * (2*aligned - 1)  # short için negatif

        # RSI 3 barlık değişim trendi
        df['RSI_Change3'] = df['RSI'].diff(3)

        # MACD histogram değişimi
        df['MACD_Hist_Change'] = df['MACD_Hist'].diff(1)

        # ATR normalize eğim
        df['EMA_Fast_Slope'] = df[f'EMA_{EMA_FAST}'].diff(3) / df['ATR'].replace(0, np.nan)
        df['EMA_Slow_Slope'] = df[f'EMA_{EMA_SLOW}'].diff(3) / df['ATR'].replace(0, np.nan)

        return df

    def generate_live_features(self, df: pd.DataFrame, current_index: int) -> pd.DataFrame:
        sliced = df.iloc[:current_index + 1].copy()
        if len(sliced) < WARMUP_BARS:
            return pd.DataFrame()

        bar      = sliced.iloc[-1]
        prev_bar = sliced.iloc[-2]
        ts       = sliced.index[-1]
        close    = float(bar['Close'])
        atr      = float(bar.get('ATR', np.nan))
        atr_safe = atr if (not np.isnan(atr) and atr > 0) else 1.0

        feat = {}

        # ── 1. Zaman (döngüsel, bounded) ─────────────────────────────────
        hf = ts.hour + ts.minute / 60.0
        feat['Sin_Hour']      = np.sin(2 * np.pi * hf / 24.0)
        feat['Cos_Hour']      = np.cos(2 * np.pi * hf / 24.0)
        dow = ts.dayofweek
        feat['Sin_DayOfWeek'] = np.sin(2 * np.pi * dow / 7.0)
        feat['Cos_DayOfWeek'] = np.cos(2 * np.pi * dow / 7.0)

        # ── 2. RSI (0-100, bounded) ───────────────────────────────────────
        feat['RSI']         = float(bar.get('RSI',        np.nan))
        feat['RSI_7']       = float(bar.get('RSI_7',      np.nan))
        feat['RSI_Change3'] = float(bar.get('RSI_Change3', 0.0) or 0.0)

        # ── 3. ADX + DI (0-100, bounded) ─────────────────────────────────
        adx = float(bar.get('ADX', np.nan))
        dmp = float(bar.get('DMP', np.nan))
        dmn = float(bar.get('DMN', np.nan))
        feat['ADX'] = adx
        feat['DMP'] = dmp
        feat['DMN'] = dmn
        feat['DI_Diff'] = (
            (dmp - dmn) / (dmp + dmn)
            if not np.isnan(dmp) and not np.isnan(dmn) and (dmp + dmn) > 0
            else 0.0
        )

        # ── 4. MACD (ATR normalize) ───────────────────────────────────────
        macd_h  = float(bar.get('MACD_Hist', np.nan))
        macd_hc = float(bar.get('MACD_Hist_Change', 0.0) or 0.0)
        feat['MACD_Hist_ATR']    = macd_h / atr_safe  if not np.isnan(macd_h)  else 0.0
        feat['MACD_Hist_Change'] = macd_hc / atr_safe if not np.isnan(macd_hc) else 0.0
        # MACD ve Signal farkı (ATR normalize)
        macd_v = float(bar.get('MACD', np.nan))
        macd_s = float(bar.get('MACD_Signal', np.nan))
        feat['MACD_Signal_Gap'] = (
            (macd_v - macd_s) / atr_safe
            if not np.isnan(macd_v) and not np.isnan(macd_s)
            else 0.0
        )

        # ── 5. Bollinger Bands (bounded 0-1) ─────────────────────────────
        feat['BB_Pct'] = float(bar.get('BB_Pct', np.nan))
        bb_u = float(bar.get('BB_Upper', np.nan))
        bb_l = float(bar.get('BB_Lower', np.nan))
        bb_m = float(bar.get('BB_Mid',   np.nan))
        feat['BB_Width'] = (
            (bb_u - bb_l) / bb_m
            if not np.isnan(bb_u) and not np.isnan(bb_l) and not np.isnan(bb_m) and bb_m != 0
            else 0.0
        )

        # ── 6. Stochastic RSI (0-100, bounded) ───────────────────────────
        sk = float(bar.get('StochRSI_K', np.nan))
        sd = float(bar.get('StochRSI_D', np.nan))
        feat['StochRSI_K']       = sk
        feat['StochRSI_D']       = sd
        feat['StochRSI_KD_Diff'] = (sk - sd) if not np.isnan(sk) and not np.isnan(sd) else 0.0

        # ── 7. Rate of Change (normalize) ────────────────────────────────
        feat['ROC'] = float(bar.get('ROC', np.nan))

        # ── 8. EMA göreceli mesafeler (normalize) ────────────────────────
        ef  = float(bar.get(f'EMA_{EMA_FAST}', np.nan))
        es  = float(bar.get(f'EMA_{EMA_SLOW}', np.nan))
        e50 = float(bar.get(f'EMA_{EMA_50}',   np.nan))
        e200= float(bar.get(f'EMA_{EMA_200}',  np.nan))

        feat['EMA_Gap_Pct']     = (ef - es)   / es    if not np.isnan(ef)  and not np.isnan(es)   and es   != 0 else 0.0
        feat['Close_vs_EMA50']  = (close-e50)  / e50   if not np.isnan(e50)  and e50  != 0 else 0.0
        feat['Close_vs_EMA200'] = (close-e200) / e200  if not np.isnan(e200) and e200 != 0 else 0.0

        # ── 9. EMA normalize eğimler (ATR birimiyle) ─────────────────────
        feat['EMA_Fast_Slope'] = float(bar.get('EMA_Fast_Slope', 0.0) or 0.0)
        feat['EMA_Slow_Slope'] = float(bar.get('EMA_Slow_Slope', 0.0) or 0.0)

        # ── 10. EMA hizalama süresi ───────────────────────────────────────
        align_bars = float(bar.get('EMA_Align_Bars', 0.0) or 0.0)
        # Clamp: -50 ile +50 arası, aşırı değerleri sınırla
        feat['EMA_Align_Bars'] = max(-50.0, min(50.0, align_bars))

        # ── 11. Mum yapısı ────────────────────────────────────────────────
        o = float(bar.get('Open', close))
        h = float(bar['High'])
        l = float(bar['Low'])
        cr = h - l
        if cr > 0:
            feat['Body_Ratio']       = abs(close - o) / cr
            feat['Upper_Wick_Ratio'] = (h - max(o, close)) / cr
            feat['Lower_Wick_Ratio'] = (min(o, close) - l) / cr
        else:
            feat['Body_Ratio'] = feat['Upper_Wick_Ratio'] = feat['Lower_Wick_Ratio'] = 0.0
        feat['Candle_Direction'] = 1.0 if close >= o else -1.0

        # ── 12. Fiyat pozisyonu son 20 bar içinde (0-1) ───────────────────
        last20 = sliced.iloc[-20:]
        ph, pl = float(last20['High'].max()), float(last20['Low'].min())
        pr = ph - pl
        feat['Price_Position_20'] = (close - pl) / pr if pr > 0 else 0.5

        # ── 13. ATR yüzde (normalize) ─────────────────────────────────────
        feat['ATR_Pct'] = atr / close if not np.isnan(atr) and close != 0 else 0.0

        # ── 14. Üst periyot EMA trend (son 4 bar = ~1 saat) ──────────────
        if len(sliced) >= 4:
            last4 = sliced.iloc[-4:]
            h1ef  = float(last4[f'EMA_{EMA_FAST}'].mean())
            h1es  = float(last4[f'EMA_{EMA_SLOW}'].mean())
            feat['H1_EMA_Trend'] = 1.0 if h1ef > h1es else -1.0
        else:
            feat['H1_EMA_Trend'] = 0.0

        # ── 15. Hacim özellikleri ─────────────────────────────────────────
        last20_vol = sliced['Volume'].iloc[-20:]
        vm, vs = float(last20_vol.mean()), float(last20_vol.std())
        feat['Volume_Z_Score'] = (float(bar['Volume']) - vm) / vs if vs > 0 else 0.0

        if len(sliced) >= 10:
            vr = float(sliced['Volume'].iloc[-5:].mean())
            vo = float(sliced['Volume'].iloc[-10:-5].mean())
            feat['Volume_Trend'] = (vr - vo) / vo if vo > 0 else 0.0
        else:
            feat['Volume_Trend'] = 0.0

        return pd.DataFrame([feat], index=[ts])
