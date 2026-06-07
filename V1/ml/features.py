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

EMA_50  = 50
EMA_200 = 200
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


def _ema(series: pd.Series, period: int) -> pd.Series:
    if _TALIB:
        return pd.Series(talib.EMA(series.values.astype(float), timeperiod=period), index=series.index)
    return series.ewm(span=period, adjust=False).mean()

def _rsi(series: pd.Series, period: int) -> pd.Series:
    if _TALIB:
        return pd.Series(talib.RSI(series.values.astype(float), timeperiod=period), index=series.index)
    delta = series.diff()
    gain  = delta.clip(lower=0).ewm(com=period - 1, adjust=False).mean()
    loss  = (-delta.clip(upper=0)).ewm(com=period - 1, adjust=False).mean()
    rs    = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))

def _atr(high: pd.Series, low: pd.Series, close: pd.Series, period: int) -> pd.Series:
    if _TALIB:
        return pd.Series(talib.ATR(high.values.astype(float), low.values.astype(float), close.values.astype(float), timeperiod=period), index=close.index)
    tr = pd.concat([
        high - low,
        (high - close.shift()).abs(),
        (low  - close.shift()).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(com=period - 1, adjust=False).mean()

def _adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int):
    if _TALIB:
        adx = pd.Series(talib.ADX(high.values.astype(float), low.values.astype(float), close.values.astype(float), timeperiod=period), index=close.index)
        dmp = pd.Series(talib.PLUS_DI(high.values.astype(float), low.values.astype(float), close.values.astype(float), timeperiod=period), index=close.index)
        dmn = pd.Series(talib.MINUS_DI(high.values.astype(float), low.values.astype(float), close.values.astype(float), timeperiod=period), index=close.index)
        return adx, dmp, dmn
    return pd.Series(np.nan, index=close.index), pd.Series(np.nan, index=close.index), pd.Series(np.nan, index=close.index)

def _macd(series: pd.Series, fast: int, slow: int, signal: int):
    if _TALIB:
        m, s, h = talib.MACD(series.values.astype(float), fastperiod=fast, slowperiod=slow, signalperiod=signal)
        idx = series.index
        return pd.Series(m, index=idx), pd.Series(s, index=idx), pd.Series(h, index=idx)
    ema_f = series.ewm(span=fast, adjust=False).mean()
    ema_s = series.ewm(span=slow, adjust=False).mean()
    macd  = ema_f - ema_s
    sig   = macd.ewm(span=signal, adjust=False).mean()
    return macd, sig, macd - sig

def _bbands(series: pd.Series, period: int, std_mult: float):
    mid   = series.rolling(period).mean()
    std   = series.rolling(period).std(ddof=0)
    upper = mid + std_mult * std
    lower = mid - std_mult * std
    band_range = upper - lower
    pct = (series - lower) / band_range.replace(0, np.nan)
    return lower, mid, upper, pct

def _stochrsi(series: pd.Series, rsi_period: int, k: int, d: int):
    rsi_s = _rsi(series, rsi_period)
    rsi_min = rsi_s.rolling(rsi_period).min()
    rsi_max = rsi_s.rolling(rsi_period).max()
    rng = (rsi_max - rsi_min).replace(0, np.nan)
    stoch = (rsi_s - rsi_min) / rng * 100
    k_line = stoch.rolling(k).mean()
    d_line = k_line.rolling(d).mean()
    return k_line, d_line

def _roc(series: pd.Series, period: int) -> pd.Series:
    return series.pct_change(period) * 100


class FeatureEngine:
    def __init__(self):
        pass

    def calculate_indicators(self, df: pd.DataFrame) -> pd.DataFrame:
        df_copy = df.copy()
        close = df_copy['Close']
        high  = df_copy['High']
        low   = df_copy['Low']

        df_copy[f'EMA_{EMA_FAST}']  = _ema(close, EMA_FAST)
        df_copy[f'EMA_{EMA_SLOW}']  = _ema(close, EMA_SLOW)
        df_copy[f'EMA_{EMA_50}']    = _ema(close, EMA_50)
        df_copy[f'EMA_{EMA_200}']   = _ema(close, EMA_200)

        df_copy['RSI']   = _rsi(close, RSI_PERIOD)
        df_copy['RSI_7'] = _rsi(close, 7)
        df_copy['ATR']   = _atr(high, low, close, ATR_PERIOD)

        adx, dmp, dmn = _adx(high, low, close, ADX_PERIOD)
        df_copy['ADX'] = adx
        df_copy['DMP'] = dmp
        df_copy['DMN'] = dmn

        macd, macd_sig, macd_hist = _macd(close, MACD_FAST, MACD_SLOW, MACD_SIGNAL)
        df_copy['MACD']        = macd
        df_copy['MACD_Signal'] = macd_sig
        df_copy['MACD_Hist']   = macd_hist

        bb_l, bb_m, bb_u, bb_pct = _bbands(close, BB_PERIOD, BB_STD)
        df_copy['BB_Lower'] = bb_l
        df_copy['BB_Mid']   = bb_m
        df_copy['BB_Upper'] = bb_u
        df_copy['BB_Pct']   = bb_pct

        stoch_k, stoch_d = _stochrsi(close, RSI_PERIOD, STOCHRSI_K, STOCHRSI_D)
        df_copy['StochRSI_K'] = stoch_k
        df_copy['StochRSI_D'] = stoch_d

        df_copy['ROC']           = _roc(close, ROC_PERIOD)
        df_copy['EMA_Fast_Slope'] = df_copy[f'EMA_{EMA_FAST}'].diff(3)
        df_copy['EMA_Slow_Slope'] = df_copy[f'EMA_{EMA_SLOW}'].diff(3)

        return df_copy

    def generate_live_features(self, df: pd.DataFrame, current_index: int) -> pd.DataFrame:
        sliced_df = df.iloc[:current_index + 1].copy()

        if len(sliced_df) < WARMUP_BARS:
            return pd.DataFrame()

        current_bar  = sliced_df.iloc[-1]
        prev_bar     = sliced_df.iloc[-2]
        current_time = sliced_df.index[-1]

        features = {}
        close = float(current_bar['Close'])

        # ── Hacim Z-Score ──────────────────────────────────────────────────
        last_20_vol = sliced_df['Volume'].iloc[-20:]
        vol_mean, vol_std = last_20_vol.mean(), last_20_vol.std()
        features['Volume_Z_Score'] = (
            (float(current_bar['Volume']) - vol_mean) / vol_std if vol_std > 0 else 0.0
        )

        # ── Döngüsel Zaman ─────────────────────────────────────────────────
        hour_float = current_time.hour + current_time.minute / 60.0
        features['Sin_Hour']      = np.sin(2 * np.pi * hour_float / 24.0)
        features['Cos_Hour']      = np.cos(2 * np.pi * hour_float / 24.0)
        dow = current_time.dayofweek
        features['Sin_DayOfWeek'] = np.sin(2 * np.pi * dow / 7.0)
        features['Cos_DayOfWeek'] = np.cos(2 * np.pi * dow / 7.0)

        # ── Korelasyon Spread'leri ─────────────────────────────────────────
        spy = current_bar.get('SPY_Close', np.nan)
        vix = current_bar.get('VIX_Close', np.nan)
        features['Spread_QQQ_SPY'] = (close - spy) / spy if (pd.notna(spy) and spy != 0) else 0.0
        features['Spread_QQQ_VIX'] = close / vix         if (pd.notna(vix) and vix != 0) else 0.0

        # ── Temel indikatörler ─────────────────────────────────────────────
        features[f'EMA_{EMA_FAST}'] = float(current_bar.get(f'EMA_{EMA_FAST}', np.nan))
        features[f'EMA_{EMA_SLOW}'] = float(current_bar.get(f'EMA_{EMA_SLOW}', np.nan))
        features[f'EMA_{EMA_50}']   = float(current_bar.get(f'EMA_{EMA_50}',   np.nan))
        features[f'EMA_{EMA_200}']  = float(current_bar.get(f'EMA_{EMA_200}',  np.nan))
        features['RSI']             = float(current_bar.get('RSI',   np.nan))
        features['RSI_7']           = float(current_bar.get('RSI_7', np.nan))
        features['ATR']             = float(current_bar.get('ATR',   np.nan))
        features['Close']           = close

        # ── EMA mesafe oranları ────────────────────────────────────────────
        ema50  = features[f'EMA_{EMA_50}']
        ema200 = features[f'EMA_{EMA_200}']
        ema_f  = features[f'EMA_{EMA_FAST}']
        ema_s  = features[f'EMA_{EMA_SLOW}']
        features['Close_vs_EMA50']  = (close - ema50)  / ema50  if (not np.isnan(ema50)  and ema50  != 0) else 0.0
        features['Close_vs_EMA200'] = (close - ema200) / ema200 if (not np.isnan(ema200) and ema200 != 0) else 0.0
        features['EMA_Gap_Pct']     = (ema_f - ema_s) / ema_s  if (not np.isnan(ema_f) and not np.isnan(ema_s) and ema_s != 0) else 0.0

        # ── ADX ────────────────────────────────────────────────────────────
        adx = float(current_bar.get('ADX', np.nan))
        dmp = float(current_bar.get('DMP', np.nan))
        dmn = float(current_bar.get('DMN', np.nan))
        features['ADX'] = adx
        features['DMP'] = dmp
        features['DMN'] = dmn
        features['DI_Diff'] = (
            (dmp - dmn) / (dmp + dmn) if (not np.isnan(dmp) and not np.isnan(dmn) and (dmp + dmn) > 0) else 0.0
        )

        # ── MACD ───────────────────────────────────────────────────────────
        curr_hist = float(current_bar.get('MACD_Hist', np.nan))
        prev_hist = float(prev_bar.get('MACD_Hist', np.nan))
        features['MACD']             = float(current_bar.get('MACD',        np.nan))
        features['MACD_Signal']      = float(current_bar.get('MACD_Signal', np.nan))
        features['MACD_Hist']        = curr_hist
        features['MACD_Hist_Change'] = (curr_hist - prev_hist) if (not np.isnan(curr_hist) and not np.isnan(prev_hist)) else 0.0

        # ── Bollinger Bands ────────────────────────────────────────────────
        features['BB_Pct'] = float(current_bar.get('BB_Pct', np.nan))
        bb_u = float(current_bar.get('BB_Upper', np.nan))
        bb_l = float(current_bar.get('BB_Lower', np.nan))
        bb_m = float(current_bar.get('BB_Mid',   np.nan))
        features['BB_Width'] = (
            (bb_u - bb_l) / bb_m if (not np.isnan(bb_u) and not np.isnan(bb_l) and not np.isnan(bb_m) and bb_m != 0) else 0.0
        )

        # ── Stochastic RSI ─────────────────────────────────────────────────
        stoch_k = float(current_bar.get('StochRSI_K', np.nan))
        stoch_d = float(current_bar.get('StochRSI_D', np.nan))
        features['StochRSI_K']       = stoch_k
        features['StochRSI_D']       = stoch_d
        features['StochRSI_KD_Diff'] = (stoch_k - stoch_d) if (not np.isnan(stoch_k) and not np.isnan(stoch_d)) else 0.0

        # ── Rate of Change & EMA eğimleri ─────────────────────────────────
        features['ROC']            = float(current_bar.get('ROC',            np.nan))
        features['EMA_Fast_Slope'] = float(current_bar.get('EMA_Fast_Slope', np.nan))
        features['EMA_Slow_Slope'] = float(current_bar.get('EMA_Slow_Slope', np.nan))

        # ── Mum özellikleri ────────────────────────────────────────────────
        o = float(current_bar.get('Open', close))
        h = float(current_bar['High'])
        l = float(current_bar['Low'])
        candle_range = h - l
        if candle_range > 0:
            features['Body_Ratio']       = abs(close - o) / candle_range
            features['Upper_Wick_Ratio'] = (h - max(o, close)) / candle_range
            features['Lower_Wick_Ratio'] = (min(o, close) - l) / candle_range
        else:
            features['Body_Ratio'] = features['Upper_Wick_Ratio'] = features['Lower_Wick_Ratio'] = 0.0
        features['Candle_Direction'] = 1.0 if close >= o else -1.0

        # ── Fiyat pozisyonu (son 20 bar) ───────────────────────────────────
        last_20     = sliced_df.iloc[-20:]
        p_high      = float(last_20['High'].max())
        p_low       = float(last_20['Low'].min())
        p_range     = p_high - p_low
        features['Price_Position_20'] = (close - p_low) / p_range if p_range > 0 else 0.5

        # ── ATR normalize ──────────────────────────────────────────────────
        atr = float(current_bar.get('ATR', np.nan))
        features['ATR_Pct'] = atr / close if (not np.isnan(atr) and close != 0) else 0.0

        # ── Üst periyot trend (son 4 bar ~1 saatlik) ──────────────────────
        if len(sliced_df) >= 4:
            last_4 = sliced_df.iloc[-4:]
            h1_ef  = float(last_4[f'EMA_{EMA_FAST}'].mean())
            h1_es  = float(last_4[f'EMA_{EMA_SLOW}'].mean())
            features['H1_EMA_Trend'] = 1.0 if h1_ef > h1_es else -1.0
        else:
            features['H1_EMA_Trend'] = 0.0

        # ── Hacim trendi ───────────────────────────────────────────────────
        if len(sliced_df) >= 10:
            vol_recent = float(sliced_df['Volume'].iloc[-5:].mean())
            vol_older  = float(sliced_df['Volume'].iloc[-10:-5].mean())
            features['Volume_Trend'] = (vol_recent - vol_older) / vol_older if vol_older > 0 else 0.0
        else:
            features['Volume_Trend'] = 0.0

        return pd.DataFrame([features], index=[current_time])
