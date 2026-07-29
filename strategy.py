import math
import numpy as np
import pandas as pd

# -----------------------------
# Indicator helpers
# -----------------------------
def ema(s, n):
    return s.ewm(span=n, adjust=False, min_periods=n).mean()

def cci(df, n=14):
    tp = (df.high + df.low + df.close) / 3
    sma = tp.rolling(n).mean()
    mad = tp.rolling(n).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - sma) / (0.015 * mad.replace(0, np.nan))

def bollinger_bands(close, n=20, stds=2):
    mid = close.rolling(n).mean()
    std = close.rolling(n).std()
    upper = mid + stds * std
    lower = mid - stds * std
    return upper, mid, lower

def wma(s, n):
    weights = np.arange(1, n + 1)
    return s.rolling(n).apply(lambda x: np.dot(x, weights) / weights.sum(), raw=True)

def hma(s, n):
    half_len = int(n / 2)
    sqrt_len = int(np.sqrt(n))
    wma_half = wma(s, half_len)
    wma_full = wma(s, n)
    diff = 2 * wma_half - wma_full
    return wma(diff, sqrt_len)

def cmf(df, n=20):
    cl_diff = (df.close - df.low) - (df.high - df.close)
    hl_diff = (df.high - df.low).replace(0, np.nan)
    mfv = (cl_diff / hl_diff) * df.volume
    return mfv.rolling(n).sum() / df.volume.rolling(n).sum().replace(0, np.nan)

def mfi(df, n=14):
    tp = (df.high + df.low + df.close) / 3
    tp_diff = tp.diff()
    raw_money_flow = tp * df.volume
    pos_flow = raw_money_flow.where(tp_diff > 0, 0.0)
    neg_flow = raw_money_flow.where(tp_diff < 0, 0.0)
    pos_mf = pos_flow.rolling(n).sum()
    neg_mf = neg_flow.rolling(n).sum()
    m_ratio = pos_mf / neg_mf.replace(0, np.nan)
    return 100 - 100 / (1 + m_ratio)

def rsi(close, n=14):
    d = close.diff()
    up = d.clip(lower=0)
    dn = -d.clip(upper=0)
    au = up.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    ad = dn.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    return 100 - 100 / (1 + au / ad.replace(0, np.nan))

def atr(df, n=14):
    pc = df.close.shift(1)
    tr = pd.concat([
        (df.high - df.low).abs(),
        (df.high - pc).abs(),
        (df.low - pc).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False, min_periods=n).mean()

def adx(df, n=14):
    high, low, close = df.high, df.low, df.close
    up = high.diff()
    dn = -low.diff()
    plus = up.where((up > dn) & (up > 0), 0.0)
    minus = dn.where((dn > up) & (dn > 0), 0.0)
    pc = close.shift(1)
    tr = pd.concat([(high-low).abs(), (high-pc).abs(), (low-pc).abs()], axis=1).max(axis=1)
    atrn = tr.ewm(alpha=1/n, adjust=False, min_periods=n).mean()
    pdi = 100 * plus.ewm(alpha=1/n, adjust=False, min_periods=n).mean() / atrn
    mdi = 100 * minus.ewm(alpha=1/n, adjust=False, min_periods=n).mean() / atrn
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1/n, adjust=False, min_periods=n).mean(), pdi, mdi

def efficiency(close, n):
    return (close - close.shift(n)).abs() / close.diff().abs().rolling(n).sum().replace(0, np.nan)

def chop_proxy(df, n=24):
    rng = (df.high.rolling(n).max() - df.low.rolling(n).min()).replace(0, np.nan)
    return df.atr14.rolling(n).sum() / rng

def cross_above(a, b):
    return (a.shift(1) <= b.shift(1)) & (a > b)

def cross_below(a, b):
    return (a.shift(1) >= b.shift(1)) & (a < b)

def sigmoid(x):
    x = max(-20, min(20, float(x)))
    return 1.0 / (1.0 + math.exp(-x))

def clamp(x, lo, hi):
    if x is None or not np.isfinite(x):
        return 0.0
    return max(lo, min(hi, float(x)))

# -----------------------------
# Asset specific configurations
# -----------------------------
SCALPER_ASSET_CONFIG = {
    # High-probability wide-stop scalping configurations
    ('BTC', 'LONG'):  {'stdev': 2.0, 'cci_th': -135, 'trend_filter': True, 'tp_atr': 1.5, 'sl_atr': 3.0, 'max_hold': 12},
    ('BTC', 'SHORT'): {'stdev': 2.2, 'cci_th': 140,  'trend_filter': True, 'tp_atr': 1.5, 'sl_atr': 3.0, 'max_hold': 12},
    ('ETH', 'LONG'):  {'stdev': 2.0, 'cci_th': -130, 'trend_filter': True, 'tp_atr': 1.5, 'sl_atr': 3.0, 'max_hold': 12},
    ('ETH', 'SHORT'): {'stdev': 2.3, 'cci_th': 135,  'trend_filter': True, 'tp_atr': 1.5, 'sl_atr': 3.0, 'max_hold': 12},
    ('SOL', 'LONG'):  {'stdev': 2.2, 'cci_th': -140, 'trend_filter': True, 'tp_atr': 1.5, 'sl_atr': 3.5, 'max_hold': 12},
    ('SOL', 'SHORT'): {'stdev': 2.4, 'cci_th': 145,  'trend_filter': True, 'tp_atr': 1.5, 'sl_atr': 3.5, 'max_hold': 12},
    ('AVAX', 'LONG'): {'stdev': 2.2, 'cci_th': -135, 'trend_filter': True, 'tp_atr': 1.5, 'sl_atr': 3.5, 'max_hold': 12},
    ('AVAX', 'SHORT'):{'stdev': 2.3, 'cci_th': 135,  'trend_filter': True, 'tp_atr': 1.5, 'sl_atr': 3.5, 'max_hold': 12},
    ('BNB', 'LONG'):  {'stdev': 2.1, 'cci_th': -135, 'trend_filter': True, 'tp_atr': 1.5, 'sl_atr': 3.5, 'max_hold': 12},
    ('BNB', 'SHORT'): {'stdev': 2.2, 'cci_th': 135,  'trend_filter': True, 'tp_atr': 1.5, 'sl_atr': 3.5, 'max_hold': 12},
    ('HYPE', 'LONG'): {'stdev': 2.5, 'cci_th': -150, 'trend_filter': True, 'tp_atr': 1.5, 'sl_atr': 4.0, 'max_hold': 12},
    ('HYPE', 'SHORT'):{'stdev': 2.5, 'cci_th': 150,  'trend_filter': True, 'tp_atr': 1.5, 'sl_atr': 4.0, 'max_hold': 12},
    ('LINK', 'LONG'): {'stdev': 2.0, 'cci_th': -130, 'trend_filter': False, 'tp_atr': 1.5, 'sl_atr': 3.5, 'max_hold': 12},
    ('LINK', 'SHORT'):{'stdev': 2.2, 'cci_th': 130,  'trend_filter': True, 'tp_atr': 1.5, 'sl_atr': 3.5, 'max_hold': 12},
    ('NEAR', 'LONG'): {'stdev': 2.2, 'cci_th': -140, 'trend_filter': True, 'tp_atr': 1.5, 'sl_atr': 3.5, 'max_hold': 12},
    ('NEAR', 'SHORT'):{'stdev': 2.3, 'cci_th': 140,  'trend_filter': True, 'tp_atr': 1.5, 'sl_atr': 3.5, 'max_hold': 12},
    ('PEPE', 'LONG'): {'stdev': 2.5, 'cci_th': -150, 'trend_filter': True, 'tp_atr': 1.5, 'sl_atr': 4.0, 'max_hold': 12},
    ('PEPE', 'SHORT'):{'stdev': 2.5, 'cci_th': 150,  'trend_filter': True, 'tp_atr': 1.5, 'sl_atr': 4.0, 'max_hold': 12},
    ('WIF', 'LONG'):  {'stdev': 2.5, 'cci_th': -150, 'trend_filter': True, 'tp_atr': 1.5, 'sl_atr': 4.0, 'max_hold': 12},
    ('WIF', 'SHORT'): {'stdev': 2.5, 'cci_th': 150,  'trend_filter': True, 'tp_atr': 1.5, 'sl_atr': 4.0, 'max_hold': 12},
    ('FET', 'LONG'):  {'stdev': 2.3, 'cci_th': -140, 'trend_filter': True, 'tp_atr': 1.5, 'sl_atr': 3.5, 'max_hold': 12},
    ('FET', 'SHORT'): {'stdev': 2.3, 'cci_th': 140,  'trend_filter': True, 'tp_atr': 1.5, 'sl_atr': 3.5, 'max_hold': 12},
}

class StrategyBrain:
    def __init__(self, cfg=None, model_path=None, meta_path=None):
        self.cfg = cfg

    def calculate_cci_bb_signal(self, pair, df):
        if df is None or df.empty or len(df) < 110:
            return None
        d = df.copy().sort_index()
        d['cci_val'] = cci(d, 14)
        d['ema100'] = ema(d.close, 100)
        
        best = None
        for side in ['LONG', 'SHORT']:
            cfg = SCALPER_ASSET_CONFIG.get((pair, side))
            if not cfg:
                cfg = {'stdev': 2.0, 'cci_th': -134 if side == 'LONG' else 134, 'trend_filter': True, 'tp_atr': 1.5, 'sl_atr': 3.0, 'max_hold': 12}
            
            upper, mid, lower = bollinger_bands(d.close, 20, cfg['stdev'])
            d['bb_lower'] = lower
            d['bb_upper'] = upper
            
            row = d.iloc[-1]
            entry = float(row.close)
            
            trend_ok = True
            if cfg['trend_filter']:
                ema_val = float(row.ema100)
                if side == 'LONG' and entry < ema_val:
                    trend_ok = False
                elif side == 'SHORT' and entry > ema_val:
                    trend_ok = False
                    
            if not trend_ok:
                continue
                
            trigger = False
            if side == 'LONG' and row.cci_val <= cfg['cci_th'] and entry < row.bb_lower:
                trigger = True
            elif side == 'SHORT' and row.cci_val >= cfg['cci_th'] and entry > row.bb_upper:
                trigger = True
                
            if not trigger:
                continue
                
            d['atr_val'] = atr(d, 14)
            atr_now = float(d['atr_val'].iloc[-1])
            if not np.isfinite(atr_now) or atr_now <= 0:
                atr_now = entry * 0.01
                
            target_abs = cfg['tp_atr'] * atr_now
            risk_abs = cfg['sl_atr'] * atr_now
            
            if side == 'LONG':
                sl = entry - risk_abs
                tp = entry + target_abs
            else:
                sl = entry + risk_abs
                tp = entry - target_abs
                
            sig = {
                'pair': pair,
                'side': side,
                'setup': 'CCI_BB_SCALPER',
                'probability': 0.85,
                'predicted_R': 0.5,
                'entry': entry,
                'sl': float(sl),
                'tp': float(tp),
                'risk_pct': float(risk_abs / entry),
                'trigger_time': str(d.index[-1]),
                'signal_time': str(d.index[-1]),
                'risk_mult': 1.0,
                'max_hold_hours': int(cfg['max_hold']),
                'meta': {
                    'family': 'CCI_BB_SCALPER',
                    'prob_th': 0.70,
                    'er_th': 0.0,
                    'decision_makers': {'cci': float(row.cci_val), 'close': entry, 'bb_lower': float(row.bb_lower), 'bb_upper': float(row.bb_upper)},
                    'predicted_R': 0.5,
                    'max_hold_hours': int(cfg['max_hold']),
                    'risk_mult': 1.0,
                }
            }
            if best is None or sig['predicted_R'] > best['predicted_R']:
                best = sig
        return best

    def calculate_smc_signal(self, pair, df, lookback, pullback_pct, fvg_required, trend_filter):
        if df is None or df.empty or len(df) < (lookback + 210):
            return None
        d = df.copy().sort_index()
        d['vol_sma20'] = d.volume.rolling(20).mean()
        d['swing_low'] = d.low.shift(1).rolling(lookback).min()
        d['swing_high'] = d.high.shift(1).rolling(lookback).max()
        d['ema200'] = ema(d.close, 200)
        
        row = d.iloc[-1]
        prev2_row = d.iloc[-3]
        
        long_sweep = (row.low < row.swing_low) and (row.close > row.swing_low)
        short_sweep = (row.high > row.swing_high) and (row.close < row.swing_high)
        
        if not long_sweep and not short_sweep:
            return None
            
        # Layer 3: Volume Capitulation Footprint
        if row.volume < 1.25 * row.vol_sma20:
            return None
            
        # Layer 4: Fair Value Gap (FVG) Check
        if fvg_required:
            fvg_long = row.low > prev2_row.high
            fvg_short = row.high < prev2_row.low
            if long_sweep and not fvg_long:
                long_sweep = False
            if short_sweep and not fvg_short:
                short_sweep = False
                
        # Layer 3 (Alternative): EMA-200 Trend Alignment
        if trend_filter:
            if long_sweep and row.close < row.ema200:
                long_sweep = False
            if short_sweep and row.close > row.ema200:
                short_sweep = False
                
        if not long_sweep and not short_sweep:
            return None
            
        side = 'LONG' if long_sweep else 'SHORT'
        
        d['atr_val'] = atr(d, 14)
        atr_now = float(d['atr_val'].iloc[-1])
        if not np.isfinite(atr_now) or atr_now <= 0:
            atr_now = row.close * 0.01
            
        # Volatility-Adjusted Breathing Stops from the actual close entry price
        # This aligns the entry price with the paper engine execution and prevents premature stop-outs
        entry_price = float(row.close)
        
        if side == 'LONG':
            # Use 3.0x ATR for majors, 3.5x ATR for mid-vol to survive double sweeps
            sl_atr_mult = 3.0 if pair in ['BTC', 'ETH'] else 3.5
            sl = entry_price - (sl_atr_mult * atr_now)
            tp = entry_price + (1.5 * atr_now)
        else:
            sl_atr_mult = 3.0 if pair in ['BTC', 'ETH'] else 3.5
            sl = entry_price + (sl_atr_mult * atr_now)
            tp = entry_price - (1.5 * atr_now)
            
        # Layer 6: Risk-to-Reward Safety Checks
        risk = abs(entry_price - sl)
        reward = abs(tp - entry_price)
        if risk <= 0 or reward <= 0:
            return None
            
        risk_pct = float(risk / entry_price)
        
        return {
            'pair': pair,
            'side': side,
            'setup': 'SMC_SWEEP_RECLAIM',
            'probability': 0.82,
            'predicted_R': 0.8,
            'entry': entry_price,
            'sl': float(sl),
            'tp': float(tp),
            'risk_pct': risk_pct,
            'trigger_time': str(d.index[-1]),
            'signal_time': str(d.index[-1]),
            'risk_mult': 1.0,
            'max_hold_hours': 24,
            'meta': {
                'family': 'SMC_SWEEP_RECLAIM',
                'prob_th': 0.70,
                'er_th': 0.0,
                'decision_makers': {'lookback': lookback, 'trend_filter': trend_filter},
                'predicted_R': 0.8,
                'max_hold_hours': 24,
                'risk_mult': 1.0,
            }
        }

    def latest_signal(self, pair, df15=None, df1h=None, df4h=None, rr=None):
        p_upper = pair.upper().strip()
        
        # 1. Category 1: Majors -> SMC Retracement (60%) on 15m (Wider 36-bar lookback window)
        if p_upper in ['BTC', 'ETH']:
            return self.calculate_smc_signal(p_upper, df4h, lookback=36, pullback_pct=60, fvg_required=False, trend_filter=True)
            
        # 2. Category 2: Mid-Vol -> SMC Pullback (30%) on 15m (Wider 36-bar lookback window)
        elif p_upper in ['SOL', 'AVAX', 'BNB', 'LINK', 'NEAR']:
            return self.calculate_smc_signal(p_upper, df4h, lookback=36, pullback_pct=30, fvg_required=False, trend_filter=True)
            
        # 3. Category 3: High-Vol -> CCI_BB_SCALPER on 5m
        elif p_upper in ['HYPE', 'PEPE', 'WIF', 'FET']:
            return self.calculate_cci_bb_signal(p_upper, df4h)
            
        else:
            return self.calculate_cci_bb_signal(p_upper, df4h)
