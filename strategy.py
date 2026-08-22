"""
Strategy brain for the paper bot.

Phase 0 (honesty pass):
- Purged all dead indicator/ML scaffolding that nothing called (HMA, MFI, CMF,
  ADX, RSI, efficiency, choppiness, sigmoid, ...). What remains is exactly the
  two engines that actually trade.
- Fixed the routing bug: SUI/APT/DOGE previously fell through to an
  undocumented CCI_BB fallback with an INVERTED risk/reward (TP 1.5xATR vs
  SL 3.0xATR) and no trend filter. Pairs are now assigned to explicit
  categories; uncategorized pairs are REJECTED instead of silently traded.
- Removed the dead pullback_pct parameter (was accepted, never used).
- Single source of truth for pair categories + timeframe mapping, shared with
  paper_engine.py so the two files can no longer drift apart.
"""
import math

import numpy as np
import pandas as pd

# -----------------------------
# Indicator helpers (only what is actually used)
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

def atr(df, n=14):
    pc = df.close.shift(1)
    tr = pd.concat([
        (df.high - df.low).abs(),
        (df.high - pc).abs(),
        (df.low - pc).abs(),
    ], axis=1).max(axis=1)
    return tr.ewm(alpha=1/n, adjust=False, min_periods=n).mean()

# -----------------------------
# Pair universe — single source of truth.
# paper_engine.py imports timeframe_for_pair() from here. Keep in sync with
# config.py's default PAIRS list.
# -----------------------------
CATEGORY_SMC = {'BTC', 'ETH', 'SOL', 'AVAX', 'BNB', 'LINK', 'NEAR', 'SUI', 'APT'}
CATEGORY_SCALPER = {'HYPE', 'PEPE', 'WIF', 'FET', 'DOGE'}

SMC_TIMEFRAME = '15m'
SCALPER_TIMEFRAME = '5m'

def timeframe_for_pair(pair):
    """Candle timeframe the strategy logic expects for this pair."""
    p = pair.upper().strip()
    if p in CATEGORY_SCALPER:
        return SCALPER_TIMEFRAME
    if p in CATEGORY_SMC:
        return SMC_TIMEFRAME
    return None  # unknown pair -> rejected upstream

# Symmetrical 1.33:1 reward-to-risk per-asset scalper tuning.
SCALPER_ASSET_CONFIG = {
    ('BTC', 'LONG'):  {'stdev': 2.0, 'cci_th': -135, 'trend_filter': True, 'tp_atr': 2.0, 'sl_atr': 1.5, 'max_hold': 12},
    ('BTC', 'SHORT'): {'stdev': 2.2, 'cci_th': 140,  'trend_filter': True, 'tp_atr': 2.0, 'sl_atr': 1.5, 'max_hold': 12},
    ('ETH', 'LONG'):  {'stdev': 2.0, 'cci_th': -130, 'trend_filter': True, 'tp_atr': 2.0, 'sl_atr': 1.5, 'max_hold': 12},
    ('ETH', 'SHORT'): {'stdev': 2.3, 'cci_th': 135,  'trend_filter': True, 'tp_atr': 2.0, 'sl_atr': 1.5, 'max_hold': 12},
    ('SOL', 'LONG'):  {'stdev': 2.2, 'cci_th': -140, 'trend_filter': True, 'tp_atr': 2.0, 'sl_atr': 1.5, 'max_hold': 12},
    ('SOL', 'SHORT'): {'stdev': 2.4, 'cci_th': 145,  'trend_filter': True, 'tp_atr': 2.0, 'sl_atr': 1.5, 'max_hold': 12},
    ('AVAX', 'LONG'): {'stdev': 2.2, 'cci_th': -135, 'trend_filter': True, 'tp_atr': 2.0, 'sl_atr': 1.5, 'max_hold': 12},
    ('AVAX', 'SHORT'):{'stdev': 2.3, 'cci_th': 135,  'trend_filter': True, 'tp_atr': 2.0, 'sl_atr': 1.5, 'max_hold': 12},
    ('BNB', 'LONG'):  {'stdev': 2.1, 'cci_th': -135, 'trend_filter': True, 'tp_atr': 2.0, 'sl_atr': 1.5, 'max_hold': 12},
    ('BNB', 'SHORT'): {'stdev': 2.2, 'cci_th': 135,  'trend_filter': True, 'tp_atr': 2.0, 'sl_atr': 1.5, 'max_hold': 12},
    ('HYPE', 'LONG'): {'stdev': 2.5, 'cci_th': -150, 'trend_filter': True, 'tp_atr': 2.0, 'sl_atr': 1.5, 'max_hold': 12},
    ('HYPE', 'SHORT'):{'stdev': 2.5, 'cci_th': 150,  'trend_filter': True, 'tp_atr': 2.0, 'sl_atr': 1.5, 'max_hold': 12},
    ('LINK', 'LONG'): {'stdev': 2.0, 'cci_th': -130, 'trend_filter': False, 'tp_atr': 2.0, 'sl_atr': 1.5, 'max_hold': 12},
    ('LINK', 'SHORT'):{'stdev': 2.2, 'cci_th': 130,  'trend_filter': True, 'tp_atr': 2.0, 'sl_atr': 1.5, 'max_hold': 12},
    ('NEAR', 'LONG'): {'stdev': 2.2, 'cci_th': -140, 'trend_filter': True, 'tp_atr': 2.0, 'sl_atr': 1.5, 'max_hold': 12},
    ('NEAR', 'SHORT'):{'stdev': 2.3, 'cci_th': 140,  'trend_filter': True, 'tp_atr': 2.0, 'sl_atr': 1.5, 'max_hold': 12},
    ('PEPE', 'LONG'): {'stdev': 2.5, 'cci_th': -150, 'trend_filter': True, 'tp_atr': 2.0, 'sl_atr': 1.5, 'max_hold': 12},
    ('PEPE', 'SHORT'):{'stdev': 2.5, 'cci_th': 150,  'trend_filter': True, 'tp_atr': 2.0, 'sl_atr': 1.5, 'max_hold': 12},
    ('WIF', 'LONG'):  {'stdev': 2.5, 'cci_th': -150, 'trend_filter': True, 'tp_atr': 2.0, 'sl_atr': 1.5, 'max_hold': 12},
    ('WIF', 'SHORT'): {'stdev': 2.5, 'cci_th': 150,  'trend_filter': True, 'tp_atr': 2.0, 'sl_atr': 1.5, 'max_hold': 12},
    ('FET', 'LONG'):  {'stdev': 2.3, 'cci_th': -140, 'trend_filter': True, 'tp_atr': 2.0, 'sl_atr': 1.5, 'max_hold': 12},
    ('FET', 'SHORT'): {'stdev': 2.3, 'cci_th': 140,  'trend_filter': True, 'tp_atr': 2.0, 'sl_atr': 1.5, 'max_hold': 12},
    ('DOGE', 'LONG'): {'stdev': 2.3, 'cci_th': -140, 'trend_filter': True, 'tp_atr': 2.0, 'sl_atr': 1.5, 'max_hold': 12},
    ('DOGE', 'SHORT'):{'stdev': 2.3, 'cci_th': 140,  'trend_filter': True, 'tp_atr': 2.0, 'sl_atr': 1.5, 'max_hold': 12},
}


class StrategyBrain:
    def __init__(self, cfg=None, model_path=None, meta_path=None):
        self.cfg = cfg

    # ---------------------------------------------------------
    # Engine 2: CCI + Bollinger Band extreme mean-reversion scalper.
    # Returns None for any pair without an explicit tuned config —
    # there is deliberately NO fallback config (see Phase 0 notes).
    # ---------------------------------------------------------
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
                # No tuned config -> refuse to trade this pair/side.
                continue

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
                # Honest metadata: there is no probability model. This field is
                # a fixed weighting hint only, NOT a calibrated probability.
                'probability': 0.5,
                'predicted_R': 0.0,
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
                    'decision_makers': {
                        'cci': float(row.cci_val),
                        'close': entry,
                        'bb_lower': float(row.bb_lower),
                        'bb_upper': float(row.bb_upper),
                    },
                    'max_hold_hours': int(cfg['max_hold']),
                },
            }
            if best is None:
                best = sig
        return best

    # ---------------------------------------------------------
    # Engine 1: liquidity-sweep reclaim ("SMC").
    # LONG: low wicks below the N-bar swing low, candle closes back above it,
    # volume capitulation confirms, EMA-200 trend alignment. (Mirror for SHORT.)
    # ---------------------------------------------------------
    def calculate_smc_signal(self, pair, df, lookback=36, fvg_required=False, trend_filter=True):
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

        # Volume capitulation filter
        if row.volume < 1.25 * row.vol_sma20:
            return None

        # Optional fair-value-gap confirmation (disabled by default)
        if fvg_required:
            fvg_long = row.low > prev2_row.high
            fvg_short = row.high < prev2_row.low
            if long_sweep and not fvg_long:
                long_sweep = False
            if short_sweep and not fvg_short:
                short_sweep = False

        # EMA-200 trend alignment
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

        # Entry at close of the trigger candle (what the paper engine actually fills).
        entry_price = float(row.close)

        # 1.33:1 reward-to-risk volatility stops
        if side == 'LONG':
            sl = entry_price - (1.5 * atr_now)
            tp = entry_price + (2.0 * atr_now)
        else:
            sl = entry_price + (1.5 * atr_now)
            tp = entry_price - (2.0 * atr_now)

        risk = abs(entry_price - sl)
        reward = abs(tp - entry_price)
        if risk <= 0 or reward <= 0:
            return None

        risk_pct = float(risk / entry_price)

        return {
            'pair': pair,
            'side': side,
            'setup': 'SMC_SWEEP_RECLAIM',
            'probability': 0.5,
            'predicted_R': 0.0,
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
                'decision_makers': {'lookback': lookback, 'trend_filter': trend_filter},
                'max_hold_hours': 24,
            },
        }

    def latest_signal(self, pair, df):
        """Route the pair to its engine. Uncategorized pairs are REJECTED (None),
        never silently traded on an undocumented fallback config."""
        p_upper = pair.upper().strip()

        if p_upper in CATEGORY_SMC:
            return self.calculate_smc_signal(p_upper, df, lookback=36, fvg_required=False, trend_filter=True)
        elif p_upper in CATEGORY_SCALPER:
            return self.calculate_cci_bb_signal(p_upper, df)
        else:
            return None
