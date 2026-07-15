import math
import numpy as np
import pandas as pd

# -----------------------------
# Indicator helpers
# -----------------------------
def ema(s, n):
    return s.ewm(span=n, adjust=False, min_periods=n).mean()

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
# Final pair-direction configs
# -----------------------------
# Source summary:
# - BTC LONG: Phase 15 local optimizer, relative_eff + smooth
# - BTC SHORT: Phase 16 asymmetric short archetype
# - ETH LONG: Phase 16 asymmetric long archetype
# - ETH SHORT: Phase 15/previous accepted weak positive config
# - SOL LONG: Phase 15 local optimizer, no_relative + smooth
# - SOL SHORT: Phase 16 asymmetric short archetype
# - AVAX SHORT: Phase 16/14 accepted short config
# - AVAX LONG: present but disabled by default after individual backtest underperformed
# - BNB: watch only, disabled
PAIR_DIRECTION_CONFIG = {
    ('BTC', 'LONG'): {
        'enabled': True, 'family': 'relative_eff_long', 'prob_th': 0.60, 'er_th': 0.03,
        'tp_atr': 3.0, 'sl_atr': 2.0, 'max_hold_hours': 120, 'risk_mult': 1.00,
        'bias': -0.38,
        'weights': {
            'rel_btc_12': 0.85, 'rel_eth_12': 0.45, 'eff12': 0.80, 'eff30': 0.55,
            'dm_4h_trend': 0.90, 'dm_4h_slope': 0.60, 'dm_vwap': -0.30,
            'atr_ok': 0.55, 'vol_ok': 0.25, 'rsi_mid': 0.35,
        },
    },
    ('BTC', 'SHORT'): {
        'enabled': True, 'family': 'short_archetype', 'prob_th': 0.58, 'er_th': 0.00,
        'tp_atr': 2.0, 'sl_atr': 1.5, 'max_hold_hours': 72, 'risk_mult': 0.75,
        'bias': -0.45,
        'weights': {
            'short_breakdown': 1.05, 'short_rejection': 0.75, 'short_sweep_reject': 0.55,
            'dm_d1_slope': 0.55, 'dm_4h_slope': 0.65, 'dm_pdi': 0.55,
            'dm_vwap': -0.25, 'eff12': 0.50, 'atr_ok': 0.40, 'vol_ok': 0.20,
        },
    },
    ('ETH', 'LONG'): {
        'enabled': True, 'family': 'long_archetype', 'prob_th': 0.58, 'er_th': 0.00,
        'tp_atr': 2.0, 'sl_atr': 1.5, 'max_hold_hours': 72, 'risk_mult': 1.00,
        'bias': -0.42,
        'weights': {
            'long_trend_pullback': 0.85, 'long_sweep_reclaim': 0.65, 'long_compression_expansion': 0.50,
            'dm_d1_trend': 0.50, 'dm_4h_trend': 0.65, 'dm_rsi': 0.55,
            'dm_vwap': -0.35, 'eff12': 0.75, 'chop_ok': 0.45, 'atr_ok': 0.45,
        },
    },
    ('ETH', 'SHORT'): {
        'enabled': True, 'family': 'no_trigger_short', 'prob_th': 0.56, 'er_th': 0.03,
        'tp_atr': 3.0, 'sl_atr': 2.0, 'max_hold_hours': 120, 'risk_mult': 0.35,
        'bias': -0.55,
        'weights': {
            'dm_d1_slope': 0.65, 'dm_4h_slope': 0.75, 'dm_pdi': 0.65,
            'dm_vwap': -0.25, 'dm_rsi': 0.35, 'eff12': 0.55,
            'chop_ok': 0.35, 'atr_ok': 0.45, 'vol_ok': 0.25,
        },
    },
    ('SOL', 'LONG'): {
        'enabled': True, 'family': 'no_relative_long', 'prob_th': 0.60, 'er_th': 0.05,
        'tp_atr': 3.0, 'sl_atr': 2.0, 'max_hold_hours': 120, 'risk_mult': 0.50,
        'bias': -0.48,
        'weights': {
            'dm_4h_trend': 0.95, 'dm_4h_slope': 0.80, 'dm_pdi': 0.45,
            'dm_vwap': -0.35, 'dm_rsi': 0.45, 'eff12': 0.85,
            'eff30': 0.45, 'chop_ok': 0.45, 'atr_ok': 0.55, 'vol_ok': 0.15,
        },
    },
    ('SOL', 'SHORT'): {
        'enabled': True, 'family': 'short_archetype', 'prob_th': 0.58, 'er_th': 0.00,
        'tp_atr': 2.0, 'sl_atr': 1.5, 'max_hold_hours': 72, 'risk_mult': 0.75,
        'bias': -0.45,
        'weights': {
            'short_breakdown': 0.95, 'short_rejection': 0.65, 'short_sweep_reject': 0.60,
            'dm_4h_slope': 0.75, 'dm_pdi': 0.60, 'dm_vwap': -0.30,
            'eff12': 0.55, 'atr_ok': 0.40, 'vol_ok': 0.25,
        },
    },
    ('AVAX', 'LONG'): {
        'enabled': False, 'family': 'no_trigger_long_watch', 'prob_th': 0.65, 'er_th': 0.10,
        'tp_atr': 2.0, 'sl_atr': 1.5, 'max_hold_hours': 72, 'risk_mult': 0.0,
        'bias': -0.80,
        'weights': {'dm_4h_trend': 0.55, 'eff12': 0.55, 'atr_ok': 0.45},
    },
    ('AVAX', 'SHORT'): {
        'enabled': True, 'family': 'short_archetype', 'prob_th': 0.56, 'er_th': 0.00,
        'tp_atr': 2.0, 'sl_atr': 1.5, 'max_hold_hours': 72, 'risk_mult': 0.35,
        'bias': -0.46,
        'weights': {
            'short_breakdown': 0.85, 'short_rejection': 0.80, 'short_sweep_reject': 0.65,
            'dm_4h_slope': 0.65, 'dm_pdi': 0.55, 'dm_vwap': -0.25,
            'eff12': 0.45, 'atr_ok': 0.35, 'vol_ok': 0.20,
        },
    },
    ('BNB', 'LONG'): {'enabled': False, 'family': 'watch', 'prob_th': 0.70, 'er_th': 0.30, 'tp_atr': 2.0, 'sl_atr': 1.5, 'max_hold_hours': 72, 'risk_mult': 0.0, 'bias': -2.0, 'weights': {}},
    ('BNB', 'SHORT'): {'enabled': False, 'family': 'watch', 'prob_th': 0.70, 'er_th': 0.30, 'tp_atr': 2.0, 'sl_atr': 1.5, 'max_hold_hours': 72, 'risk_mult': 0.0, 'bias': -2.0, 'weights': {}},
}

class StrategyBrain:
    """
    Config-driven MTF local-opt strategy for paper trading.

    This replaces the previous ANN artifact model.  It uses the same bot shell
    (Railway worker, Telegram, paper DB) but uses our final researched
    pair-direction configs and a transparent weighted decision-maker score.
    """
    def __init__(self, cfg=None, model_path=None, meta_path=None):
        self.cfg = cfg
        self.configs = PAIR_DIRECTION_CONFIG

    def add_4h_features(self, df):
        d = df.copy().sort_index()
        for n in [10, 20, 50, 100, 200]:
            d[f'ema{n}'] = ema(d.close, n)
        d['ema10_slope'] = d.ema10.pct_change(3)
        d['ema20_slope'] = d.ema20.pct_change(4)
        d['ema50_slope'] = d.ema50.pct_change(8)
        d['ema200_slope'] = d.ema200.pct_change(24)
        d['rsi14'] = rsi(d.close)
        d['atr14'] = atr(d)
        d['atr_pct'] = d.atr14 / d.close
        d['adx14'], d['pdi14'], d['mdi14'] = adx(d)
        mid = d.close.rolling(20).mean(); sd = d.close.rolling(20).std()
        d['bb_width_pct'] = 4 * sd / d.close
        d['vol_sma20'] = d.volume.rolling(20).mean()
        d['vol_ratio20'] = d.volume / d.vol_sma20.replace(0, np.nan)
        typ = (d.high + d.low + d.close) / 3
        d['vwap42'] = (typ * d.volume).rolling(42).sum() / d.volume.rolling(42).sum().replace(0, np.nan)
        d['dist_vwap_atr'] = (d.close - d.vwap42) / d.atr14.replace(0, np.nan)
        d['eff12'] = efficiency(d.close, 12)
        d['eff30'] = efficiency(d.close, 30)
        d['chop12'] = chop_proxy(d, 12)
        d['chop30'] = chop_proxy(d, 30)
        d['atr_rank250'] = d.atr_pct.rolling(250, min_periods=80).rank(pct=True)
        d['bb_width_rank250'] = d.bb_width_pct.rolling(250, min_periods=80).rank(pct=True)
        d['rh12'] = d.high.shift(1).rolling(12).max()
        d['rl12'] = d.low.shift(1).rolling(12).min()
        d['cross_ema10_up'] = cross_above(d.close, d.ema10)
        d['cross_ema10_dn'] = cross_below(d.close, d.ema10)
        d['vwap_up'] = cross_above(d.close, d.vwap42)
        d['vwap_dn'] = cross_below(d.close, d.vwap42)
        rng = (d.high - d.low).replace(0, np.nan)
        d['body_ratio'] = (d.close - d.open).abs() / rng
        d['lower_wick'] = (d[['open','close']].min(axis=1) - d.low) / rng
        d['upper_wick'] = (d.high - d[['open','close']].max(axis=1)) / rng
        return d

    def add_daily_features(self, d4):
        dd = d4[['open','high','low','close','volume']].resample('1D').agg({
            'open':'first','high':'max','low':'min','close':'last','volume':'sum'
        }).dropna()
        dd = self.add_4h_features(dd)
        dd = dd.add_prefix('d1_')
        return pd.merge_asof(
            d4.sort_index(), dd.sort_index(),
            left_index=True, right_index=True, direction='backward'
        )

    def rel_features(self, pair, d):
        # Live bot intentionally uses pair-local features only.  Relative BTC/ETH
        # features from research are represented indirectly through family configs.
        # If later desired, the engine can pass BTC/ETH reference data into this method.
        d['rel_btc_12'] = 0.0
        d['rel_btc_30'] = 0.0
        d['rel_eth_12'] = 0.0
        d['rel_eth_30'] = 0.0
        return d

    def decision_makers(self, row, side):
        d = 1 if side == 'LONG' else -1
        dm = {}
        dm['dm_d1_trend'] = clamp(d * (row.d1_close - row.d1_ema100) / row.d1_close, -0.20, 0.20) * 8
        dm['dm_d1_slope'] = clamp(d * row.d1_ema50_slope, -0.08, 0.08) * 18
        dm['dm_4h_trend'] = clamp(d * (row.close - row.ema100) / row.close, -0.20, 0.20) * 8
        dm['dm_4h_slope'] = clamp(d * row.ema50_slope, -0.08, 0.08) * 18
        dm['dm_pdi'] = clamp(d * (row.pdi14 - row.mdi14) / 25.0, -2, 2)
        dm['dm_vwap'] = clamp(d * row.dist_vwap_atr / 3.0, -2, 2)
        dm['dm_rsi'] = clamp(((row.rsi14 if d == 1 else 100 - row.rsi14) - 50) / 20.0, -2, 2)
        dm['eff12'] = clamp((row.eff12 - 0.10) / 0.35, -1, 2)
        dm['eff30'] = clamp((row.eff30 - 0.08) / 0.35, -1, 2)
        dm['chop_ok'] = clamp((7.0 - row.chop12) / 4.0, -1, 2)
        dm['atr_ok'] = clamp((0.95 - row.atr_rank250) / 0.50, -1, 2)
        dm['vol_ok'] = clamp((row.vol_ratio20 - 0.25) / 2.0, -1, 2)
        dm['rsi_mid'] = 1.0 if 35 <= row.rsi14 <= 65 else -0.4
        dm['rel_btc_12'] = 0.0
        dm['rel_eth_12'] = 0.0
        dm['long_trend_pullback'] = 1.0 if side == 'LONG' and (row.cross_ema10_up or row.vwap_up) and row.eff12 > 0.10 else 0.0
        dm['long_sweep_reclaim'] = 1.0 if side == 'LONG' and row.low < row.rl12 and row.close > row.rl12 and row.lower_wick > 0.30 else 0.0
        dm['long_compression_expansion'] = 1.0 if side == 'LONG' and row.bb_width_rank250 < 0.45 and row.close > row.rh12 and row.body_ratio > 0.40 else 0.0
        dm['short_breakdown'] = 1.0 if side == 'SHORT' and (row.cross_ema10_dn or row.vwap_dn or row.close < row.ema20) and row.eff12 > 0.10 else 0.0
        dm['short_rejection'] = 1.0 if side == 'SHORT' and row.high > row.ema20 and row.close < row.ema20 and row.upper_wick > 0.28 else 0.0
        dm['short_sweep_reject'] = 1.0 if side == 'SHORT' and row.high > row.rh12 and row.close < row.rh12 and row.upper_wick > 0.32 else 0.0
        return dm

    def score_config(self, cfg, dm):
        z = cfg.get('bias', -0.5)
        for k, w in cfg.get('weights', {}).items():
            z += w * dm.get(k, 0.0)
        prob = sigmoid(z)
        # Expected R proxy. This is deliberately conservative and must be paper-validated.
        pred_r = (prob - 0.50) * (cfg['tp_atr'] / cfg['sl_atr']) * 2.0 - 0.03
        return prob, pred_r

    def latest_signal(self, pair, df15=None, df1h=None, df4h=None, rr=None):
        if df4h is None or df4h.empty:
            return None
        d = self.add_4h_features(df4h)
        d = self.add_daily_features(d)
        d = self.rel_features(pair, d)
        d = d.dropna().copy()
        if len(d) < 260:
            return None
        row = d.iloc[-1]
        entry = float(row.close)  # paper entry is current/last confirmed 4h close approximation
        best = None
        for side in ['LONG', 'SHORT']:
            cfg = self.configs.get((pair, side))
            if not cfg or not cfg.get('enabled', False):
                continue
            dm = self.decision_makers(row, side)
            prob, pred_r = self.score_config(cfg, dm)
            if prob < cfg['prob_th'] or pred_r < cfg['er_th']:
                continue
            risk_abs = cfg['sl_atr'] * float(row.atr14)
            target_abs = cfg['tp_atr'] * float(row.atr14)
            if risk_abs <= 0:
                continue
            if side == 'LONG':
                sl = entry - risk_abs
                tp = entry + target_abs
            else:
                sl = entry + risk_abs
                tp = entry - target_abs
            sig = {
                'pair': pair,
                'side': side,
                'setup': cfg['family'],
                'probability': float(prob),
                'predicted_R': float(pred_r),
                'entry': entry,
                'sl': float(sl),
                'tp': float(tp),
                'risk_pct': float(risk_abs / entry),
                'trigger_time': str(d.index[-1]),
                'signal_time': str(d.index[-1]),
                'risk_mult': float(cfg.get('risk_mult', 1.0)),
                'max_hold_hours': int(cfg.get('max_hold_hours', 120)),
                'meta': {
                    'family': cfg['family'],
                    'prob_th': cfg['prob_th'],
                    'er_th': cfg['er_th'],
                    'decision_makers': {k: round(float(v), 4) for k, v in dm.items()},
                    'predicted_R': float(pred_r),
                    'max_hold_hours': int(cfg.get('max_hold_hours', 120)),
                    'risk_mult': float(cfg.get('risk_mult', 1.0)),
                }
            }
            if best is None or sig['predicted_R'] > best['predicted_R']:
                best = sig
        return best
