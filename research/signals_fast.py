"""Vectorized signal layer (fast path) with a strict-mode parity harness.

The strategy engines in strategy.py evaluate ONE growing window per bar —
correct but O(bars × window) and too slow for multi-year research across 12
pairs. This module computes the IDENTICAL signal logic vectorized over the
full frame (same helper functions from strategy.py: ema/cci/atr/bollinger), so
research runs take seconds instead of hours.

Parity discipline: ewm/rolling on a full series vs a 420-bar window can differ
microscopically near indicator warm-up. `parity_report()` quantifies the real
disagreement rate against the strict path, and the recursive-analysis audit
(whose whole job is warmup-sensitivity detection) is the second safety net.
Divergence beyond tolerance => run --strict and distrust fast-path numbers.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from strategy import (
    StrategyBrain, ema, cci, atr, bollinger_bands,
    CATEGORY_SMC, CATEGORY_SCALPER, SCALPER_ASSET_CONFIG,
)

SMC_LOOKBACK = 36
SMC_MAX_HOLD = 24


def compute_signals(pair: str, df: pd.DataFrame) -> pd.DataFrame:
    """Per-bar signal frame: bool flag + full trade geometry, identical to
    StrategyBrain outputs (entry at bar close; 1.5xATR SL / 2.0xATR TP)."""
    p = pair.upper().strip()
    close, low, high, vol = df.close, df.low, df.high, df.volume
    out = pd.DataFrame(index=df.index)
    out['signal'] = False
    out['side'] = ''
    out['sl'] = np.nan
    out['tp'] = np.nan
    out['risk_pct'] = np.nan
    out['max_hold_hours'] = np.nan
    out['setup'] = ''

    atr14 = atr(df, 14)

    if p in CATEGORY_SMC:
        swing_low = low.shift(1).rolling(SMC_LOOKBACK).min()
        swing_high = high.shift(1).rolling(SMC_LOOKBACK).max()
        vol_sma20 = vol.rolling(20).mean()
        ema200 = ema(close, 200)
        volboom = vol >= 1.25 * vol_sma20

        long = (low < swing_low) & (close > swing_low) & volboom & (close > ema200)
        short = (high > swing_high) & (close < swing_high) & volboom & (close < ema200)
        side = np.where(long, 'LONG', np.where(short, 'SHORT', ''))
        mask = long | short
        entry = close
        sl = np.where(long, entry - 1.5 * atr14, entry + 1.5 * atr14)
        tp = np.where(long, entry + 2.0 * atr14, entry - 2.0 * atr14)
        out.loc[mask, 'signal'] = True
        out.loc[mask, 'side'] = side[mask]
        out.loc[mask, 'sl'] = sl[mask]
        out.loc[mask, 'tp'] = tp[mask]
        out.loc[mask, 'risk_pct'] = (1.5 * atr14 / entry)[mask]
        out.loc[mask, 'max_hold_hours'] = SMC_MAX_HOLD
        out.loc[mask, 'setup'] = 'SMC_SWEEP_RECLAIM'

    elif p in CATEGORY_SCALPER:
        cci14 = cci(df, 14)
        ema100 = ema(close, 100)
        combined = None
        for side in ('LONG', 'SHORT'):
            cfg = SCALPER_ASSET_CONFIG.get((p, side))
            if not cfg:
                continue
            upper, mid, lower = bollinger_bands(close, 20, cfg['stdev'])
            if side == 'LONG':
                cond = (close < lower) & (cci14 <= cfg['cci_th'])
                if cfg['trend_filter']:
                    cond &= close >= ema100
            else:
                cond = (close > upper) & (cci14 >= cfg['cci_th'])
                if cfg['trend_filter']:
                    cond &= close <= ema100
            cond = cond.fillna(False)
            entry = close
            slx = entry - 1.5 * atr14 if side == 'LONG' else entry + 1.5 * atr14
            tpx = entry + 2.0 * atr14 if side == 'LONG' else entry - 2.0 * atr14
            m = cond & ~out['signal']
            out.loc[m, ['signal']] = True
            out.loc[m, 'side'] = side
            out.loc[m, 'sl'] = slx[m]
            out.loc[m, 'tp'] = tpx[m]
            out.loc[m, 'risk_pct'] = (1.5 * atr14 / entry)[m]
            out.loc[m, 'max_hold_hours'] = int(cfg['max_hold'])
            out.loc[m, 'setup'] = 'CCI_BB_SCALPER'
            combined = cond if combined is None else (combined | cond)

    out.loc[~np.isfinite(atr14), 'signal'] = False
    ok = np.isfinite(out['sl']) & np.isfinite(out['tp'])
    out.loc[~ok, 'signal'] = False
    return out


def parity_report(pair: str, df: pd.DataFrame, n_samples=200, warmup=420, seed=3) -> dict:
    """Strict-vs-fast comparison at random bars. The strict path calls the real
    StrategyBrain on a trailing window — exactly what the live bot computes."""
    brain = StrategyBrain()
    fast = compute_signals(pair, df)
    st = np.random.default_rng(seed)
    idx = st.choice(np.arange(warmup, len(df)), size=min(n_samples, max(1, len(df) - warmup)), replace=False)
    total, agree, flipped = 0, 0, []
    for i in idx:
        window = df.iloc[i - warmup:i + 1]  # noqa: E203
        strict = brain.latest_signal(pair, window)
        strict_side = strict['side'] if strict else ''
        fast_side = fast.side.iat[i] if fast.signal.iat[i] else ''
        total += 1
        if strict_side == fast_side:
            agree += 1
            if strict and fast_side:
                rel = abs(strict['sl'] - fast.sl.iat[i]) / max(strict['sl'], 1e-9)
                if rel > 2e-3:  # windowed-ewm residue beyond tolerance on LEVELS
                    flipped.append((int(i), 'level_drift', round(rel, 5)))
        else:
            flipped.append((int(i), strict_side or 'none', fast_side or 'none'))
    rate = agree / total if total else 1.0
    return {'pair': pair, 'samples': total, 'agreement': round(rate, 4),
            'mismatches': flipped[:10]}
