"""Regime classification + trade slicing (Phase 2 workstream W6).

Every closed trade gets tagged with the market regime AT ENTRY TIME:
  - trend:      close vs EMA-200 on the traded timeframe
  - adx_bucket: ADX(14) — 'ranging' (<20), 'transitional' (20-25), 'trending' (>=25)
  - vol_bucket: quartile of the efficiency ratio (trend quality / noise proxy)

Output tables tell us WHEN each engine earns and when it bleeds — the evidence
base for regime-gated deployment in Phase 3.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _ema(s, n):
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def _adx(df, n=14):
    up = df.high.diff(); dn = -df.low.diff()
    plus = up.where((up > dn) & (up > 0), 0.0)
    minus = dn.where((dn > up) & (dn > 0), 0.0)
    pc = df.close.shift(1)
    tr = pd.concat([(df.high - df.low).abs(), (df.high - pc).abs(), (df.low - pc).abs()], axis=1).max(axis=1)
    atrn = tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    pdi = 100 * plus.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / atrn
    mdi = 100 * minus.ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / atrn
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def regime_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Per-bar regime labels for a candle frame."""
    out = pd.DataFrame(index=df.index)
    out['ema200'] = _ema(df.close, 200)
    out['adx14'] = _adx(df, 14)
    er_num = (df.close - df.close.shift(20)).abs()
    er_den = df.close.diff().abs().rolling(20).sum()
    out['eff_ratio'] = er_num / er_den.replace(0, np.nan)
    out['trend'] = np.where(df.close > out.ema200, 'up', 'down')
    out['adx_bucket'] = pd.cut(out.adx14, [-1, 20, 25, 100], labels=['ranging', 'transitional', 'trending'])
    try:
        out['vol_bucket'] = pd.qcut(out.eff_ratio, 4, labels=['choppy', 'low', 'high', 'directional'])
    except ValueError:
        out['vol_bucket'] = 'unknown'
    return out


def tag_trades(trades: pd.DataFrame, data: dict[str, pd.DataFrame]) -> pd.DataFrame:
    if trades.empty:
        return trades
    reg = {p: regime_frame(df) for p, df in data.items()}
    tagged = trades.copy()
    for col in ('trend', 'adx_bucket', 'vol_bucket'):
        vals = []
        for _, tr in tagged.iterrows():
            rf = reg.get(tr['pair'])
            if rf is None or tr['entry_time'] not in rf.index:
                vals.append('unknown')
                continue
            vals.append(str(rf.loc[tr['entry_time'], col]))
        tagged[col] = vals
    return tagged


def slice_table(trades: pd.DataFrame, by: str) -> pd.DataFrame:
    if trades.empty or by not in trades.columns:
        return pd.DataFrame()
    rows = []
    for key, g in trades.groupby(by, observed=True):
        w = g.loc[g.pnl > 0, 'pnl'].sum(); l = abs(g.loc[g.pnl <= 0, 'pnl'].sum())
        rows.append({
            by: key, 'trades': len(g),
            'win_rate': round(100 * (g.pnl > 0).mean(), 1),
            'net_pnl': round(float(g.pnl.sum()), 2),
            'PF': round(float(w / l), 2) if l > 0 else None,
            'avg_R': round(float(g.R_multiple.mean()), 3),
        })
    return pd.DataFrame(rows).sort_values('net_pnl', ascending=False)
