"""Synthetic multi-regime OHLCV generator.

PURPOSE: pipeline verification ONLY. Lets the whole research gate (backtester,
bias audits, regime slicing, reporting) execute and be unit-tested inside
sandboxes/CI where exchange APIs are unreachable. Synthetic results say
NOTHING about strategy edge — they prove the machinery works. Real evidence
comes from research/fetch_data.py output (frozen OKX data).

Process: regime-dependent stochastic volatility model with vol clustering,
fat-tailed shocks, asymmetric wicks, and volume that spikes on range expansion
(so liquidity-sweep signatures exist in the data to be detected).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REGIMES = {
    # name: (drift_per_bar, base_vol_per_bar, vol_cluster_persistence)
    'bull': (0.00025, 0.0022, 0.94),
    'chop': (0.0,     0.0011, 0.90),
    'bear': (-0.00030, 0.0035, 0.96),
}

def _gen_regime_prices(regime, n, start_price, seed):
    drift, base_vol, persist = REGIMES[regime]
    out = np.empty(n)
    price = start_price
    vol = base_vol
    st = np.random.default_rng(seed)
    for i in range(n):
        # vol clustering: AR(1) in log-vol around regime base
        vol = base_vol * np.exp(persist * np.log(max(vol, 1e-6) / base_vol) + 0.25 * st.normal())
        shock = st.standard_t(df=4) * 0.8  # fat tails
        r = drift + vol * shock
        price = max(price * (1 + r), start_price * 0.01)
        out[i] = price
    return out

def generate_pair_history(pair, tf, regime_blocks, seed=7):
    """regime_blocks: list of (regime_name, n_bars). Returns OHLCV DataFrame."""
    freq = {'5m': '5min', '15m': '15min'}[tf]
    vol_scale = 1.0 if pair in ('BTC', 'ETH') else 2.2  # memes move harder
    closes_all = []
    price = 100.0 * (1 if pair in ('BTC', 'ETH') else 3.0)
    for k, (regime, n) in enumerate(regime_blocks):
        c = _gen_regime_prices(regime, n, price, seed + k * 131)
        closes_all.append(c)
        price = c[-1]
    close = np.concatenate(closes_all)
    n = len(close)

    st = np.random.default_rng(seed + 999)
    prev_close = np.concatenate([[close[0]], close[:-1]])
    open_ = prev_close * (1 + st.normal(0, 0.0004, n))
    # Wicks: small fraction of the bar's move plus a vol-scaled floor.
    # (Linear-in-move wicks produced absurd -30% lows under t-tail outliers;
    # real 15m wicks scale with volatility, ~10-25% of the bar body.)
    move = np.abs(close / np.maximum(prev_close, 1e-9) - 1)
    base_wick = 0.0008 * vol_scale + 0.15 * move
    up_wick = base_wick * (0.35 + st.exponential(0.9, n))
    dn_wick = base_wick * (0.35 + st.exponential(1.3, n))  # deeper down-wicks = sweep signature
    high = np.maximum(open_, close) * (1 + up_wick)
    low = np.minimum(open_, close) * (1 - dn_wick)
    volume = st.lognormal(9.2, 0.45, n) * (1 + 90 * move) / (1 if vol_scale == 1 else 3)

    ts = pd.date_range('2024-01-01', periods=n, freq=freq, tz='UTC')
    return pd.DataFrame(
        {'open': open_, 'high': high, 'low': low, 'close': close, 'volume': volume},
        index=ts,
    )

def regime_labels(regime_blocks, tf):
    freq_min = 5 if tf == '5m' else 15
    labels = []
    t = pd.Timestamp('2024-01-01', tz='UTC')
    for name, n in regime_blocks:
        labels += [(t, name)]
        t += pd.Timedelta(minutes=freq_min * n)
    return labels
