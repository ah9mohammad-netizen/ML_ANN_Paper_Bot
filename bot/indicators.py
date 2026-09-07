"""Vectorised indicators. Every function is strictly causal: value at bar t
uses only bars <= t. No shift(-n) anywhere."""
import numpy as np
import pandas as pd


def ema(s, n):
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def sma(s, n):
    return s.rolling(n, min_periods=n).mean()


def rma(s, n):
    """Wilder's smoothing."""
    return s.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def true_range(df):
    pc = df.close.shift(1)
    return pd.concat([(df.high - df.low).abs(),
                      (df.high - pc).abs(),
                      (df.low - pc).abs()], axis=1).max(axis=1)


def atr(df, n=14):
    return rma(true_range(df), n)


def rsi(close, n=14):
    d = close.diff()
    au = rma(d.clip(lower=0), n)
    ad = rma((-d).clip(lower=0), n)
    return 100 - 100 / (1 + au / ad.replace(0, np.nan))


def adx(df, n=14):
    up = df.high.diff()
    dn = -df.low.diff()
    plus = up.where((up > dn) & (up > 0), 0.0)
    minus = dn.where((dn > up) & (dn > 0), 0.0)
    a = rma(true_range(df), n)
    pdi = 100 * rma(plus, n) / a
    mdi = 100 * rma(minus, n) / a
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return rma(dx, n), pdi, mdi


def cci(df, n=14):
    tp = (df.high + df.low + df.close) / 3
    m = tp.rolling(n, min_periods=n).mean()
    mad = tp.rolling(n, min_periods=n).apply(
        lambda x: np.abs(x - x.mean()).mean(), raw=True)
    return (tp - m) / (0.015 * mad.replace(0, np.nan))


def bbands(close, n=20, k=2.0):
    m = close.rolling(n, min_periods=n).mean()
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    return m + k * sd, m, m - k * sd


def bb_width(close, n=20, k=2.0):
    u, m, l = bbands(close, n, k)
    return (u - l) / m.replace(0, np.nan)


def donchian(df, n):
    """Prior-n-bar extremes, EXCLUDING the current bar (shift(1))."""
    return (df.high.shift(1).rolling(n, min_periods=n).max(),
            df.low.shift(1).rolling(n, min_periods=n).min())


def realized_vol(close, n=96):
    """Annualised-ish realized vol of log returns over n bars (unitless per-bar)."""
    r = np.log(close).diff()
    return r.rolling(n, min_periods=n // 2).std()


def efficiency_ratio(close, n=20):
    """Kaufman efficiency: |net move| / sum(|moves|). 1 = clean trend, 0 = chop."""
    net = (close - close.shift(n)).abs()
    path = close.diff().abs().rolling(n, min_periods=n).sum()
    return net / path.replace(0, np.nan)


def zscore(s, n):
    m = s.rolling(n, min_periods=n).mean()
    sd = s.rolling(n, min_periods=n).std(ddof=0)
    return (s - m) / sd.replace(0, np.nan)


def vwap_session(df, freq="1D"):
    """Rolling session VWAP, reset each period."""
    tp = (df.high + df.low + df.close) / 3
    g = df.index.floor(freq)
    pv = (tp * df.volume).groupby(g).cumsum()
    v = df.volume.groupby(g).cumsum()
    return pv / v.replace(0, np.nan)


def supertrend(df, n=10, mult=3.0):
    a = atr(df, n)
    hl2 = (df.high + df.low) / 2
    ub = hl2 + mult * a
    lb = hl2 - mult * a
    close = df.close.values
    ubv, lbv = ub.values, lb.values
    fu = np.full(len(df), np.nan)
    fl = np.full(len(df), np.nan)
    dirn = np.ones(len(df))
    for i in range(1, len(df)):
        if np.isnan(ubv[i]):
            continue
        fu[i] = ubv[i] if (ubv[i] < fu[i-1] or close[i-1] > fu[i-1]
                           or np.isnan(fu[i-1])) else fu[i-1]
        fl[i] = lbv[i] if (lbv[i] > fl[i-1] or close[i-1] < fl[i-1]
                           or np.isnan(fl[i-1])) else fl[i-1]
        if not np.isnan(fu[i-1]) and close[i] > fu[i-1]:
            dirn[i] = 1
        elif not np.isnan(fl[i-1]) and close[i] < fl[i-1]:
            dirn[i] = -1
        else:
            dirn[i] = dirn[i-1]
    return pd.Series(dirn, index=df.index), pd.Series(fu, index=df.index), pd.Series(fl, index=df.index)


def resample(df, rule):
    """Downsample OHLCV. label/closed='left' so a bar is stamped at its OPEN time
    and only completes at open+rule — caller must shift to avoid look-ahead."""
    o = df.resample(rule, label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min",
         "close": "last", "volume": "sum"})
    return o.dropna()


def align_htf(htf_series, ltf_index):
    """Map a higher-timeframe series onto a lower-timeframe index WITHOUT
    look-ahead: the HTF bar stamped at time T is only knowable after T+rule,
    so we shift it one HTF bar forward before reindexing."""
    s = htf_series.shift(1)
    return s.reindex(ltf_index, method="ffill")
