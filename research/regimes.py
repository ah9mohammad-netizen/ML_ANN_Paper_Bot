"""Market-regime labelling, driven by BTC daily structure (causal)."""
import numpy as np
import pandas as pd
import indicators as ta


def btc_regime(btc_15m, ret_win=60, er_win=30, thresh=0.10):
    """Label each UTC day bull / bear / chop from BTC daily structure.

    bull : trading above the 100d mean AND up meaningfully over 60d
    bear : below the 100d mean AND down meaningfully over 60d
    chop : everything else — directionless or conflicted

    Everything is shifted one day so a label is only ever applied to a day it
    could actually have been known at the start of.
    """
    d = ta.resample(btc_15m, "1D")
    e100 = ta.ema(d.close, 100)
    r = d.close.pct_change(ret_win)
    er = ta.efficiency_ratio(d.close, er_win)
    vol = np.log(d.close).diff().rolling(30).std() * np.sqrt(365)

    lab = pd.Series("chop", index=d.index, dtype=object)
    lab[(d.close > e100) & (r > thresh)] = "bull"
    lab[(d.close < e100) & (r < -thresh)] = "bear"
    lab = lab.shift(1)
    out = pd.DataFrame({"regime": lab, "er": er.shift(1),
                        "ann_vol": vol.shift(1), "ret": r.shift(1)})
    return out.dropna(subset=["regime"])


def label_index(reg_daily, index):
    """Broadcast the daily label onto an intraday index."""
    return reg_daily["regime"].reindex(index, method="ffill")


def segments(reg_daily, min_days=21):
    """Contiguous regime blocks of at least `min_days`, for reporting."""
    r = reg_daily["regime"]
    grp = (r != r.shift()).cumsum()
    out = []
    for _, g in r.groupby(grp):
        if len(g) >= min_days:
            out.append((g.iloc[0], g.index[0], g.index[-1], len(g)))
    return out


def summarise(reg_daily):
    c = reg_daily["regime"].value_counts()
    tot = c.sum()
    return {k: f"{v} days ({100*v/tot:.0f}%)" for k, v in c.items()}
