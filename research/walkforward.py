"""Walk-forward optimisation.

The only number that means anything is the one produced by parameters that were
chosen WITHOUT seeing the data they are scored on. Everything here is built
around that: parameters are fitted on a training window, then applied unchanged
to the next, unseen window. The stitched out-of-sample folds are the result.

Also reports parameter STABILITY across folds. If the best parameters jump
around between folds, there is no edge — only curve fitting.
"""
from __future__ import annotations
import itertools, math, random
from typing import Callable, Dict, List

import numpy as np
import pandas as pd

from engine import Backtester, RiskConfig, ExecConfig, build_report


def make_folds(index, n_folds=6, train_days=270, test_days=90, anchored=False):
    """Rolling (default) or anchored walk-forward windows."""
    t0, t1 = index[0], index[-1]
    span = (t1 - t0).days
    need = train_days + n_folds * test_days
    if span < need:                       # shrink to fit the data we have
        scale = span / need
        train_days = max(90, int(train_days * scale))
        test_days = max(21, int(test_days * scale))
    folds = []
    test_start = t0 + pd.Timedelta(days=train_days)
    while test_start + pd.Timedelta(days=test_days) <= t1 and len(folds) < n_folds:
        tr0 = t0 if anchored else test_start - pd.Timedelta(days=train_days)
        folds.append((tr0, test_start, test_start + pd.Timedelta(days=test_days)))
        test_start += pd.Timedelta(days=test_days)
    return folds


def slice_all(d: Dict[str, pd.DataFrame], a, b):
    return {k: v.loc[(v.index >= a) & (v.index < b)] for k, v in d.items()
            if len(v.loc[(v.index >= a) & (v.index < b)]) > 80}


def score(rep, min_trades=25, dd_cap=0.35):
    """Objective: risk-adjusted, sample-size aware, drawdown penalised.
    Deliberately NOT total return — that rewards luck and leverage."""
    n = rep.get("n_trades", 0)
    if n < min_trades:
        return -9e9
    exp = rep.get("expectancy_R", 0.0)
    sd = rep.get("std_R", 1.0) or 1.0
    t = exp / (sd / math.sqrt(n))                  # t-stat of the edge
    dd = rep.get("max_dd_pct", 100) / 100
    pen = 1.0 if dd <= dd_cap else (dd_cap / dd) ** 2
    return t * pen


def run_one(data, sigfn, params, risk, ex, funding, bar_minutes):
    sig = {}
    for s, d in data.items():
        try:
            sig[s] = sigfn(d, s, **params)
        except Exception:
            return None
    bt = Backtester(data, sig, risk=risk, ex=ex, funding=funding,
                    bar_minutes=bar_minutes)
    rep = bt.run()
    return rep, bt


def sample_grid(space: Dict[str, list], n, seed=0):
    keys = list(space)
    full = list(itertools.product(*[space[k] for k in keys]))
    rnd = random.Random(seed)
    if len(full) > n:
        full = rnd.sample(full, n)
    return [dict(zip(keys, c)) for c in full]


def walk_forward(data, sigfn, space, risk: RiskConfig, ex: ExecConfig,
                 funding=None, bar_minutes=15, n_folds=6, train_days=270,
                 test_days=90, n_candidates=40, min_trades_train=30,
                 anchored=False, verbose=True, seed=0):
    idx = None
    for d in data.values():
        idx = d.index if idx is None else idx.union(d.index)
    idx = idx.sort_values()
    folds = make_folds(idx, n_folds, train_days, test_days, anchored)
    cands = sample_grid(space, n_candidates, seed)
    if verbose:
        print(f"  walk-forward: {len(folds)} folds × {len(cands)} candidates "
              f"(train {train_days}d / test {test_days}d)", flush=True)

    chosen, oos_trades, oos_curves = [], [], []
    for fi, (tr0, te0, te1) in enumerate(folds):
        tr = slice_all(data, tr0, te0)
        te = slice_all(data, te0, te1)
        if not tr or not te:
            continue
        best, best_s = None, -9e18
        for p in cands:
            r = run_one(tr, sigfn, p, risk, ex, funding, bar_minutes)
            if not r:
                continue
            s = score(r[0], min_trades=min_trades_train)
            if s > best_s:
                best_s, best = s, p
        if best is None:
            if verbose:
                print(f"   fold {fi+1}: no candidate met the training bar — skipped")
            continue
        r = run_one(te, sigfn, best, risk, ex, funding, bar_minutes)
        if not r:
            continue
        rep, bt = r
        chosen.append({"fold": fi + 1, "train": str(tr0)[:10], "test": str(te0)[:10],
                       "score_train": round(best_s, 2), **best,
                       "oos_trades": rep.get("n_trades", 0),
                       "oos_pf": round(rep.get("profit_factor", 0), 2),
                       "oos_expR": round(rep.get("expectancy_R", 0), 3),
                       "oos_wr": round(rep.get("win_rate", 0), 1)})
        t = bt.trades_df()
        if len(t):
            t["fold"] = fi + 1
            oos_trades.append(t)
        oos_curves.append(bt.equity_df())
        if verbose:
            print(f"   fold {fi+1} test {str(te0)[:10]}→{str(te1)[:10]}  "
                  f"n={rep.get('n_trades',0):3d}  PF={rep.get('profit_factor',0):.2f}  "
                  f"expR={rep.get('expectancy_R',0):+.3f}  {best}", flush=True)

    if not oos_trades:
        return {"folds": pd.DataFrame(chosen), "report": {}, "trades": pd.DataFrame()}

    T = pd.concat(oos_trades, ignore_index=True)
    # stitch the fold equity curves into one continuous compounded curve
    eq = stitch(oos_curves, risk.starting_equity)
    rep = build_report(T, eq, risk.starting_equity, bar_minutes)
    rep["n_folds_used"] = len(chosen)
    return {"folds": pd.DataFrame(chosen), "report": rep, "trades": T,
            "equity": eq}


def stitch(curves, start):
    out, base = [], start
    for c in curves:
        if not len(c):
            continue
        rel = c.equity / c.equity.iloc[0]
        seg = rel * base
        out.append(seg)
        base = seg.iloc[-1]
    if not out:
        return pd.DataFrame({"equity": [start]},
                            index=[pd.Timestamp("2020-01-01", tz="UTC")])
    s = pd.concat(out)
    s = s[~s.index.duplicated(keep="last")].sort_index()
    return pd.DataFrame({"equity": s})


def stability(folds_df, keys):
    """How consistent were the winning parameters across folds?"""
    if folds_df.empty:
        return {}
    out = {}
    for k in keys:
        if k not in folds_df:
            continue
        v = folds_df[k]
        if pd.api.types.is_numeric_dtype(v):
            out[k] = {"mode": v.mode().iloc[0], "mean": round(v.mean(), 3),
                      "cv": round(v.std() / abs(v.mean()), 2) if v.mean() else None,
                      "n_distinct": v.nunique()}
        else:
            out[k] = {"mode": v.mode().iloc[0], "n_distinct": v.nunique()}
    return out
