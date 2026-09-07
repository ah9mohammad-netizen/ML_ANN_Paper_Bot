"""Vol-targeted time-series momentum on daily bars — walk-forward.

This is the approach with actual published support in crypto (time-series
momentum / trend following), tested the same honest way as everything else:
signal from closed bars, applied next bar, costs on turnover, funding charged,
parameters chosen on a training window and scored only out of sample.
"""
import os, json, warnings, itertools
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import indicators as ta
import regimes as rg
import portfolio as pf
import okx_data as od

SYMS = ["BTC", "ETH", "SOL", "LINK", "XRP", "DOGE", "AVAX", "NEAR", "APT", "SUI"]
BARS = {"1d": 365, "8h": 365 * 3, "4h": 365 * 6}
RULE = {"1d": "1D", "8h": "8h", "4h": "4h"}


def load(tf="1d", start="2022-01-01"):
    px, vol = {}, {}
    for s in SYMS:
        f = f"data/SWAP_{s}_15m.parquet"
        if not os.path.exists(f):
            continue
        d = pd.read_parquet(f)
        if len(d) < 30000:
            continue
        r = ta.resample(d, RULE[tf]).loc[pd.Timestamp(start, tz="UTC"):]
        px[s] = r.close
    P = pd.DataFrame(px).sort_index()
    return P.dropna(how="all")


def funding_frame(index):
    out = {}
    for s in SYMS:
        try:
            f = od.funding(s, "2022-01-01")
        except Exception:
            continue
        if not len(f):
            continue
        # sum the 8h rates falling inside each bar
        r = f["rate"].resample("1D").sum().reindex(index).fillna(0.0)
        out[s] = r
    return pd.DataFrame(out).reindex(index).fillna(0.0) if out else None


def signals(P, kind, fast, slow, ret_win, breakout_n, vote):
    """Return -1/0/+1 per asset. All strictly causal."""
    if kind == "ma":
        sig = np.sign(ta.ema(P, fast) - ta.ema(P, slow))
    elif kind == "ret":
        sig = np.sign(P.pct_change(ret_win))
    elif kind == "breakout":
        hi = P.shift(1).rolling(breakout_n).max()
        lo = P.shift(1).rolling(breakout_n).min()
        raw = pd.DataFrame(0.0, index=P.index, columns=P.columns)
        raw[P > hi] = 1.0
        raw[P < lo] = -1.0
        sig = raw.replace(0.0, np.nan).ffill().fillna(0.0)
    elif kind == "vote":
        a = np.sign(ta.ema(P, fast) - ta.ema(P, slow))
        b = np.sign(P.pct_change(ret_win))
        hi = P.shift(1).rolling(breakout_n).max()
        lo = P.shift(1).rolling(breakout_n).min()
        c = pd.DataFrame(0.0, index=P.index, columns=P.columns)
        c[P > hi] = 1.0; c[P < lo] = -1.0
        c = c.replace(0.0, np.nan).ffill().fillna(0.0)
        tot = a.fillna(0) + b.fillna(0) + c
        sig = np.sign(tot) * (tot.abs() >= vote)
    else:
        raise ValueError(kind)
    return sig.fillna(0.0)


SPACE = dict(
    kind=["ma", "ret", "breakout", "vote"],
    fast=[10, 20, 30],
    slow=[50, 80, 120],
    ret_win=[20, 40, 60],
    breakout_n=[20, 40, 60],
    vote=[2, 3],
    target_vol=[0.30, 0.50],
    max_w=[0.25, 0.40],
    gross_cap=[1.0, 1.5, 2.0],
    long_only=[False, True],
)


def build(P, p, bars_per_year):
    sig = signals(P, p["kind"], p["fast"], p["slow"], p["ret_win"],
                  p["breakout_n"], p["vote"])
    if p["long_only"]:
        sig = sig.clip(lower=0)
    rv = np.log(P).diff().rolling(30).std() * np.sqrt(bars_per_year)
    return pf.vol_target_weights(sig, rv, p["target_vol"], p["max_w"],
                                 p["gross_cap"])


def score(st, n):
    if n < 60:
        return -9e9
    return st["sharpe"] - max(0.0, st["max_dd_pct"] / 100 - 0.30) * 5


def walk(P, F, bars_per_year, n_folds=10, train_days=420, test_days=120,
         cands=60, seed=1, fee=0.0005, slip=0.0002):
    import random
    keys = list(SPACE)
    combos = list(itertools.product(*[SPACE[k] for k in keys]))
    rnd = random.Random(seed)
    cand = [dict(zip(keys, c)) for c in rnd.sample(combos, min(cands, len(combos)))]

    t0, t1 = P.index[0], P.index[-1]
    folds, ts = [], t0 + pd.Timedelta(days=train_days)
    while ts + pd.Timedelta(days=test_days) <= t1 and len(folds) < n_folds:
        folds.append((ts - pd.Timedelta(days=train_days), ts,
                      ts + pd.Timedelta(days=test_days)))
        ts += pd.Timedelta(days=test_days)

    rows, segs = [], []
    for i, (a, b, c) in enumerate(folds):
        best, bs = None, -9e18
        for p in cand:
            W = build(P, p, bars_per_year)
            m = (P.index >= a) & (P.index < b)
            eq, st, net = pf.run(P[m], W[m], fee=fee, slip=slip, funding=None,
                                 bars_per_year=bars_per_year)
            s = score(st, m.sum())
            if s > bs:
                bs, best = s, p
        W = build(P, best, bars_per_year)
        m = (P.index >= b) & (P.index < c)
        Fm = F[m] if F is not None else None
        eq, st, net = pf.run(P[m], W[m], fee=fee, slip=slip, funding=Fm,
                             bars_per_year=bars_per_year)
        rows.append({"fold": i + 1, "test": str(b)[:10], **best,
                     "oos_sharpe": round(st["sharpe"], 2),
                     "oos_cagr": round(st["cagr_pct"], 1),
                     "oos_dd": round(st["max_dd_pct"], 1)})
        segs.append(net)
        print(f"   fold {i+1} {str(b)[:10]}→{str(c)[:10]}  "
              f"Sharpe {st['sharpe']:+.2f}  CAGR {st['cagr_pct']:+7.1f}%  "
              f"DD {st['max_dd_pct']:5.1f}%   {best['kind']} "
              f"f{best['fast']}/s{best['slow']}/r{best['ret_win']}/b{best['breakout_n']} "
              f"tv{best['target_vol']} g{best['gross_cap']} "
              f"{'LONG-ONLY' if best['long_only'] else 'L/S'}", flush=True)

    if not segs:
        return None, None
    net = pd.concat(segs).sort_index()
    net = net[~net.index.duplicated()]
    eq = 1000 * (1 + net).cumprod()
    dd = (eq.cummax() - eq) / eq.cummax()
    days = max((eq.index[-1] - eq.index[0]).days, 1)
    ann = np.sqrt(bars_per_year)
    st = {"start": str(eq.index[0])[:10], "end": str(eq.index[-1])[:10],
          "total_return_pct": 100 * (eq.iloc[-1] / 1000 - 1),
          "cagr_pct": 100 * ((eq.iloc[-1] / 1000) ** (365 / days) - 1),
          "vol_pct": 100 * net.std() * ann,
          "sharpe": net.mean() / net.std() * ann if net.std() > 0 else 0,
          "sortino": (net.mean() / net[net < 0].std() * ann
                      if net[net < 0].std() > 0 else 0),
          "max_dd_pct": 100 * dd.max(),
          "calmar": ((eq.iloc[-1] / 1000) ** (365 / days) - 1) / dd.max()
                    if dd.max() > 0 else np.inf,
          "avg_gross_exposure": np.nan, "turnover_ann": np.nan,
          "cost_drag_ann_pct": np.nan, "funding_drag_ann_pct": np.nan,
          "hit_rate_pct": 100 * (net > 0).mean(),
          "final_equity": eq.iloc[-1],
          "t_stat": net.mean() / (net.std() / np.sqrt(len(net))) if net.std() > 0 else 0}
    return st, (pd.DataFrame(rows), eq, net)


def main():
    reg = rg.btc_regime(pd.read_parquet("data/SWAP_BTC_15m.parquet"))
    for tf in ["1d", "8h"]:
        P = load(tf)
        bpy = BARS[tf]
        F = funding_frame(P.index) if tf == "1d" else None
        print(f"\n{'#'*70}\n# TS-MOMENTUM PORTFOLIO @ {tf}   "
              f"{P.shape[1]} assets, {len(P)} bars, "
              f"{str(P.index[0])[:10]}→{str(P.index[-1])[:10]}\n{'#'*70}", flush=True)
        st, extra = walk(P, F, bpy, n_folds=11, train_days=420,
                         test_days=110, cands=70)
        if st is None:
            print("  no folds"); continue
        print(); print(pf.fmt(st, f"TS-momentum @ {tf} · OUT-OF-SAMPLE"))
        folds, eq, net = extra
        print("\n  fold detail:"); print(folds.to_string(index=False))
        eq.to_frame("equity").to_csv(f"eq_mom_{tf}.csv")
        folds.to_csv(f"folds_mom_{tf}.csv", index=False)

        lab = reg["regime"].reindex(net.index.normalize(), method="ffill")
        g = pd.DataFrame({"ret": net.values, "mkt": lab.values}, index=net.index)
        rows = []
        for k, gg in g.groupby("mkt"):
            rows.append({"regime": k, "bars": len(gg),
                         "ann_ret%": round(100 * ((1 + gg.ret).prod() **
                                                  (bpy / len(gg)) - 1), 1),
                         "sharpe": round(gg.ret.mean() / gg.ret.std() *
                                         np.sqrt(bpy), 2) if gg.ret.std() > 0 else 0,
                         "hit%": round(100 * (gg.ret > 0).mean(), 1)})
        print("\n  by market regime:")
        print(pd.DataFrame(rows).to_string(index=False))

        # buy & hold BTC benchmark over the same OOS window
        b = P["BTC"].reindex(net.index).ffill()
        br = b.pct_change().fillna(0)
        beq = (1 + br).cumprod()
        bdd = (beq.cummax() - beq) / beq.cummax()
        print(f"\n  benchmark BTC buy&hold same window: "
              f"total {100*(beq.iloc[-1]-1):+.1f}%  "
              f"Sharpe {br.mean()/br.std()*np.sqrt(bpy):.2f}  "
              f"maxDD {100*bdd.max():.1f}%")


if __name__ == "__main__":
    main()
