"""Re-run the single most promising configuration with the liquidation-order
bug fixed, plus the robustness checks that decide whether it is deployable:

  A. walk-forward, folds to 2026-08          (the honest headline)
  B. per-regime, per-symbol, per-side
  C. cost sensitivity
  D. parameter neighbourhood on the FULL sample
  E. randomised-entry control: same exits, shuffled entry days
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import strategies as st
import regimes as rg
import indicators as ta
from engine import Backtester, RiskConfig, ExecConfig, fmt
from walkforward import walk_forward, stability

SYMS = ["BTC", "ETH", "SOL", "LINK", "XRP", "DOGE", "AVAX", "NEAR", "APT", "SUI"]
TF, RULE, BARMIN = "8h", "8h", 480

RISK = RiskConfig(starting_equity=1000, risk_per_trade=0.0075, leverage=4,
                  max_open=5, max_gross_notional=2.0,
                  max_notional_per_trade=0.6, max_daily_loss=0.06,
                  max_drawdown_halt=0.35)
EX = ExecConfig(taker_fee=0.0005, slip_entry_bps=2.0, slip_stop_bps=5.0,
                use_funding=True, pessimistic_ambiguous_bar=True, chandelier=True)

SPACE = dict(entry_n=[20, 30, 48, 72], sl_atr=[2.0, 3.0],
             trail_atr=[2.0, 3.0, 4.5], trail_start_r=[0.0, 0.5],
             adx_min=[0, 18], ema_filter=[0, 200], max_hold_bars=[0])

_BTC = pd.read_parquet("data/SWAP_BTC_15m.parquet")
_f = {}


def mfilter(index):
    k = (index[0], index[-1], len(index))
    if k not in _f:
        _f[k] = st.btc_trend_filter(_BTC, index)
    return _f[k]


def fn(df, symbol="", **kw):
    return st.apply_market_filter(
        st.donchian_trend_v2(df, symbol, long_only=True, **kw),
        mfilter(df.index), "align")


def load():
    out = {}
    for s in SYMS:
        p = f"data/SWAP_{s}_15m.parquet"
        if not os.path.exists(p):
            continue
        d = pd.read_parquet(p)
        if len(d) < 30000:
            continue
        out[s] = ta.resample(d, RULE).loc[pd.Timestamp("2022-01-01", tz="UTC"):]
    return out


def tbl(t, reg, col):
    if col == "mkt":
        lab = reg["regime"].reindex(pd.DatetimeIndex(t.entry_ts).normalize(),
                                    method="ffill")
        t = t.copy(); t["mkt"] = lab.values
    elif col == "dir":
        t = t.copy(); t["dir"] = np.where(t.side > 0, "LONG", "SHORT")
    elif col == "year":
        t = t.copy(); t["year"] = pd.DatetimeIndex(t.entry_ts).year
    rows = []
    for k, g in t.groupby(col):
        w = g[g.pnl > 0]; l = g[g.pnl <= 0]; gl = abs(l.pnl.sum())
        rows.append({col: k, "n": len(g), "win%": round(100 * len(w) / len(g), 1),
                     "PF": round(w.pnl.sum() / gl, 2) if gl > 0 else 999,
                     "expR": round(g.r_multiple.mean(), 3),
                     "pnl": round(g.pnl.sum(), 1)})
    return pd.DataFrame(rows)


def main():
    reg = rg.btc_regime(_BTC)
    data = load()
    print(f"assets={list(data)}  bars={ {k: len(v) for k, v in data.items()} }\n",
          flush=True)

    print("#" * 70)
    print("# A. WALK-FORWARD, liquidation ordering fixed")
    print("#" * 70, flush=True)
    res = walk_forward(data, fn, SPACE, RISK, EX, bar_minutes=BARMIN,
                       n_folds=12, train_days=365, test_days=110,
                       n_candidates=36, min_trades_train=15, verbose=True)
    r = res["report"]
    if not r or not r.get("n_trades"):
        print("no trades"); return
    print(); print(fmt(r, "donchian@8h + BTC-align · OUT-OF-SAMPLE"))
    T = res["trades"]
    T.to_csv("out_best_LO.csv", index=False)
    res["folds"].to_csv("folds_best_LO.csv", index=False)
    res["equity"].to_csv("eq_best_LO.csv")

    print("\n" + "#" * 70); print("# B. BREAKDOWN"); print("#" * 70)
    for c in ["mkt", "year", "dir", "symbol"]:
        print(f"\n  by {c}:")
        print(tbl(T, reg, c).sort_values("pnl", ascending=False).to_string(index=False))

    print("\n  parameter stability:")
    print(json.dumps(stability(res["folds"], list(SPACE)), indent=1, default=str))

    # pick the most-selected joint configuration, not marginal modes
    f = res["folds"]
    keys = [k for k in SPACE if k in f]
    joint = f.groupby(keys).size().sort_values(ascending=False)
    best_joint = dict(zip(keys, joint.index[0]))
    best_joint = {k: (int(v) if float(v).is_integer() else float(v))
                  for k, v in best_joint.items()}
    print(f"\n  most-selected JOINT parameter set ({joint.iloc[0]}/{len(f)} folds):")
    print("   ", best_joint)
    json.dump({"strategy": "donchian_trend_v2", "timeframe": TF,
               "market_filter": "align", "params": best_joint},
              open("params_best_LO.json", "w"), indent=2)

    sig = {s: fn(d, s, **best_joint) for s, d in data.items()}

    print("\n" + "#" * 70); print("# C. COST SENSITIVITY (full sample, joint params)")
    print("#" * 70)
    for lab, fee, slip in [("maker 2bps", 0.0002, 1.0), ("taker 5bps", 0.0005, 2.0),
                           ("taker 7.5bps", 0.00075, 3.0), ("taker 10bps", 0.0010, 4.0)]:
        bt = Backtester(data, sig, risk=RISK, bar_minutes=BARMIN,
                        ex=ExecConfig(taker_fee=fee, slip_entry_bps=slip,
                                      slip_stop_bps=slip * 2.5, use_funding=True,
                                      pessimistic_ambiguous_bar=True, chandelier=True))
        rr = bt.run()
        print(f"  {lab:14s} n={rr['n_trades']:4d} PF={rr['profit_factor']:.2f} "
              f"expR={rr['expectancy_R']:+.3f} net={rr['net_pnl']:+8.1f} "
              f"DD={rr['max_dd_pct']:5.1f}% Sharpe={rr['sharpe']:.2f}", flush=True)

    print("\n" + "#" * 70); print("# D. PARAMETER NEIGHBOURHOOD (full sample)")
    print("#" * 70)
    for k, vals in SPACE.items():
        if len(vals) < 2:
            continue
        out = []
        for v in vals:
            p = dict(best_joint); p[k] = v
            sg = {s: fn(d, s, **p) for s, d in data.items()}
            rr = Backtester(data, sg, risk=RISK, ex=EX, bar_minutes=BARMIN).run()
            out.append(f"{v}:PF{rr.get('profit_factor', 0):.2f}/n{rr.get('n_trades', 0)}")
        print(f"  {k:15s} " + "   ".join(out), flush=True)

    print("\n" + "#" * 70)
    print("# E. RANDOM-ENTRY CONTROL (long-only) (same exits, entries shuffled in time)")
    print("#" * 70)
    rng = np.random.default_rng(11)
    pfs = []
    for it in range(20):
        sg = {}
        for s, g in sig.items():
            g2 = g.copy()
            nz = np.flatnonzero(g2.side.to_numpy() != 0)
            if len(nz) == 0:
                sg[s] = g2; continue
            valid = np.flatnonzero(g2.atr.notna().to_numpy())
            valid = valid[valid > 300]
            if len(valid) < len(nz):
                sg[s] = g2; continue
            new = rng.choice(valid, size=len(nz), replace=False)
            side_vals = g2.side.to_numpy()[nz]
            g2["side"] = 0
            arr = g2.side.to_numpy().copy()
            arr[new] = side_vals
            g2["side"] = arr
            # rebuild levels at the shuffled bars
            a = g2.atr.to_numpy()
            close = g2.ref_price.to_numpy()
            sl = np.where(arr > 0, close - best_joint["sl_atr"] * a,
                          close + best_joint["sl_atr"] * a)
            g2["sl"] = sl
            g2["tp"] = np.where(arr > 0, close + 40 * a, close - 40 * a)
            sg[s] = g2
        rr = Backtester(data, sg, risk=RISK, ex=EX, bar_minutes=BARMIN).run()
        if rr.get("n_trades"):
            pfs.append(rr["profit_factor"])
    real = Backtester(data, sig, risk=RISK, ex=EX, bar_minutes=BARMIN).run()
    pfs = np.array([p for p in pfs if np.isfinite(p)])
    print(f"  real entries      PF={real['profit_factor']:.2f} "
          f"expR={real['expectancy_R']:+.3f} n={real['n_trades']}")
    if len(pfs):
        print(f"  random entries    PF mean={pfs.mean():.2f} "
              f"p5={np.percentile(pfs,5):.2f} p95={np.percentile(pfs,95):.2f} "
              f"({len(pfs)} runs)")
        print(f"  -> the timing signal beats random in "
              f"{100*(real['profit_factor'] > pfs).mean():.0f}% of controls")


if __name__ == "__main__":
    main()
