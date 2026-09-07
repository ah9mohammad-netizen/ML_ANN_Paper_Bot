"""Focused study of the one strategy that showed out-of-sample edge.

Answers, in order:
  1. which timeframe?          walk-forward on 1h / 2h / 4h / 8h / 1d
  2. does it survive to today? folds extended through 2026-08
  3. how fragile is it?        fee sensitivity + parameter neighbourhood
  4. where does it earn?       per regime, per symbol, per side
"""
import sys, json, os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import strategies as st
import regimes as rg
import indicators as ta
from engine import Backtester, RiskConfig, ExecConfig, fmt
from walkforward import walk_forward, stability

RULE = {"1h": "1h", "2h": "2h", "4h": "4h", "8h": "8h", "1d": "1D"}
BARMIN = {"1h": 60, "2h": 120, "4h": 240, "8h": 480, "1d": 1440}
SYMS = ["BTC", "ETH", "SOL", "LINK", "XRP", "DOGE", "AVAX", "NEAR", "APT", "SUI"]

SPACE = dict(squeeze_n=[96, 192], squeeze_q=[0.2, 0.35], entry_n=[20, 30, 48],
             sl_atr=[2.0, 2.5, 3.5], trail_atr=[2.0, 3.0, 4.5],
             trail_start_r=[0.0, 0.5], ema_filter=[0, 200])

RISK = RiskConfig(starting_equity=1000, risk_per_trade=0.0075, leverage=4,
                  max_open=5, max_gross_notional=2.0,
                  max_notional_per_trade=0.6, max_daily_loss=0.06,
                  max_drawdown_halt=0.35)
EX = ExecConfig(taker_fee=0.0005, slip_entry_bps=2.0, slip_stop_bps=5.0,
                use_funding=True, pessimistic_ambiguous_bar=True)

_cache = {}


def load(tf, start="2022-01-01"):
    if tf in _cache:
        return _cache[tf]
    out = {}
    for s in SYMS:
        f = f"data/SWAP_{s}_15m.parquet"
        if not os.path.exists(f):
            continue
        d = pd.read_parquet(f)
        if len(d) < 30000:
            continue
        r = ta.resample(d, RULE[tf]) if tf != "15m" else d
        out[s] = r.loc[pd.Timestamp(start, tz="UTC"):]
    _cache[tf] = out
    return out


def breakdown(t, reg, title):
    if t.empty:
        return
    lab = reg["regime"].reindex(pd.DatetimeIndex(t.entry_ts).normalize(),
                                method="ffill")
    t = t.copy(); t["mkt"] = lab.values
    t["dir"] = np.where(t.side > 0, "LONG", "SHORT")
    for col in ["mkt", "symbol", "dir"]:
        rows = []
        for k, g in t.groupby(col):
            w = g[g.pnl > 0]; l = g[g.pnl <= 0]; gl = abs(l.pnl.sum())
            rows.append({col: k, "n": len(g),
                         "win%": round(100 * len(w) / len(g), 1),
                         "PF": round(w.pnl.sum() / gl, 2) if gl > 0 else 999,
                         "expR": round(g.r_multiple.mean(), 3),
                         "pnl": round(g.pnl.sum(), 1)})
        print(f"\n  {title} by {col}:")
        print(pd.DataFrame(rows).sort_values("pnl", ascending=False)
              .to_string(index=False))


def main():
    reg = rg.btc_regime(pd.read_parquet("data/SWAP_BTC_15m.parquet"))
    best = None
    print("#" * 70)
    print("# 1. TIMEFRAME SWEEP (walk-forward, folds extended toward 2026-08)")
    print("#" * 70, flush=True)
    for tf in ["1h", "2h", "4h", "8h", "1d"]:
        data = load(tf)
        res = walk_forward(data, st.vol_breakout_v2, SPACE, RISK, EX,
                           bar_minutes=BARMIN[tf], n_folds=12, train_days=365,
                           test_days=110, n_candidates=40, min_trades_train=20,
                           verbose=False)
        r = res["report"]
        if not r:
            print(f"  {tf}: no trades", flush=True); continue
        print(f"  {tf:3s} n={r['n_trades']:4d} PF={r['profit_factor']:.2f} "
              f"CI[{r['pf_ci95'][0]:.2f},{r['pf_ci95'][1]:.2f}] "
              f"expR={r['expectancy_R']:+.3f} Sharpe={r['sharpe']:5.2f} "
              f"DD={r['max_dd_pct']:5.1f}% t={r.get('t_stat', 0):5.2f} "
              f"win={r['win_rate']:.1f}% folds={r.get('n_folds_used')} "
              f"{r['start']}->{r['end']}", flush=True)
        if best is None or r.get("t_stat", 0) > best[1].get("t_stat", 0):
            best = (tf, r, res)

    if not best:
        return
    tf, r, res = best
    print("\n" + "#" * 70)
    print(f"# 2. BEST TIMEFRAME = {tf}")
    print("#" * 70)
    print(fmt(r, f"squeeze @ {tf} · OUT-OF-SAMPLE walk-forward"))
    print("\n  fold-by-fold:")
    print(res["folds"].to_string(index=False))
    print("\n  parameter stability:")
    print(json.dumps(stability(res["folds"], list(SPACE)), indent=1, default=str))
    breakdown(res["trades"], reg, f"squeeze@{tf}")
    res["trades"].to_csv(f"out_squeeze_{tf}.csv", index=False)
    res["folds"].to_csv(f"folds_squeeze_{tf}.csv", index=False)
    res["equity"].to_csv(f"eq_squeeze_{tf}.csv")

    f = res["folds"]
    modal = {}
    for k in SPACE:
        if k in f:
            v = f[k].mode().iloc[0]
            modal[k] = (int(v) if isinstance(v, (np.integer, int)) and
                        not isinstance(v, bool) else
                        float(v) if isinstance(v, (np.floating, float)) else v)
    print("\n  modal params:", modal)
    with open(f"best_squeeze_{tf}.json", "w") as fh:
        json.dump({"timeframe": tf, "params": modal}, fh, indent=2)

    data = load(tf)
    sig = {s: st.vol_breakout_v2(d, s, **modal) for s, d in data.items()}

    print("\n" + "#" * 70)
    print("# 3. COST SENSITIVITY (modal params, full 2022-2026 sample)")
    print("#" * 70)
    for label, fee, slip in [("maker 2bps", 0.0002, 1.0),
                             ("taker 5bps (base)", 0.0005, 2.0),
                             ("taker 7.5bps", 0.00075, 3.0),
                             ("taker 10bps", 0.0010, 4.0)]:
        bt = Backtester(data, sig, risk=RISK, bar_minutes=BARMIN[tf],
                        ex=ExecConfig(taker_fee=fee, slip_entry_bps=slip,
                                      slip_stop_bps=slip * 2.5, use_funding=True,
                                      pessimistic_ambiguous_bar=True))
        rr = bt.run()
        print(f"  {label:18s} n={rr['n_trades']:4d} PF={rr['profit_factor']:.2f} "
              f"expR={rr['expectancy_R']:+.3f} net={rr['net_pnl']:+8.1f} "
              f"DD={rr['max_dd_pct']:5.1f}% Sharpe={rr['sharpe']:.2f}", flush=True)

    print("\n" + "#" * 70)
    print("# 4. PARAMETER NEIGHBOURHOOD (full sample, one knob at a time)")
    print("#" * 70)
    for k, vals in SPACE.items():
        line = []
        for v in vals:
            p = dict(modal); p[k] = v
            sg = {s: st.vol_breakout_v2(d, s, **p) for s, d in data.items()}
            bt = Backtester(data, sg, risk=RISK, ex=EX, bar_minutes=BARMIN[tf])
            rr = bt.run()
            line.append(f"{v}:PF{rr.get('profit_factor', 0):.2f}/n{rr.get('n_trades', 0)}")
        print(f"  {k:15s} " + "   ".join(line), flush=True)

    print("\n" + "#" * 70)
    print("# 5. FULL-SAMPLE RUN WITH MODAL PARAMS (in-sample reference only)")
    print("#" * 70)
    bt = Backtester(data, sig, risk=RISK, ex=EX, bar_minutes=BARMIN[tf])
    rr = bt.run()
    print(fmt(rr, f"squeeze @ {tf} · full sample, modal params"))
    breakdown(bt.trades_df(), reg, "full-sample")
    bt.equity_df().to_csv(f"eq_full_squeeze_{tf}.csv")


if __name__ == "__main__":
    main()
