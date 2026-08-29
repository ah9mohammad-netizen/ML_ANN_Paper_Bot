"""V2: walk-forward optimise the rebuilt strategies, then report per regime."""
import sys, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd

import okx_data as od
import strategies as st
import regimes as rg
from engine import Backtester, RiskConfig, ExecConfig, build_report, fmt
from walkforward import walk_forward, stability, slice_all

import os
PAIRS = [p for p in ["BTC","ETH","SOL","LINK","XRP","DOGE","AVAX","NEAR","APT","SUI"]
         if os.path.exists(f"data/SWAP_{p}_15m.parquet")]

RISK = RiskConfig(starting_equity=1000, risk_per_trade=0.0075, leverage=5,
                  max_open=4, max_gross_notional=2.0,
                  max_notional_per_trade=0.8, max_daily_loss=0.05,
                  max_drawdown_halt=0.30)
EX = ExecConfig(taker_fee=0.0005, slip_entry_bps=2.0, slip_stop_bps=5.0,
                use_funding=True, pessimistic_ambiguous_bar=True)

SPACES = {
    "sweep": dict(
        lookback=[36, 48, 72],
        tp_atr=[2.0, 2.6, 3.2],
        vol_mult=[1.0, 1.3, 1.6],
        max_hold_bars=[48, 64, 96],
        trail_atr=[0.0, 2.0, 3.0],
        be_at_r=[0.0, 1.0],
        with_trend_only=[True, False],
        require_reclaim_body=[True, False],
    ),
    "pullback": dict(
        ema_fast=[13, 21, 34],
        ema_slow=[55, 89],
        tp_atr=[2.5, 3.5, 4.5],
        sl_atr=[1.0, 1.3, 1.8],
        adx_min=[18, 22, 27],
        max_hold_bars=[64, 96, 160],
        trail_atr=[0.0, 2.5, 3.5],
    ),
    "meanrev": dict(
        bb_k=[2.2, 2.5, 3.0],
        sl_atr=[1.2, 1.6, 2.2],
        adx_max=[18, 22, 26],
        rsi_lo=[18, 22, 27],
        max_hold_bars=[16, 24, 40],
        only_chop=[True, False],
    ),
}
FNS = {"sweep": st.sweep_reclaim_v2,
       "pullback": st.trend_pullback_v2,
       "meanrev": st.mean_reversion_v2}


def mirror_rsi(space):
    return space


def load(tf, start):
    out = {}
    for s in PAIRS:
        f = f"data/SWAP_{s}_{tf}.parquet"
        if not os.path.exists(f):
            continue
        d = pd.read_parquet(f)
        if len(d) > 30000:
            out[s] = d.loc[pd.Timestamp(start, tz="UTC"):]
    return out


def by_regime(trades, reg_daily, label=""):
    if trades.empty:
        return
    r = reg_daily["regime"]
    lab = r.reindex(pd.DatetimeIndex(trades.entry_ts).normalize(), method="ffill")
    trades = trades.copy(); trades["mkt"] = lab.values
    print(f"\n  ── {label} by market regime (BTC-defined) ──")
    rows = []
    for k, g in trades.groupby("mkt"):
        w = g[g.pnl > 0]; l = g[g.pnl <= 0]
        gl = abs(l.pnl.sum())
        rows.append({"regime": k, "n": len(g),
                     "win%": round(100 * len(w) / len(g), 1),
                     "PF": round(w.pnl.sum() / gl, 2) if gl > 0 else np.inf,
                     "expR": round(g.r_multiple.mean(), 3),
                     "pnl": round(g.pnl.sum(), 1)})
    print(pd.DataFrame(rows).to_string(index=False))


def main(tf="15m", start="2022-01-01", which=("sweep", "pullback", "meanrev"),
         folds=8, train=240, test=90, cands=45):
    data = load(tf, start)
    print("loaded:", {k: len(v) for k, v in data.items()}, flush=True)
    if len(data) < 3:
        print("not enough data yet"); return
    fund = {}
    for s in data:
        try:
            f = od.funding(s, start)
            if len(f): fund[s] = f
        except Exception:
            pass
    reg = rg.btc_regime(data["BTC"])
    print("regime mix:", rg.summarise(reg))
    for lab, a, b, n in rg.segments(reg, 30):
        print(f"   {lab:5s} {str(a)[:10]} → {str(b)[:10]}  ({n}d)")

    bar_min = 15 if tf == "15m" else (5 if tf == "5m" else 60)
    results = {}
    for name in which:
        print(f"\n{'='*68}\n{name.upper()}  walk-forward\n{'='*68}", flush=True)
        res = walk_forward(data, FNS[name], SPACES[name], RISK, EX,
                           funding=fund, bar_minutes=bar_min, n_folds=folds,
                           train_days=train, test_days=test, n_candidates=cands)
        results[name] = res
        if res["report"]:
            print()
            print(fmt(res["report"], f"{name} · OUT-OF-SAMPLE (stitched folds)"))
            print("\n  parameter stability across folds:")
            print(json.dumps(stability(res["folds"], list(SPACES[name])), indent=2,
                             default=str))
            by_regime(res["trades"], reg, name)
            res["trades"].to_csv(f"/home/claude/quant/out_v2_{name}_{tf}.csv", index=False)
            res["folds"].to_csv(f"/home/claude/quant/folds_v2_{name}_{tf}.csv", index=False)
    return results


if __name__ == "__main__":
    main(*(sys.argv[1:3] or ["15m", "2022-01-01"]))
