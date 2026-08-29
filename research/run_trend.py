"""Do the higher timeframes have the edge the 15m tape doesn't?

Fees are a fixed ~14bps round trip. On a 15m scalp targeting 0.5% that is a
quarter of the gross edge; on a 4h/1d trend trade targeting 6% it is 2%.
This script tests trend-following on resampled 1h / 4h / 1d bars with the same
walk-forward discipline.
"""
import sys, json, os, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import strategies as st, regimes as rg, indicators as ta
from engine import RiskConfig, ExecConfig, fmt
from walkforward import walk_forward, stability

RULE = {"1h": "1h", "2h": "2h", "4h": "4h", "8h": "8h", "1d": "1D"}
BARMIN = {"1h": 60, "2h": 120, "4h": 240, "8h": 480, "1d": 1440}

SPACES = {
 "donchian": dict(entry_n=[20, 30, 48, 72], sl_atr=[2.0, 2.5, 3.5],
                  trail_atr=[2.0, 3.0, 4.5], trail_start_r=[0.0, 0.5, 1.0],
                  adx_min=[0, 15, 22], ema_filter=[0, 100, 200],
                  max_hold_bars=[0]),
 "squeeze": dict(squeeze_n=[96, 192], squeeze_q=[0.2, 0.35], entry_n=[20, 30, 48],
                 sl_atr=[2.0, 2.5, 3.5], trail_atr=[2.0, 3.0, 4.5],
                 trail_start_r=[0.0, 0.5], ema_filter=[0, 200]),
 "pullback": dict(ema_fast=[13, 21, 34], ema_slow=[55, 89], tp_atr=[3.0, 4.5, 6.0],
                  sl_atr=[1.5, 2.0, 2.5], adx_min=[15, 20, 25],
                  max_hold_bars=[0, 60, 120], trail_atr=[0.0, 3.0, 4.0], htf=["1d"]),
}
FNS = {"donchian": st.donchian_trend_v2, "squeeze": st.vol_breakout_v2,
       "pullback": st.trend_pullback_v2}

RISK = RiskConfig(starting_equity=1000, risk_per_trade=0.0075, leverage=4,
                  max_open=5, max_gross_notional=2.0,
                  max_notional_per_trade=0.6, max_daily_loss=0.06,
                  max_drawdown_halt=0.35)
EX = ExecConfig(taker_fee=0.0005, slip_entry_bps=2.0, slip_stop_bps=5.0,
                use_funding=True, pessimistic_ambiguous_bar=True)

SYMS = ["BTC","ETH","SOL","LINK","XRP","DOGE","AVAX","NEAR","APT","SUI"]


def load(tf, start="2022-01-01"):
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
    return out


def by_regime(trades, reg_daily, label=""):
    if trades.empty: return
    lab = reg_daily["regime"].reindex(
        pd.DatetimeIndex(trades.entry_ts).normalize(), method="ffill")
    t = trades.copy(); t["mkt"] = lab.values
    rows = []
    for k, g in t.groupby("mkt"):
        w = g[g.pnl > 0]; l = g[g.pnl <= 0]; gl = abs(l.pnl.sum())
        rows.append({"regime": k, "n": len(g),
                     "win%": round(100*len(w)/len(g), 1),
                     "PF": round(w.pnl.sum()/gl, 2) if gl > 0 else 999,
                     "expR": round(g.r_multiple.mean(), 3),
                     "pnl": round(g.pnl.sum(), 1)})
    print(f"\n  ── {label} by market regime ──")
    print(pd.DataFrame(rows).to_string(index=False))


def main(tfs=("4h","1h"), which=("donchian","squeeze","pullback"),
         folds=7, train=365, test=120, cands=40):
    base = load("4h")
    reg = rg.btc_regime(pd.read_parquet("data/SWAP_BTC_15m.parquet"))
    for tf in tfs:
        data = load(tf)
        print(f"\n{'#'*70}\n# TIMEFRAME {tf}   {[(k,len(v)) for k,v in data.items()]}\n{'#'*70}", flush=True)
        for name in which:
            print(f"\n===== {name} @ {tf} =====", flush=True)
            res = walk_forward(data, FNS[name], SPACES[name], RISK, EX,
                               bar_minutes=BARMIN[tf], n_folds=folds,
                               train_days=train, test_days=test,
                               n_candidates=cands, min_trades_train=20)
            if not res["report"]:
                print("  no out-of-sample trades"); continue
            print(); print(fmt(res["report"], f"{name} @ {tf} · OUT-OF-SAMPLE"))
            print("\n  parameter stability:")
            print(json.dumps(stability(res["folds"], list(SPACES[name])), indent=1, default=str))
            by_regime(res["trades"], reg, f"{name}@{tf}")
            res["trades"].to_csv(f"out_trend_{name}_{tf}.csv", index=False)
            res["folds"].to_csv(f"folds_trend_{name}_{tf}.csv", index=False)


if __name__ == "__main__":
    main()
