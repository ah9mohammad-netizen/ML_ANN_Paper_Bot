"""Final experiment: does a chandelier trail + BTC-regime alignment rescue the
4h breakout? Walk-forward, folds running to 2026-08.

This is the LAST hypothesis tested. Every additional strategy tried inflates the
chance of a false positive, so the bar is raised: to be called an edge a result
must show t-stat > 2.5 AND a bootstrap PF 95% CI whose lower bound clears 1.10
AND positive expectancy in at least two of the three market regimes.
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
RULE = {"2h": "2h", "4h": "4h", "8h": "8h"}
BARMIN = {"2h": 120, "4h": 240, "8h": 480}

RISK = RiskConfig(starting_equity=1000, risk_per_trade=0.0075, leverage=4,
                  max_open=5, max_gross_notional=2.0,
                  max_notional_per_trade=0.6, max_daily_loss=0.06,
                  max_drawdown_halt=0.35)
EX = ExecConfig(taker_fee=0.0005, slip_entry_bps=2.0, slip_stop_bps=5.0,
                use_funding=True, pessimistic_ambiguous_bar=True, chandelier=True)

_BTC = pd.read_parquet("data/SWAP_BTC_15m.parquet")
_filt = {}


def mfilter(index):
    key = (index[0], index[-1], len(index))
    if key not in _filt:
        _filt[key] = st.btc_trend_filter(_BTC, index)
    return _filt[key]


def load(tf, start="2022-01-01"):
    out = {}
    for s in SYMS:
        f = f"data/SWAP_{s}_15m.parquet"
        if not os.path.exists(f):
            continue
        d = pd.read_parquet(f)
        if len(d) < 30000:
            continue
        out[s] = ta.resample(d, RULE[tf]).loc[pd.Timestamp(start, tz="UTC"):]
    return out


def make_filtered(base_fn, market_mode):
    def fn(df, symbol="", **kw):
        sig = base_fn(df, symbol, **kw)
        if market_mode == "none":
            return sig
        return st.apply_market_filter(sig, mfilter(df.index), market_mode)
    return fn


SPACES = {
    "squeeze": dict(squeeze_n=[96, 192], squeeze_q=[0.2, 0.35], entry_n=[20, 30, 48],
                    sl_atr=[2.0, 3.0], trail_atr=[2.0, 3.0, 4.5],
                    trail_start_r=[0.0, 0.5], ema_filter=[0, 200]),
    "donchian": dict(entry_n=[20, 30, 48, 72], sl_atr=[2.0, 3.0],
                     trail_atr=[2.0, 3.0, 4.5], trail_start_r=[0.0, 0.5],
                     adx_min=[0, 18], ema_filter=[0, 200], max_hold_bars=[0]),
}
BASE = {"squeeze": st.vol_breakout_v2, "donchian": st.donchian_trend_v2}


def regime_table(t, reg):
    lab = reg["regime"].reindex(pd.DatetimeIndex(t.entry_ts).normalize(),
                                method="ffill")
    t = t.copy(); t["mkt"] = lab.values
    rows = []
    for k, g in t.groupby("mkt"):
        w = g[g.pnl > 0]; l = g[g.pnl <= 0]; gl = abs(l.pnl.sum())
        rows.append({"regime": k, "n": len(g),
                     "win%": round(100 * len(w) / len(g), 1),
                     "PF": round(w.pnl.sum() / gl, 2) if gl > 0 else 999,
                     "expR": round(g.r_multiple.mean(), 3),
                     "pnl": round(g.pnl.sum(), 1)})
    return pd.DataFrame(rows)


def verdict(r, rt):
    ok_t = r.get("t_stat", 0) > 2.5
    ok_pf = r.get("pf_ci95", (0, 0))[0] > 1.10
    ok_reg = int((rt.expR > 0).sum()) >= 2
    tag = "EDGE" if (ok_t and ok_pf and ok_reg) else "not proven"
    return (f"    t>2.5:{ok_t}  PF_CI_lo>1.10:{ok_pf}  "
            f"positive in >=2 regimes:{ok_reg}  ->  {tag}")


def main():
    reg = rg.btc_regime(_BTC)
    results = []
    for tf in ["2h", "4h", "8h"]:
        data = load(tf)
        for name in ["squeeze", "donchian"]:
            for mm in ["none", "align"]:
                fn = make_filtered(BASE[name], mm)
                res = walk_forward(data, fn, SPACES[name], RISK, EX,
                                   bar_minutes=BARMIN[tf], n_folds=12,
                                   train_days=365, test_days=110,
                                   n_candidates=36, min_trades_train=20,
                                   verbose=False)
                r = res["report"]
                if not r or not r.get("n_trades"):
                    print(f"  {name:9s} {tf:3s} filter={mm:5s}  no trades", flush=True)
                    continue
                rt = regime_table(res["trades"], reg)
                print(f"  {name:9s} {tf:3s} filter={mm:5s}  n={r['n_trades']:4d} "
                      f"PF={r['profit_factor']:.2f} "
                      f"CI[{r['pf_ci95'][0]:.2f},{r['pf_ci95'][1]:.2f}] "
                      f"expR={r['expectancy_R']:+.3f} Sh={r['sharpe']:5.2f} "
                      f"DD={r['max_dd_pct']:5.1f}% t={r.get('t_stat',0):5.2f} "
                      f"win={r['win_rate']:.1f}%", flush=True)
                print(verdict(r, rt), flush=True)
                results.append((name, tf, mm, r, res, rt))

    if not results:
        return
    best = max(results, key=lambda x: x[3].get("t_stat", -9))
    name, tf, mm, r, res, rt = best
    print("\n" + "#" * 70)
    print(f"# BEST OVERALL: {name} @ {tf}, market filter = {mm}")
    print("#" * 70)
    print(fmt(r, f"{name}@{tf}/{mm} · OUT-OF-SAMPLE"))
    print("\n  by regime:"); print(rt.to_string(index=False))
    print("\n  folds:"); print(res["folds"].to_string(index=False))
    print("\n  stability:")
    print(json.dumps(stability(res["folds"], list(SPACES[name])), indent=1, default=str))
    print("\n " + verdict(r, rt))
    res["trades"].to_csv(f"out_final_{name}_{tf}_{mm}.csv", index=False)
    res["folds"].to_csv(f"folds_final_{name}_{tf}_{mm}.csv", index=False)
    res["equity"].to_csv(f"eq_final_{name}_{tf}_{mm}.csv")
    with open("best_final.json", "w") as fh:
        json.dump({"strategy": name, "timeframe": tf, "market_filter": mm,
                   "folds": res["folds"].to_dict("records")}, fh, indent=2,
                  default=str)


if __name__ == "__main__":
    main()
