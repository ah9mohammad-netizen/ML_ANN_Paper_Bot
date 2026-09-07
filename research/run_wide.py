"""Does the edge survive a much wider universe — and does diversification fix
the trade-count problem that makes the 11-pair version uncompoundable?

Same strategy, same walk-forward discipline, same costs. Only the universe and
the concurrency limits change.
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
import strategies as st, regimes as rg, indicators as ta
from engine import Backtester, RiskConfig, ExecConfig, fmt
from walkforward import walk_forward, stability

TF, BARMIN = "8h", 480
MIN_BARS_4H = 2200

EX = ExecConfig(taker_fee=0.0005, slip_entry_bps=2.0, slip_stop_bps=5.0,
                use_funding=True, pessimistic_ambiguous_bar=True, chandelier=True)
SPACE = dict(entry_n=[20, 30, 48, 72], sl_atr=[2.0, 3.0],
             trail_atr=[3.0, 4.5], trail_start_r=[0.0, 0.5],
             adx_min=[0, 18], ema_filter=[0, 200], max_hold_bars=[0])

_BTC = pd.read_parquet("data/SWAP_BTC_4h.parquet")
_fc = {}
def mfilter(ix):
    k = (ix[0], ix[-1], len(ix))
    if k not in _fc: _fc[k] = st.btc_trend_filter(_BTC, ix)
    return _fc[k]

def sigfn(df, symbol="", **kw):
    kw = dict(kw); kw.setdefault("long_only", True)
    return st.apply_market_filter(st.donchian_trend_v2(df, symbol, **kw),
                                  mfilter(df.index), "align")

def load():
    out = {}
    for f in sorted(os.listdir("data")):
        if not f.endswith("_4h.parquet") or f.startswith("SPOT_"):
            continue
        s = f.replace("SWAP_", "").replace("_4h.parquet", "")
        d = pd.read_parquet("data/" + f)
        if len(d) < MIN_BARS_4H:
            continue
        out[s] = ta.resample(d, TF).loc[pd.Timestamp("2022-01-01", tz="UTC"):]
    return out

def tbl(t, col, reg=None):
    if col == "mkt":
        lab = reg["regime"].reindex(pd.DatetimeIndex(t.entry_ts).normalize(), method="ffill")
        t = t.copy(); t["mkt"] = lab.values
    if col == "year":
        t = t.copy(); t["year"] = pd.DatetimeIndex(t.entry_ts).year
    rows = []
    for k, g in t.groupby(col):
        w = g[g.pnl > 0]; l = g[g.pnl <= 0]; gl = abs(l.pnl.sum())
        rows.append({col: k, "n": len(g), "win%": round(100*len(w)/len(g),1),
                     "PF": round(w.pnl.sum()/gl,2) if gl>0 else 999,
                     "expR": round(g.r_multiple.mean(),3), "pnl": round(g.pnl.sum(),1)})
    return pd.DataFrame(rows)

def main():
    data = load()
    print(f"universe: {len(data)} perps\n  {sorted(data)}\n", flush=True)
    reg = rg.btc_regime(_BTC)
    CONFIGS = {
      "narrow  (11 pairs, 5 open, 0.75%)": None,
      "wide    (all, 8 open, 0.50%)":  RiskConfig(starting_equity=1000, risk_per_trade=0.005,
            leverage=4, max_open=8, max_gross_notional=2.0,
            max_notional_per_trade=0.35, max_daily_loss=0.06, max_drawdown_halt=0.35),
      "wide    (all, 14 open, 0.35%)": RiskConfig(starting_equity=1000, risk_per_trade=0.0035,
            leverage=4, max_open=14, max_gross_notional=2.5,
            max_notional_per_trade=0.25, max_daily_loss=0.06, max_drawdown_halt=0.35),
      "wide    (all, 20 open, 0.25%)": RiskConfig(starting_equity=1000, risk_per_trade=0.0025,
            leverage=4, max_open=20, max_gross_notional=3.0,
            max_notional_per_trade=0.20, max_daily_loss=0.08, max_drawdown_halt=0.40),
    }
    USER11 = ["BTC","ETH","SOL","LINK","NEAR","SUI","APT","HYPE","PEPE","WIF","DOGE"]
    CONFIGS["narrow  (11 pairs, 5 open, 0.75%)"] = RiskConfig(
        starting_equity=1000, risk_per_trade=0.0075, leverage=4, max_open=5,
        max_gross_notional=2.0, max_notional_per_trade=0.6,
        max_daily_loss=0.06, max_drawdown_halt=0.35)

    best = None
    for name, R in CONFIGS.items():
        d = {k: v for k, v in data.items() if k in USER11} if name.startswith("narrow") else data
        res = walk_forward(d, sigfn, SPACE, R, EX, bar_minutes=BARMIN, n_folds=12,
                           train_days=365, test_days=110, n_candidates=30,
                           min_trades_train=12, verbose=False)
        r = res["report"]
        if not r.get("n_trades"):
            print(f"  {name:34s} no trades", flush=True); continue
        print(f"  {name:34s} n={r['n_trades']:4d} ({r['trades_per_week']:.1f}/wk) "
              f"PF={r['profit_factor']:.2f} CI[{r['pf_ci95'][0]:.2f},{r['pf_ci95'][1]:.2f}] "
              f"expR={r['expectancy_R']:+.3f} Sh={r['sharpe']:5.2f} "
              f"DD={r['max_dd_pct']:5.1f}% Calmar={r['calmar']:5.2f} "
              f"CAGR={r['cagr_pct']:+6.1f}% t={r.get('t_stat',0):5.2f} "
              f"win={r['win_rate']:.1f}%", flush=True)
        if best is None or r["sharpe"] > best[1]["sharpe"]:
            best = (name, r, res, R)

    if not best: return
    name, r, res, R = best
    print("\n" + "#"*72); print(f"# BEST: {name}"); print("#"*72)
    print(fmt(r, f"wide universe · OUT-OF-SAMPLE"))
    T = res["trades"]
    for c in ["mkt", "year"]:
        print(f"\n  by {c}:"); print(tbl(T, c, reg).to_string(index=False))
    print("\n  top / bottom symbols:")
    s = tbl(T, "symbol").sort_values("pnl", ascending=False)
    print(pd.concat([s.head(12), s.tail(8)]).to_string(index=False))
    print("\n  folds:"); print(res["folds"].to_string(index=False))
    T.to_csv("out_wide.csv", index=False); res["folds"].to_csv("folds_wide.csv", index=False)
    res["equity"].to_csv("eq_wide.csv")

    print("\n" + "#"*72); print("# RISK SCALING (same signals, risk-per-trade dial)")
    print("#"*72)
    for rp in [0.0025, 0.005, 0.0075, 0.010, 0.015]:
        R2 = RiskConfig(starting_equity=1000, risk_per_trade=rp, leverage=4,
                        max_open=R.max_open, max_gross_notional=R.max_gross_notional,
                        max_notional_per_trade=R.max_notional_per_trade,
                        max_daily_loss=0.10, max_drawdown_halt=0.60)
        res2 = walk_forward(data, sigfn, SPACE, R2, EX, bar_minutes=BARMIN, n_folds=12,
                            train_days=365, test_days=110, n_candidates=30,
                            min_trades_train=12, verbose=False)
        r2 = res2["report"]
        if r2.get("n_trades"):
            print(f"  risk {100*rp:.2f}%/trade  CAGR={r2['cagr_pct']:+7.1f}%  "
                  f"maxDD={r2['max_dd_pct']:5.1f}%  Sharpe={r2['sharpe']:.2f}  "
                  f"Calmar={r2['calmar']:.2f}", flush=True)

if __name__ == "__main__":
    main()
