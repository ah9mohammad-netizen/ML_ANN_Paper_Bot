"""Final configuration, run on the user's exact 12-pair universe.

Reports (a) the honest walk-forward out-of-sample estimate and (b) the
full-sample run with the locked parameters, per pair and per tier, so the
forecast is about THIS universe rather than the research universe.
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import strategies as st
import regimes as rg
import indicators as ta
from engine import Backtester, RiskConfig, ExecConfig, fmt
from walkforward import walk_forward

USER_PAIRS = ["BTC", "ETH", "SOL", "LINK", "NEAR", "SUI", "APT",
              "HYPE", "PEPE", "WIF", "FET", "DOGE"]
TIER = {"BTC": "large", "ETH": "large",
        "SOL": "mid", "LINK": "mid", "NEAR": "mid", "SUI": "mid", "APT": "mid",
        "HYPE": "low", "PEPE": "low", "WIF": "low", "FET": "low", "DOGE": "low"}

TF, BARMIN = "8h", 480

# locked configuration — see FINDINGS.md section 4
PARAMS = dict(entry_n=48, sl_atr=3.0, trail_atr=4.5, trail_start_r=0.5,
              adx_min=18, ema_filter=200, long_only=True, max_hold_bars=0)

RISK = RiskConfig(starting_equity=1000, risk_per_trade=0.0075, leverage=4,
                  max_open=5, max_gross_notional=2.0,
                  max_notional_per_trade=0.6, max_daily_loss=0.06,
                  max_drawdown_halt=0.35)
EX = ExecConfig(taker_fee=0.0005, slip_entry_bps=2.0, slip_stop_bps=5.0,
                use_funding=True, pessimistic_ambiguous_bar=True, chandelier=True)

SPACE = dict(entry_n=[20, 30, 48, 72], sl_atr=[2.0, 3.0],
             trail_atr=[3.0, 4.5], trail_start_r=[0.0, 0.5],
             adx_min=[0, 18], ema_filter=[0, 200], max_hold_bars=[0])

_BTC = pd.read_parquet("data/SWAP_BTC_4h.parquet")
_fc = {}


def mfilter(index):
    k = (index[0], index[-1], len(index))
    if k not in _fc:
        _fc[k] = st.btc_trend_filter(_BTC, index)
    return _fc[k]


def sigfn(df, symbol="", **kw):
    kw = dict(kw); kw.setdefault("long_only", True)
    return st.apply_market_filter(st.donchian_trend_v2(df, symbol, **kw),
                                  mfilter(df.index), "align")


def load():
    out, missing = {}, []
    for s in USER_PAIRS:
        p = f"data/SWAP_{s}_4h.parquet"
        if not os.path.exists(p):
            missing.append(s); continue
        d = pd.read_parquet(p)
        if len(d) < 1500:
            missing.append(f"{s}(thin)"); continue
        out[s] = ta.resample(d, TF).loc[pd.Timestamp("2022-01-01", tz="UTC"):]
    return out, missing


def table(t, col, reg=None):
    if col == "tier":
        t = t.copy(); t["tier"] = t.symbol.map(TIER)
    if col == "mkt":
        lab = reg["regime"].reindex(pd.DatetimeIndex(t.entry_ts).normalize(),
                                    method="ffill")
        t = t.copy(); t["mkt"] = lab.values
    if col == "year":
        t = t.copy(); t["year"] = pd.DatetimeIndex(t.entry_ts).year
    rows = []
    for k, g in t.groupby(col):
        w = g[g.pnl > 0]; l = g[g.pnl <= 0]; gl = abs(l.pnl.sum())
        rows.append({col: k, "n": len(g),
                     "win%": round(100 * len(w) / len(g), 1),
                     "PF": round(w.pnl.sum() / gl, 2) if gl > 0 else 999,
                     "payoff": round(w.pnl.mean() / abs(l.pnl.mean()), 2)
                     if len(l) and l.pnl.mean() else 999,
                     "expR": round(g.r_multiple.mean(), 3),
                     "pnl": round(g.pnl.sum(), 1)})
    return pd.DataFrame(rows)


def main():
    data, missing = load()
    print(f"universe: {list(data)}")
    if missing:
        print(f"NO DATA on OKX perps: {missing}")
    for s, d in data.items():
        print(f"  {s:5s} {TIER[s]:5s} {len(d):5d} bars  {str(d.index[0])[:10]} → {str(d.index[-1])[:10]}")
    reg = rg.btc_regime(_BTC)

    print("\n" + "#" * 72)
    print("# A. WALK-FORWARD OUT-OF-SAMPLE  (the honest forecast)")
    print("#" * 72, flush=True)
    res = walk_forward(data, sigfn, SPACE, RISK, EX, bar_minutes=BARMIN,
                       n_folds=12, train_days=365, test_days=110,
                       n_candidates=36, min_trades_train=12, verbose=True)
    r = res["report"]
    if r.get("n_trades"):
        print(); print(fmt(r, "YOUR 12 PAIRS · walk-forward OOS"))
        T = res["trades"]
        for c in ["tier", "mkt", "year", "symbol"]:
            print(f"\n  by {c}:")
            print(table(T, c, reg).sort_values("pnl", ascending=False).to_string(index=False))
        T.to_csv("out_user_oos.csv", index=False)
        res["folds"].to_csv("folds_user.csv", index=False)

    print("\n" + "#" * 72)
    print("# B. LOCKED PARAMETERS, FULL SAMPLE  (in-sample reference)")
    print("#" * 72)
    print(f"  {PARAMS}")
    sig = {s: sigfn(d, s, **PARAMS) for s, d in data.items()}
    bt = Backtester(data, sig, risk=RISK, ex=EX, bar_minutes=BARMIN)
    rr = bt.run()
    print(); print(fmt(rr, "YOUR 12 PAIRS · full sample, locked params"))
    T2 = bt.trades_df()
    for c in ["tier", "mkt", "year", "symbol"]:
        print(f"\n  by {c}:")
        print(table(T2, c, reg).sort_values("pnl", ascending=False).to_string(index=False))
    T2.to_csv("out_user_full.csv", index=False)

    print("\n" + "#" * 72)
    print("# C. EXPECTED TRADE FREQUENCY")
    print("#" * 72)
    days = (list(data.values())[0].index[-1] - list(data.values())[0].index[0]).days
    print(f"  full sample: {rr['n_trades']} trades over {days} days "
          f"= {rr['trades_per_week']:.2f}/week = {rr['n_trades']/(days/30.4):.1f}/month")
    print(f"  walk-forward OOS: {r.get('trades_per_week', 0):.2f}/week")
    print(f"  time to 100 closed trades at OOS rate: "
          f"{100/max(r.get('trades_per_week', 0.01), 0.01)/52:.1f} years")


if __name__ == "__main__":
    main()
