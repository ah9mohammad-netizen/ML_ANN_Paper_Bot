"""Does splitting the universe into large / mid / low-cap tiers, and fitting a
separate strategy to each, beat one strategy fitted across everything?

Judged only out of sample. Per-tier fitting has three times the freedom, so if
it wins it must win on unseen data, not on the training window.

Each tier is free to choose a different strategy FAMILY (trend breakout, squeeze
breakout, mean reversion, trend pullback) as well as its own thresholds and
stop/target geometry — which is the hypothesis as stated.
"""
import os, json, warnings
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import strategies as st
import regimes as rg
import indicators as ta
from engine import Backtester, RiskConfig, ExecConfig, build_report, fmt
from walkforward import walk_forward, stability, make_folds, slice_all, run_one, sample_grid, score, stitch

TIERS = {
    "large": ["BTC", "ETH", "BNB", "XRP", "TRX", "LTC", "ADA"],
    "mid":   ["SOL", "LINK", "AVAX", "NEAR", "APT", "SUI"],
    "low":   ["DOGE", "PEPE", "SHIB", "WIF", "HYPE"],
}
ALL = [s for v in TIERS.values() for s in v]
TF = os.getenv("TIER_TF", "8h")
BARMIN = {"4h": 240, "8h": 480, "12h": 720}[TF]

RISK = RiskConfig(starting_equity=1000, risk_per_trade=0.0075, leverage=4,
                  max_open=5, max_gross_notional=2.0,
                  max_notional_per_trade=0.6, max_daily_loss=0.06,
                  max_drawdown_halt=0.35)
EX = ExecConfig(taker_fee=0.0005, slip_entry_bps=2.0, slip_stop_bps=5.0,
                use_funding=True, pessimistic_ambiguous_bar=True, chandelier=True)

_BTC = pd.read_parquet("data/SWAP_BTC_4h.parquet")
_fc = {}


def mfilter(index):
    k = (index[0], index[-1], len(index))
    if k not in _fc:
        _fc[k] = st.btc_trend_filter(_BTC, index)
    return _fc[k]


def load():
    out = {}
    for s in ALL:
        p = f"data/SWAP_{s}_4h.parquet"
        if not os.path.exists(p):
            continue
        d = pd.read_parquet(p)
        if len(d) < 3000:
            continue
        r = ta.resample(d, TF) if TF != "4h" else d
        out[s] = r.loc[pd.Timestamp("2022-01-01", tz="UTC"):]
    return out


# ── one signal function that dispatches on a 'family' parameter ────────

FAMILY_ARGS = {
    "donchian": ["entry_n", "sl_atr", "trail_atr", "trail_start_r", "adx_min",
                 "ema_filter", "long_only"],
    "squeeze":  ["squeeze_n", "squeeze_q", "entry_n", "sl_atr", "trail_atr",
                 "trail_start_r", "ema_filter"],
    "meanrev":  ["bb_k", "sl_atr", "adx_max", "rsi_lo", "max_hold_bars",
                 "only_chop"],
    "pullback": ["ema_fast", "ema_slow", "tp_atr", "sl_atr", "adx_min",
                 "max_hold_bars", "trail_atr"],
}
FAMILY_FN = {"donchian": st.donchian_trend_v2, "squeeze": st.vol_breakout_v2,
             "meanrev": st.mean_reversion_v2, "pullback": st.trend_pullback_v2}


def sigfn(df, symbol="", **kw):
    fam = kw.get("family", "donchian")
    mkt = kw.get("market_filter", "align")
    args = {k: kw[k] for k in FAMILY_ARGS[fam] if k in kw}
    sig = FAMILY_FN[fam](df, symbol, **args)
    if mkt != "none":
        sig = st.apply_market_filter(sig, mfilter(df.index), mkt)
    return sig


SPACE = dict(
    family=["donchian", "squeeze", "meanrev", "pullback"],
    market_filter=["align", "none"],
    # donchian / squeeze
    entry_n=[20, 30, 48], sl_atr=[2.0, 3.0], trail_atr=[3.0, 4.5],
    trail_start_r=[0.0, 0.5], adx_min=[0, 18], ema_filter=[0, 200],
    long_only=[True, False],
    squeeze_n=[96, 192], squeeze_q=[0.2, 0.35],
    # meanrev
    bb_k=[2.2, 2.8], adx_max=[20, 26], rsi_lo=[20, 27], only_chop=[True, False],
    # pullback
    ema_fast=[13, 21], ema_slow=[55, 89], tp_atr=[3.0, 4.5],
    max_hold_bars=[0, 30],
)


def portfolio_report(trade_frames, start_equity=1000.0):
    """Merge OOS trades from several sub-books into one equity curve."""
    T = pd.concat([t for t in trade_frames if t is not None and len(t)],
                  ignore_index=True)
    if T.empty:
        return {}, T
    T = T.sort_values("exit_ts")
    eq = start_equity + T.pnl.cumsum()
    curve = pd.DataFrame({"equity": eq.values}, index=pd.DatetimeIndex(T.exit_ts))
    return build_report(T, curve, start_equity, BARMIN), T


def wf(data, folds, cands, label):
    res = walk_forward(data, sigfn, SPACE, RISK, EX, bar_minutes=BARMIN,
                       n_folds=folds, train_days=365, test_days=110,
                       n_candidates=cands, min_trades_train=12, verbose=False)
    r = res["report"]
    if r and r.get("n_trades"):
        print(f"  {label:22s} n={r['n_trades']:4d} PF={r['profit_factor']:.2f} "
              f"CI[{r['pf_ci95'][0]:.2f},{r['pf_ci95'][1]:.2f}] "
              f"expR={r['expectancy_R']:+.3f} Sh={r['sharpe']:5.2f} "
              f"DD={r['max_dd_pct']:5.1f}% t={r.get('t_stat',0):5.2f} "
              f"win={r['win_rate']:.1f}%", flush=True)
    else:
        print(f"  {label:22s} no out-of-sample trades", flush=True)
    return res


def main():
    data = load()
    print(f"timeframe {TF}   assets loaded: {len(data)}")
    for k, v in TIERS.items():
        have = [s for s in v if s in data]
        print(f"  {k:6s} {have}")
    reg = rg.btc_regime(_BTC)

    print("\n" + "#" * 72)
    print("# A. POOLED — one parameter set fitted across the whole universe")
    print("#" * 72, flush=True)
    pooled = wf(data, 12, 60, "pooled (all assets)")

    print("\n" + "#" * 72)
    print("# B. PER-TIER — each tier picks its own strategy family and thresholds")
    print("#" * 72, flush=True)
    tier_res, frames = {}, []
    for name, syms in TIERS.items():
        d = {s: data[s] for s in syms if s in data}
        if len(d) < 2:
            print(f"  {name}: too few assets"); continue
        tier_res[name] = wf(d, 12, 60, f"tier {name}")
        if tier_res[name]["report"].get("n_trades"):
            t = tier_res[name]["trades"].copy(); t["tier"] = name
            frames.append(t)

    if frames:
        rep, T = portfolio_report(frames)
        print()
        print(fmt(rep, "PER-TIER combined (3 books traded together)"))
        T.to_csv(f"out_tiers_{TF}.csv", index=False)

    print("\n" + "#" * 72)
    print("# C. VERDICT")
    print("#" * 72)
    p = pooled["report"]
    if p.get("n_trades") and frames:
        rows = [
            {"design": "pooled (1 strategy)", "n": p["n_trades"],
             "PF": round(p["profit_factor"], 2),
             "PF_CI_lo": round(p["pf_ci95"][0], 2),
             "expR": round(p["expectancy_R"], 3),
             "Sharpe": round(p["sharpe"], 2),
             "maxDD%": round(p["max_dd_pct"], 1),
             "t": round(p.get("t_stat", 0), 2)},
            {"design": "per-tier (3 strategies)", "n": rep["n_trades"],
             "PF": round(rep["profit_factor"], 2),
             "PF_CI_lo": round(rep["pf_ci95"][0], 2),
             "expR": round(rep["expectancy_R"], 3),
             "Sharpe": round(rep["sharpe"], 2),
             "maxDD%": round(rep["max_dd_pct"], 1),
             "t": round(rep.get("t_stat", 0), 2)},
        ]
        print(pd.DataFrame(rows).to_string(index=False))

    print("\n  what each tier actually selected (fold by fold):")
    for name, res in tier_res.items():
        f = res["folds"]
        if f.empty:
            continue
        fam = f["family"].value_counts().to_dict()
        mkt = f["market_filter"].value_counts().to_dict() if "market_filter" in f else {}
        lo = f["long_only"].value_counts().to_dict() if "long_only" in f else {}
        print(f"   {name:6s} families={fam}  filter={mkt}  long_only={lo}")
        f.to_csv(f"folds_tier_{name}_{TF}.csv", index=False)

    if frames:
        print("\n  per-tier out-of-sample detail:")
        rows = []
        for name, res in tier_res.items():
            r = res["report"]
            if not r.get("n_trades"):
                continue
            rows.append({"tier": name, "n": r["n_trades"],
                         "win%": round(r["win_rate"], 1),
                         "PF": round(r["profit_factor"], 2),
                         "PF_CI_lo": round(r["pf_ci95"][0], 2),
                         "expR": round(r["expectancy_R"], 3),
                         "maxDD%": round(r["max_dd_pct"], 1),
                         "t": round(r.get("t_stat", 0), 2)})
        print(pd.DataFrame(rows).to_string(index=False))

        lab = reg["regime"].reindex(pd.DatetimeIndex(T.entry_ts).normalize(),
                                    method="ffill")
        T2 = T.copy(); T2["mkt"] = lab.values
        rows = []
        for (tier, mk), g in T2.groupby(["tier", "mkt"]):
            w = g[g.pnl > 0]; l = g[g.pnl <= 0]; gl = abs(l.pnl.sum())
            rows.append({"tier": tier, "regime": mk, "n": len(g),
                         "win%": round(100 * len(w) / len(g), 1),
                         "PF": round(w.pnl.sum() / gl, 2) if gl > 0 else 999,
                         "expR": round(g.r_multiple.mean(), 3)})
        print("\n  tier x regime:")
        print(pd.DataFrame(rows).to_string(index=False))

        print("\n  per-symbol (per-tier design):")
        rows = []
        for (tier, s), g in T2.groupby(["tier", "symbol"]):
            w = g[g.pnl > 0]; l = g[g.pnl <= 0]; gl = abs(l.pnl.sum())
            rows.append({"tier": tier, "symbol": s, "n": len(g),
                         "win%": round(100 * len(w) / len(g), 1),
                         "PF": round(w.pnl.sum() / gl, 2) if gl > 0 else 999,
                         "pnl": round(g.pnl.sum(), 1)})
        print(pd.DataFrame(rows).sort_values(["tier", "pnl"], ascending=[True, False])
              .to_string(index=False))


if __name__ == "__main__":
    main()
