"""Honest baseline: run the CURRENT repo strategy through a correct backtester.

Three fill regimes are compared so we can attribute the repo's reported edge:
  A) 'repo'   – fill at the signal bar's close, exits checked only on later
                bar CLOSES (what paper_engine.py actually does), 4bps drag
  B) 'wick'   – same fills, but exits checked against intrabar high/low
  C) 'honest' – next-bar-open fill, intrabar exits, real taker fees + funding
"""
import sys, json, warnings
import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

import okx_data as od
import strategies as st
import regimes as rg
from engine import Backtester, RiskConfig, ExecConfig, build_report, fmt

SMC_PAIRS = ["BTC", "ETH", "SOL", "LINK", "NEAR", "AVAX", "BNB"]
CCI_PAIRS_15 = ["SUI", "APT", "DOGE"]
CCI_PAIRS_5 = ["HYPE", "PEPE", "WIF", "FET"]


def load(syms, tf, start):
    out = {}
    for s in syms:
        try:
            d = od.get(s, tf, start)
            if len(d) > 5000:
                out[s] = d
        except Exception as e:
            print("  skip", s, e)
    return out


def legacy_signals(data, tf):
    sig = {}
    for s, d in data.items():
        if s in SMC_PAIRS:
            sig[s] = st.legacy_smc(d, max_hold_bars=96 if tf == "15m" else 288)
        else:
            mh = 48 if tf == "15m" else 144
            sig[s] = st.legacy_cci_bb(d, s, max_hold_bars=mh)
    return sig


MODES = {
    # name: (exec cfg, fill_at_close, intrabar)
    "A_repo_assumptions": dict(taker_fee=0.0004, slip_entry_bps=2.0,
                               slip_stop_bps=0.0, use_funding=False,
                               fill_at_signal_close=True, exits_on_close_only=True,
                               model_liquidation=False),
    "B_intrabar_wicks":   dict(taker_fee=0.0004, slip_entry_bps=2.0,
                               slip_stop_bps=0.0, use_funding=False,
                               fill_at_signal_close=True,
                               pessimistic_ambiguous_bar=True),
    "C_honest":           dict(taker_fee=0.0005, slip_entry_bps=2.0,
                               slip_stop_bps=5.0, use_funding=True,
                               pessimistic_ambiguous_bar=True),
}


def main(tf="15m", start="2022-01-01"):
    syms = SMC_PAIRS + CCI_PAIRS_15 if tf == "15m" else CCI_PAIRS_5
    print(f"loading {tf} …", flush=True)
    data = load(syms, tf, start)
    print("  loaded:", {k: len(v) for k, v in data.items()}, flush=True)
    if not data:
        print("no data"); return
    fund = {}
    for s in data:
        try:
            f = od.funding(s, start)
            if len(f): fund[s] = f
        except Exception:
            pass

    sig = legacy_signals(data, tf)
    n_sig = {s: int((v.side != 0).sum()) for s, v in sig.items()}
    print("  raw signals:", n_sig, "total", sum(n_sig.values()), flush=True)

    bt = None
    for name, kw in MODES.items():
        risk = RiskConfig(starting_equity=1000, risk_per_trade=0.01,
                          leverage=10, max_open=6,
                          max_gross_notional=2.5, max_notional_per_trade=1.5,
                          max_daily_loss=1.0, max_drawdown_halt=0.99)
        bt = Backtester(data, sig, risk=risk, ex=ExecConfig(**kw),
                        funding=fund, bar_minutes=15 if tf == "15m" else 5)
        r = bt.run()
        print()
        print(fmt(r, f"LEGACY {tf} · {name}"), flush=True)
        if name == "C_honest":
            t = bt.trades_df()
            t.to_csv(f"/home/claude/quant/out_legacy_{tf}.csv", index=False)
            bt.equity_df().to_csv(f"/home/claude/quant/eq_legacy_{tf}.csv")
            print("\n  per-symbol (honest):")
            g = t.groupby("symbol").agg(
                n=("pnl", "size"), wr=("pnl", lambda x: 100*(x > 0).mean()),
                pnl=("pnl", "sum"), expR=("r_multiple", "mean"))
            print(g.round(2).to_string())
    return bt


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "15m",
         sys.argv[2] if len(sys.argv) > 2 else "2022-01-01")
