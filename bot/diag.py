"""Why is the bot not trading?

Reproduces exactly what runner.signal_for() sees, for every configured pair,
and prints the reason each one did or did not produce a signal. Run it locally
or as a one-off Railway command:

    python diag.py
"""
import os
import sys
import logging

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd

from botconfig import Config
from feed import Feed
import strategies as strat
import indicators as ta

logging.basicConfig(level=logging.WARNING, format="%(levelname)-7s %(message)s")


def main():
    cfg = Config()
    feed = Feed(cfg.venue, cfg.inst_type, warmup=400)
    import json
    params = {}
    if os.path.exists(cfg.params_file):
        params = json.load(open(cfg.params_file)).get(cfg.strategies[0], {})
    print(f"strategy={cfg.strategies[0]}  tf={cfg.timeframe}  "
          f"market_filter={cfg.market_filter}  pairs={len(cfg.pairs)}")
    print(f"params={params}\n")

    # ---- the market-wide gate, which is the usual culprit
    btc = feed.bars(cfg.market_filter_symbol, "1d", need=300)
    print(f"=== MARKET GATE ({cfg.market_filter_symbol} daily) ===")
    if btc is None or len(btc) < 130:
        print(f"  only {0 if btc is None else len(btc)} daily bars — gate "
              f"DISABLED (needs 130). Every signal passes through.")
        gate_now = None
    else:
        d = ta.resample(btc, "1D")
        e100 = ta.ema(d.close, 100)
        r30 = d.close.pct_change(30)
        px = float(d.close.iloc[-1])
        ev = float(e100.iloc[-1])
        rv = float(r30.iloc[-1])
        gate_now = 1.0 if (px > ev and rv > 0) else (-1.0 if (px < ev and rv < 0) else 0.0)
        print(f"  close {px:,.0f}   EMA100 {ev:,.0f}   "
              f"{'ABOVE' if px > ev else 'BELOW'} by {100*(px/ev-1):+.1f}%")
        print(f"  30-day return {100*rv:+.1f}%")
        print(f"  ==> gate = {gate_now:+.0f}   "
              f"({'longs allowed' if gate_now > 0 else 'LONGS BLOCKED'})")
        if gate_now <= 0 and params.get("long_only"):
            print("\n  *** This alone explains zero trades. The strategy is")
            print("      long-only and the gate is not positive, so every")
            print("      long signal is being zeroed before it reaches risk. ***")

    print(f"\n=== PER-PAIR (entry_n={params.get('entry_n')}, "
          f"adx_min={params.get('adx_min')}) ===")
    print(f"{'sym':10s} {'bars':>5s} {'close':>12s} {'donch_hi':>12s} "
          f"{'gap%':>7s} {'adx':>5s} {'raw':>4s} {'gated':>5s}  note")
    n_raw = n_gated = n_short = 0
    rows = []
    for s in cfg.pairs:
        try:
            df = feed.bars(s, cfg.timeframe, need=400)
        except Exception as e:
            print(f"{s:10s} FEED ERROR {e}"); continue
        if df is None or len(df) < 260:
            n_short += 1
            print(f"{s:10s} {0 if df is None else len(df):5d} "
                  f"{'':>12s} {'':>12s} {'':>7s} {'':>5s} {'':>4s} {'':>5s}  "
                  f"TOO FEW BARS (needs 260) -> never evaluated")
            continue
        try:
            sg = strat.donchian_trend_v2(df, s, **params)
        except Exception as e:
            print(f"{s:10s} STRATEGY ERROR {e}"); continue
        raw = int((sg.side != 0).sum())
        raw_now = int(sg.side.iloc[-1])
        gated = sg
        if cfg.market_filter != "none" and btc is not None and len(btc) >= 130:
            mf = strat.btc_trend_filter(btc, sg.index)
            gated = strat.apply_market_filter(sg, mf, cfg.market_filter)
        g = int((gated.side != 0).sum())
        g_now = int(gated.side.iloc[-1])
        n_raw += raw; n_gated += g

        hi, _ = ta.donchian(df, params.get("entry_n", 48))
        adx, _, _ = ta.adx(df, 14)
        close = float(df.close.iloc[-1]); h = float(hi.iloc[-1])
        gap = 100 * (close / h - 1) if h else float("nan")
        note = ""
        if g_now:
            note = "SIGNAL NOW"
        elif raw_now and not g_now:
            note = "signal killed by market gate"
        elif gap > -1.0:
            note = "within 1% of breakout"
        rows.append((s, gap))
        print(f"{s:10s} {len(df):5d} {close:12.6g} {h:12.6g} {gap:+7.2f} "
              f"{float(adx.iloc[-1]):5.1f} {raw:4d} {g:5d}  {note}")

    print(f"\n=== SUMMARY ===")
    print(f"  pairs with too little history : {n_short}")
    print(f"  historical signals, ungated   : {n_raw}")
    print(f"  historical signals, after gate: {n_gated}"
          + (f"  ({100*n_gated/n_raw:.0f}% survive)" if n_raw else ""))
    if rows:
        rows.sort(key=lambda x: -x[1])
        print("\n  closest to firing (distance to the breakout level):")
        for s, g in rows[:8]:
            print(f"    {s:10s} {g:+.2f}%")
    print("\n  Expected rate is ~1.3 trades/week across the whole universe.")
    print("  Two days of silence is the single most likely outcome, not a fault.")


if __name__ == "__main__":
    main()
