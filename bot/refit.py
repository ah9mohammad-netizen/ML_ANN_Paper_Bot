"""Quarterly parameter refit.

The walk-forward result the strategy was validated on (out-of-sample Sharpe
1.26, PF 2.13) was produced by an ADAPTIVE system: parameters were re-chosen on
the trailing 365 days every 110 days, then applied unchanged to the next unseen
window. Freezing one parameter set for all time gives materially less
(PF ~1.5, Sharpe ~0.77). So the refit is not an optional extra — it is part of
the strategy that was measured.

Run this on a schedule (Railway cron, every ~110 days). It:
  1. downloads trailing history for the configured universe
  2. searches the same parameter grid the research used
  3. scores candidates on the t-statistic of expectancy, drawdown-penalised
  4. writes params.json ONLY if the winner beats a minimum quality bar
  5. reports what changed to Telegram

It never touches open positions. The running bot picks up the new file on its
next restart, or immediately if RELOAD_PARAMS_EACH_CYCLE is on.
"""
import json
import os
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from botconfig import Config
import indicators as ta
import strategies as strat
from feed import Feed
from tg import Telegram

# the same grid the research searched — do not widen it without redoing the
# out-of-sample work; every extra dimension is another way to fit the past
GRID = {
    "donchian": dict(entry_n=[20, 30, 48, 72], sl_atr=[2.0, 3.0],
                     trail_atr=[3.0, 4.5], trail_start_r=[0.0, 0.5],
                     adx_min=[0, 18], ema_filter=[0, 200]),
    "squeeze": dict(squeeze_n=[96, 192], squeeze_q=[0.2, 0.35],
                    entry_n=[20, 30, 48], sl_atr=[2.0, 3.0],
                    trail_atr=[3.0, 4.5], trail_start_r=[0.0, 0.5],
                    ema_filter=[0, 200]),
}
FN = {"donchian": strat.donchian_trend_v2, "squeeze": strat.vol_breakout_v2}

MIN_TRADES = 25          # below this the search is fitting noise
MIN_T_STAT = 1.0         # keep the incumbent unless the winner is at least this


def log(*a):
    print(datetime.now(timezone.utc).strftime("%F %T"), *a, flush=True)


def fetch(cfg, feed, need_bars):
    data, btc = {}, None
    for s in cfg.pairs:
        try:
            d = feed.bars(s, cfg.timeframe, need=need_bars)
        except Exception as e:
            log(f"  {s}: {e}"); continue
        if d is not None and len(d) >= 400:
            data[s] = d
    try:
        btc = feed.bars(cfg.market_filter_symbol, "1d", need=400)
    except Exception:
        btc = None
    return data, btc


def build(fam, df, symbol, btc, cfg, params):
    sg = FN[fam](df, symbol, long_only=True, max_hold_bars=0, **params)
    if cfg.market_filter != "none" and btc is not None and len(btc) >= 130:
        sg = strat.apply_market_filter(sg, strat.btc_trend_filter(btc, sg.index),
                                       cfg.market_filter)
    return sg


def simulate(data, sigs, cfg, bar_minutes):
    """Minimal, self-contained trade simulation matching the research engine's
    ordering: next-bar-open fill, wick-aware exits, stop wins ambiguous bars."""
    from engine_lite import run_portfolio
    return run_portfolio(data, sigs, cfg, bar_minutes)


def score(rep):
    n = rep.get("n_trades", 0)
    if n < MIN_TRADES:
        return -9e9
    sd = rep.get("std_R") or 1.0
    t = rep["expectancy_R"] / (sd / np.sqrt(n))
    dd = rep.get("max_dd_pct", 100) / 100
    return t * (1.0 if dd <= 0.30 else (0.30 / dd) ** 2)


def main():
    cfg = Config()
    tg = Telegram(cfg.telegram_token, cfg.telegram_chat_id, None, None, None,
                  cfg, __import__("logging").getLogger("refit"))
    feed = Feed(cfg.venue, cfg.inst_type, warmup=1200)
    bar_min = {"15m": 15, "30m": 30, "1h": 60, "2h": 120, "4h": 240,
               "8h": 480, "1d": 1440}[cfg.timeframe]
    need = int(365 * 24 * 60 / bar_min) + 300      # ~1 year plus warmup

    log(f"refit: {len(cfg.pairs)} pairs on {cfg.timeframe}, need {need} bars")
    data, btc = fetch(cfg, feed, need)
    log(f"  usable: {len(data)} symbols")
    if len(data) < 5:
        tg.send("Refit aborted: fewer than 5 symbols returned usable history.")
        return 1

    fam = cfg.strategies[0]
    if fam not in GRID:
        tg.send(f"Refit aborted: no grid defined for strategy {fam!r}.")
        return 1

    import itertools
    keys = list(GRID[fam])
    combos = [dict(zip(keys, c)) for c in itertools.product(*GRID[fam].values())]
    log(f"  searching {len(combos)} candidates of {fam}")

    best, best_s, best_rep = None, -9e18, None
    t0 = time.time()
    for i, p in enumerate(combos):
        try:
            sigs = {s: build(fam, d, s, btc, cfg, p) for s, d in data.items()}
            rep = simulate(data, sigs, cfg, bar_min)
        except Exception as e:
            continue
        s = score(rep)
        if s > best_s:
            best_s, best, best_rep = s, p, rep
        if i % 20 == 0:
            log(f"   {i}/{len(combos)}  best score {best_s:.2f}")
    log(f"  search done in {time.time()-t0:.0f}s")

    if best is None or best_s < MIN_T_STAT:
        msg = (f"*Refit produced nothing usable* — best score {best_s:.2f} "
               f"below the {MIN_T_STAT} bar. Keeping the existing parameters.\n"
               "This is a signal worth heeding: the strategy has not worked on "
               "the trailing year.")
        log(msg); tg.send(msg)
        return 0

    if best_rep.get("funding_warning"):
        log("WARNING:", best_rep["funding_warning"])
        tg.send("Refit warning: no historical funding costs were supplied. "
                "This is not a fully costed validation result.")

    path = cfg.params_file
    old = {}
    if os.path.exists(path):
        try:
            old = json.load(open(path))
        except Exception:
            old = {}
    prev = old.get(fam, {})
    new = dict(old)
    new["_refit_utc"] = datetime.now(timezone.utc).isoformat()
    new["_refit_window_bars"] = need
    new["_refit_symbols"] = sorted(data)
    new["_refit_score"] = round(best_s, 3)
    new["_refit_funding_warning"] = best_rep.get("funding_warning", "")
    new["_refit_insample"] = {k: (round(v, 3) if isinstance(v, float) else v)
                              for k, v in best_rep.items()
                              if k in ("n_trades", "win_rate", "profit_factor",
                                       "expectancy_R", "max_dd_pct", "sharpe")}
    new[fam] = dict(best, long_only=True, max_hold_bars=0)
    tmp = path + ".tmp"
    json.dump(new, open(tmp, "w"), indent=2)
    os.replace(tmp, path)

    changed = {k: (prev.get(k), v) for k, v in new[fam].items() if prev.get(k) != v}
    tg.send(
        f"*Parameters refit* ({fam} @ {cfg.timeframe}, {len(data)} symbols)\n"
        f"score {best_s:.2f}\n"
        f"in-sample: n={best_rep['n_trades']} "
        f"PF={best_rep['profit_factor']:.2f} "
        f"win={best_rep['win_rate']:.1f}% "
        f"DD={best_rep['max_dd_pct']:.1f}%\n"
        f"changed: {changed if changed else 'nothing'}\n"
        "_These are in-sample fit numbers, not a forecast._")
    log("wrote", path, new[fam])
    return 0


if __name__ == "__main__":
    sys.exit(main())
