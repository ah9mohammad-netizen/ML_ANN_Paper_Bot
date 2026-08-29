"""Minimal portfolio simulator used by refit.py.

Deliberately mirrors research/engine.py's ordering so the refit scores
candidates the same way the research did:
  * a signal on bar i fills at bar i+1's open, plus slippage
  * exits check the whole bar: gap-through first, then stop, then target
  * a bar touching both stop and target is assumed to hit the STOP
  * Chandelier trail off the running extreme and the current ATR
  * taker fees both legs
"""
import numpy as np
import pandas as pd


def run_portfolio(data, sigs, cfg, bar_minutes):
    syms = [s for s in data if s in sigs]
    if not syms:
        return {"n_trades": 0}
    idx = None
    for s in syms:
        idx = data[s].index if idx is None else idx.union(data[s].index)
    idx = idx.sort_values()

    arr = {s: {c: data[s][c].to_numpy(float) for c in
               ("open", "high", "low", "close")} for s in syms}
    sa = {s: {c: sigs[s][c].to_numpy(float) if c not in ("tag", "regime")
              else sigs[s][c].to_numpy()
              for c in ("side", "sl", "tp", "ref_price", "atr",
                        "trail_atr", "trail_start_r", "be_at_r")} for s in syms}
    kpos = {s: {t: k for k, t in enumerate(data[s].index)} for s in syms}

    eq = cfg.starting_equity
    peak = eq
    open_pos = {}
    trades = []
    curve = []
    slip_e = cfg.slip_entry_bps / 1e4
    slip_s = cfg.slip_stop_bps / 1e4

    for ts in idx:
        kmap = {}
        for s in syms:
            k = kpos[s].get(ts)
            if k is not None:
                kmap[s] = k

        # ---- exits
        for s in list(open_pos):
            k = kmap.get(s)
            if k is None:
                continue
            p = open_pos[s]
            a = arr[s]
            o, h, l, c = a["open"][k], a["high"][k], a["low"][k], a["close"][k]
            side = p["side"]
            px = reason = None
            if side > 0 and o <= p["sl"]:
                px, reason = o * (1 - slip_s), "SL_GAP"
            elif side < 0 and o >= p["sl"]:
                px, reason = o * (1 + slip_s), "SL_GAP"
            elif (l <= p["sl"] if side > 0 else h >= p["sl"]):
                px = p["sl"] * (1 - slip_s) if side > 0 else p["sl"] * (1 + slip_s)
                reason = "SL"
            elif (h >= p["tp"] if side > 0 else l <= p["tp"]):
                px, reason = p["tp"], "TP"
            if reason:
                gross = side * (px - p["entry"]) * p["qty"]
                fee = abs(px * p["qty"]) * cfg.taker_fee
                pnl = gross - fee - p["entry_fee"]
                eq += pnl
                trades.append({"pnl": pnl,
                               "r": pnl / p["r_unit"] if p["r_unit"] else 0.0})
                del open_pos[s]
                continue
            # trail
            tr = p["trail"]
            if tr:
                p["peak"] = max(p["peak"], h) if side > 0 else min(p["peak"], l)
                av = sa[s]["atr"][k]
                av = av if np.isfinite(av) and av > 0 else p["atr0"]
                risk = abs(p["entry"] - p["sl0"]) or 1e-9
                if side * (c - p["entry"]) / risk >= p["tstart"] and av > 0:
                    cand = p["peak"] - side * tr * av
                    p["sl"] = max(p["sl"], cand) if side > 0 else min(p["sl"], cand)

        # ---- mark
        unreal = sum(p["side"] * (arr[s]["close"][kmap[s]] - p["entry"]) * p["qty"]
                     for s, p in open_pos.items() if s in kmap)
        e = eq + unreal
        peak = max(peak, e)
        curve.append(e)
        if peak > 0 and (peak - e) / peak >= 0.60:
            break

        # ---- entries
        for s, k in kmap.items():
            if k == 0 or s in open_pos or len(open_pos) >= cfg.max_open:
                continue
            side = sa[s]["side"][k - 1]
            if not side:
                continue
            side = int(side)
            ref = sa[s]["ref_price"][k - 1]
            entry = arr[s]["open"][k] * (1 + side * slip_e)
            sl = entry - (ref - sa[s]["sl"][k - 1])
            tp = entry + (sa[s]["tp"][k - 1] - ref)
            if side > 0 and not sl < entry < tp:
                continue
            if side < 0 and not tp < entry < sl:
                continue
            stop_d = abs(entry - sl)
            if stop_d <= 0:
                continue
            qty = eq * cfg.risk_per_trade / stop_d
            notional = min(qty * entry, eq * cfg.max_notional_per_trade)
            gross_n = sum(p["qty"] * p["entry"] for p in open_pos.values())
            notional = min(notional, max(0.0, eq * cfg.max_gross_notional - gross_n))
            if notional < 10:
                continue
            qty = notional / entry
            fee = notional * cfg.taker_fee
            eq -= fee
            atr0 = sa[s]["atr"][k - 1]
            open_pos[s] = {"side": side, "entry": entry, "qty": qty, "sl": sl,
                           "tp": tp, "sl0": sl, "entry_fee": fee,
                           "r_unit": qty * stop_d, "peak": entry,
                           "trail": sa[s]["trail_atr"][k - 1],
                           "tstart": sa[s]["trail_start_r"][k - 1],
                           "atr0": atr0 if np.isfinite(atr0) else 0.0}

    if not trades:
        return {"n_trades": 0}
    t = pd.DataFrame(trades)
    w, l = t[t.pnl > 0], t[t.pnl <= 0]
    gl = abs(l.pnl.sum())
    c = pd.Series(curve)
    dd = ((c.cummax() - c) / c.cummax()).max()
    ret = c.pct_change().fillna(0)
    ann = np.sqrt(365 * 1440 / bar_minutes)
    return {
        "n_trades": len(t),
        "win_rate": 100 * len(w) / len(t),
        "profit_factor": (w.pnl.sum() / gl) if gl > 0 else float("inf"),
        "expectancy_R": t.r.mean(),
        "std_R": t.r.std(),
        "max_dd_pct": 100 * dd,
        "sharpe": (ret.mean() / ret.std() * ann) if ret.std() > 0 else 0.0,
        "net_pnl": t.pnl.sum(),
    }
