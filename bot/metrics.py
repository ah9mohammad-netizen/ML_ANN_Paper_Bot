"""Performance metrics computed from the live store.

Reports the numbers that decide whether an edge exists — expectancy in R with a
confidence interval, profit factor with a bootstrap CI, drawdown on the real
mark-to-market curve — alongside win rate, so win rate can never be read on its
own again.
"""
import math
import numpy as np
import pandas as pd


def from_store(store, starting_equity=None, post_fix_only=False):
    rows = store.closed()
    excluded = 0
    if post_fix_only:
        included = [r for r in rows if not r.get("legacy_execution")
                    and not r.get("history_warning")]
        excluded = len(rows) - len(included)
        rows = included
    eq = pd.DataFrame(store.equity_curve())
    out = {"n": len(rows), "post_fix_only": post_fix_only, "excluded_trades": excluded,
           "legacy_trades": sum(bool(r.get("legacy_execution")) for r in rows),
           "history_warning_trades": sum(bool(r.get("history_warning")) for r in rows)}
    if len(eq) and "ts" in eq:
        start = pd.to_datetime(eq.ts, utc=True).min()
        if post_fix_only and rows:
            start = pd.to_datetime([r["opened_at"] for r in rows], utc=True).min()
        weeks = (pd.Timestamp.now(tz="UTC") - start).total_seconds() / (7 * 86400)
        if weeks > 0:
            out["observed_trades_per_week"] = len(rows) / weeks
    if not rows:
        return out
    t = pd.DataFrame(rows)
    t["pnl"] = pd.to_numeric(t.pnl, errors="coerce").fillna(0.0)
    t["r_multiple"] = pd.to_numeric(t.r_multiple, errors="coerce").fillna(0.0)
    w = t[t.pnl > 0]; l = t[t.pnl <= 0]
    gw, gl = w.pnl.sum(), abs(l.pnl.sum())
    R = t.r_multiple

    out.update({
        "win_rate": 100 * len(w) / len(t),
        "profit_factor": (gw / gl) if gl > 0 else float("inf"),
        "expectancy_R": R.mean(),
        "expectancy_usd": t.pnl.mean(),
        "net_pnl": t.pnl.sum(),
        "avg_win": w.pnl.mean() if len(w) else 0.0,
        "avg_loss": l.pnl.mean() if len(l) else 0.0,
        "payoff": (w.pnl.mean() / abs(l.pnl.mean())) if len(l) and l.pnl.mean() else float("inf"),
        "fees": pd.to_numeric(t.fees, errors="coerce").fillna(0).sum(),
        "funding": pd.to_numeric(t.funding, errors="coerce").fillna(0).sum(),
        "exit_mix": t.exit_reason.value_counts().to_dict(),
        "by_tag": t.groupby("tag").pnl.agg(["size", "sum"]).round(2).to_dict("index"),
        "by_symbol": t.groupby("symbol").pnl.agg(["size", "sum"]).round(2).to_dict("index"),
    })
    if len(t) > 5 and R.std() > 0:
        se = R.std() / math.sqrt(len(t))
        out["t_stat"] = R.mean() / se
    if len(t) >= 6:
        rng = np.random.default_rng(7)
        p = t.pnl.values
        idx = rng.integers(0, len(p), size=(2000, len(p)))
        s = p[idx]
        pf = np.where(np.abs(np.where(s <= 0, s, 0).sum(1)) > 0,
                      np.where(s > 0, s, 0).sum(1) /
                      np.maximum(np.abs(np.where(s <= 0, s, 0).sum(1)), 1e-9),
                      np.inf)
        # Keep all-win resamples as infinite PF; dropping them biases the CI.
        lo, hi = np.quantile(pf, [0.025, 0.975], method="inverted_cdf")
        out["pf_ci95"] = (float(lo), float(hi))
        rm = R.to_numpy()[idx].mean(1)
        out["expectancy_R_ci95"] = tuple(float(x) for x in np.quantile(rm, [0.025, 0.975]))
        out["bootstrap_positive_fraction"] = float((s.mean(1) > 0).mean())
        out["ci_method"] = "IID trade bootstrap; does not model correlated trades"

    if len(eq):
        e = pd.to_numeric(eq.equity, errors="coerce").dropna()
        if len(e) > 2:
            dd = (e.cummax() - e) / e.cummax()
            out["max_dd_pct"] = 100 * dd.max()
            out["current_dd_pct"] = 100 * dd.iloc[-1]
            out["equity"] = float(e.iloc[-1])
    return out


def verdict(m, min_trades=100):
    """Is there evidence of an edge? Deliberately hard to satisfy."""
    n = m.get("n", 0)
    if m.get("legacy_trades", 0) or m.get("history_warning_trades", 0):
        return ("UNVALIDATED EXECUTION DATA",
                f"{m.get('legacy_trades', 0)} legacy trade(s), "
                f"{m.get('history_warning_trades', 0)} with missing-history flags. "
                "Fees are normalized, but fills have not been repaired. "
                "Do not infer an edge from this mixed sample.")
    if n < min_trades:
        rate = m.get("observed_trades_per_week", 0)
        timing = (f" At the observed {rate:.2f} closed trades/week, "
                  f"roughly {(min_trades-n)/rate:.0f} more weeks if that pace persists."
                  if rate > 0 else "")
        return ("INSUFFICIENT DATA",
                f"{n}/{min_trades} closed trades. The sample cannot establish a "
                f"reliable edge; 100 is a checkpoint, not proof.{timing}")
    t = m.get("t_stat", 0)
    pf = m.get("profit_factor", 0)
    dd = m.get("max_dd_pct", 100)
    lo = m.get("pf_ci95", (0, 0))[0]
    if t > 2.5 and lo > 1.1 and dd < 20:
        return ("EDGE SUPPORTED",
                f"t={t:.2f}, PF 95% CI lower bound {lo:.2f}, max DD {dd:.1f}%. "
                "Still paper — size up slowly if you go live.")
    if t > 1.5 and lo > 0.95:
        return ("PROMISING, NOT PROVEN",
                f"t={t:.2f}, PF CI lower bound {lo:.2f}. Keep paper trading; "
                "this is not yet distinguishable from luck.")
    return ("NO EDGE DEMONSTRATED",
            f"t={t:.2f}, PF {pf:.2f} (CI lower {lo:.2f}), max DD {dd:.1f}%. "
            "Do not risk money on this configuration.")


def fmt(m):
    if not m.get("n"):
        return (f"No post-fix closed trades yet ({m.get('excluded_trades', 0)} excluded)."
                if m.get("post_fix_only") else "No closed trades yet.")
    v, why = verdict(m)
    L = [f"n={m['n']}  win {m['win_rate']:.1f}%  payoff {m['payoff']:.2f}",
         f"PF {m['profit_factor']:.2f}" + (
             f"  CI95 [{m['pf_ci95'][0]:.2f}, {m['pf_ci95'][1]:.2f}]"
             if "pf_ci95" in m else ""),
         f"expectancy {m['expectancy_R']:+.3f} R ({m['expectancy_usd']:+.2f} USDT)" + (
             f"  CI95 [{m['expectancy_R_ci95'][0]:+.3f}, {m['expectancy_R_ci95'][1]:+.3f}]"
             if "expectancy_R_ci95" in m else ""),
         f"net {m['net_pnl']:+.2f}  fees {m['fees']:.2f}  funding cost {m['funding']:+.2f} (+paid/-received)",
         f"account max DD {m.get('max_dd_pct', 0):.1f}%  now {m.get('current_dd_pct', 0):.1f}%",
         f"t-stat {m.get('t_stat', 0):.2f}",
         (f"positive bootstrap means {m['bootstrap_positive_fraction']:.2f} "
          "(not P(true edge>0))" if "bootstrap_positive_fraction" in m else
          "bootstrap: insufficient trades"),
         "CI method: IID trade bootstrap; correlation is not accounted for",
         f"exits {m['exit_mix']}",
         f"→ {v}: {why}"]
    if m.get("post_fix_only"):
        L.insert(0, f"Post-fix trades only; {m.get('excluded_trades', 0)} excluded. "
                    "DD is the whole-account curve, not a separate strategy curve.")
    return "\n".join(L)
