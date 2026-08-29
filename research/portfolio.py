"""Weights-based portfolio backtester.

The stop-loss/take-profit engine tests *discrete trades*. The documented crypto
momentum edge is not a discrete trade — it is a continuously held, volatility-
targeted position that is rebalanced on a schedule. That needs a different
simulator, so here it is.

  * signal_t is computed from bars up to and including t (strictly causal)
  * the weight it implies is applied from t+1 onward — you cannot trade on a
    close you have only just observed
  * costs are charged on TURNOVER: |w_t - w_{t-1}| * equity * (fee + slippage)
  * funding is charged on the held notional at each settlement
  * per-asset weights are volatility-scaled to a target, then the whole book is
    scaled so gross exposure respects a leverage cap
"""
import numpy as np
import pandas as pd


def vol_target_weights(sig: pd.DataFrame, vol: pd.DataFrame, target_vol=0.40,
                       max_w=0.35, gross_cap=1.5):
    """sig: -1/0/+1 per asset per bar. vol: annualised realised vol per asset."""
    w = sig * (target_vol / vol.replace(0, np.nan))
    w = w.clip(-max_w, max_w).fillna(0.0)
    gross = w.abs().sum(axis=1)
    scale = (gross_cap / gross).clip(upper=1.0).replace([np.inf, -np.inf], 1.0)
    return w.mul(scale, axis=0).fillna(0.0)


def run(prices: pd.DataFrame, weights: pd.DataFrame, fee=0.0005, slip=0.0002,
        funding: pd.DataFrame = None, bars_per_year=365, start_equity=1000.0,
        rebalance_every=1):
    """prices/weights indexed identically. Returns (equity_series, stats)."""
    prices = prices.sort_index()
    weights = weights.reindex(prices.index).fillna(0.0)
    # a weight decided on bar t is held over bar t+1
    w = weights.shift(1).fillna(0.0)
    if rebalance_every > 1:
        keep = (np.arange(len(w)) % rebalance_every) == 0
        w = w.where(pd.Series(keep, index=w.index), np.nan).ffill().fillna(0.0)

    ret = prices.pct_change().fillna(0.0)
    gross_ret = (w * ret).sum(axis=1)

    turn = (w - w.shift(1).fillna(0.0)).abs().sum(axis=1)
    cost = turn * (fee + slip)

    fund_cost = pd.Series(0.0, index=prices.index)
    if funding is not None and len(funding):
        f = funding.reindex(prices.index).fillna(0.0)
        fund_cost = (w * f).sum(axis=1)      # long pays a positive rate

    net = gross_ret - cost - fund_cost
    eq = start_equity * (1 + net).cumprod()

    dd = (eq.cummax() - eq) / eq.cummax()
    days = max((eq.index[-1] - eq.index[0]).days, 1)
    ann = np.sqrt(bars_per_year)
    sharpe = net.mean() / net.std() * ann if net.std() > 0 else 0.0
    dsd = net[net < 0].std()
    sortino = net.mean() / dsd * ann if dsd and dsd > 0 else 0.0
    cagr = (eq.iloc[-1] / start_equity) ** (365 / days) - 1
    stats = {
        "start": str(eq.index[0])[:10], "end": str(eq.index[-1])[:10],
        "total_return_pct": 100 * (eq.iloc[-1] / start_equity - 1),
        "cagr_pct": 100 * cagr,
        "vol_pct": 100 * net.std() * ann,
        "sharpe": sharpe, "sortino": sortino,
        "max_dd_pct": 100 * dd.max(),
        "calmar": cagr / dd.max() if dd.max() > 0 else np.inf,
        "avg_gross_exposure": w.abs().sum(axis=1).mean(),
        "turnover_ann": turn.mean() * bars_per_year,
        "cost_drag_ann_pct": 100 * cost.mean() * bars_per_year,
        "funding_drag_ann_pct": 100 * fund_cost.mean() * bars_per_year,
        "pct_bars_long": 100 * (w.sum(axis=1) > 0).mean(),
        "hit_rate_pct": 100 * (net > 0).mean(),
        "final_equity": eq.iloc[-1],
        "t_stat": net.mean() / (net.std() / np.sqrt(len(net))) if net.std() > 0 else 0.0,
    }
    return eq, stats, net


def fmt(s, title=""):
    return "\n".join([
        f"── {title} " + "─" * max(0, 56 - len(title)),
        f"  {s['start']} → {s['end']}   gross exposure {s['avg_gross_exposure']:.2f}x",
        f"  CAGR {s['cagr_pct']:+.1f}%   vol {s['vol_pct']:.1f}%   "
        f"total {s['total_return_pct']:+.1f}%",
        f"  Sharpe {s['sharpe']:.2f}   Sortino {s['sortino']:.2f}   "
        f"t-stat {s['t_stat']:.2f}",
        f"  max DD {s['max_dd_pct']:.1f}%   Calmar {s['calmar']:.2f}   "
        f"bars up {s['hit_rate_pct']:.1f}%",
        f"  turnover {s['turnover_ann']:.1f}x/yr → cost drag "
        f"{s['cost_drag_ann_pct']:.2f}%/yr, funding {s['funding_drag_ann_pct']:+.2f}%/yr",
    ])
