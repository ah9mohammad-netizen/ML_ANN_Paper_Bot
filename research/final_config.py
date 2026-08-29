import warnings, json; warnings.filterwarnings('ignore')
import numpy as np, pandas as pd, strategies as st, regimes as rg, indicators as ta
from engine import Backtester, RiskConfig, ExecConfig, fmt
SYMS=["BTC","ETH","SOL","LINK","XRP","DOGE","AVAX","NEAR","APT","SUI"]
RISK=RiskConfig(starting_equity=1000,risk_per_trade=0.0075,leverage=4,max_open=5,
                max_gross_notional=2.0,max_notional_per_trade=0.6,
                max_daily_loss=0.06,max_drawdown_halt=0.35)
EX=ExecConfig(taker_fee=0.0005,slip_entry_bps=2.0,slip_stop_bps=5.0,
              use_funding=True,pessimistic_ambiguous_bar=True,chandelier=True)
BTC=pd.read_parquet('data/SWAP_BTC_15m.parquet'); reg=rg.btc_regime(BTC)
data={}
for s in SYMS:
    import os
    p=f'data/SWAP_{s}_15m.parquet'
    if os.path.exists(p):
        d=pd.read_parquet(p)
        if len(d)>=30000: data[s]=ta.resample(d,'8h').loc[pd.Timestamp('2022-01-01',tz='UTC'):]
F=st.btc_trend_filter(BTC, list(data.values())[0].index)
def mk(p, lo, filt):
    out={}
    for s,d in data.items():
        g=st.donchian_trend_v2(d,s,long_only=lo,**p)
        if filt: g=st.apply_market_filter(g, st.btc_trend_filter(BTC,d.index),'align')
        out[s]=g
    return out
base=dict(entry_n=48,sl_atr=3.0,trail_atr=4.5,trail_start_r=0.5,adx_min=0,max_hold_bars=0)
for name,p,lo,filt in [
    ("A both sides + filter", dict(base,ema_filter=0), False, True),
    ("B LONG-ONLY + filter",  dict(base,ema_filter=0), True,  True),
    ("C LONG-ONLY + filter + ema200", dict(base,ema_filter=200), True, True),
    ("D LONG-ONLY no filter", dict(base,ema_filter=200), True, False),
]:
    bt=Backtester(data,mk(p,lo,filt),risk=RISK,ex=EX,bar_minutes=480); r=bt.run()
    if not r.get('n_trades'): print(name,'no trades'); continue
    t=bt.trades_df()
    lab=reg['regime'].reindex(pd.DatetimeIndex(t.entry_ts).normalize(),method='ffill')
    t['mkt']=lab.values
    rr=[]
    for k,g in t.groupby('mkt'):
        w=g[g.pnl>0]; l=g[g.pnl<=0]; gl=abs(l.pnl.sum())
        rr.append(f"{k}:PF{(w.pnl.sum()/gl if gl>0 else 99):.2f}/n{len(g)}")
    yr=t.groupby(pd.DatetimeIndex(t.entry_ts).year).pnl.sum().round(0).to_dict()
    print(f"{name:34s} n={r['n_trades']:4d} PF={r['profit_factor']:.2f} "
          f"CI[{r['pf_ci95'][0]:.2f},{r['pf_ci95'][1]:.2f}] expR={r['expectancy_R']:+.3f} "
          f"Sh={r['sharpe']:.2f} DD={r['max_dd_pct']:.1f}% t={r.get('t_stat',0):.2f} "
          f"win={r['win_rate']:.1f}%")
    print(f"{'':34s} regimes: {'  '.join(rr)}")
    print(f"{'':34s} by year: {yr}")
