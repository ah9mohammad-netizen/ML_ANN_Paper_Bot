import warnings; warnings.filterwarnings('ignore')
import pandas as pd, okx_data as od, strategies as st, regimes as rg
from engine import Backtester, RiskConfig, ExecConfig, fmt

SYMS=['BTC','ETH','SOL','LINK','XRP']
data={s:pd.read_parquet(f'data/SWAP_{s}_15m.parquet') for s in SYMS}
print({s:len(d) for s,d in data.items()},flush=True)
fund={}
for s in SYMS:
    try:
        f=od.funding(s,'2022-01-01')
        if len(f): fund[s]=f
    except Exception as e: print('funding',s,e)
print('funding series:',{s:len(v) for s,v in fund.items()},flush=True)

reg=rg.btc_regime(data['BTC'])
print('regime mix:',rg.summarise(reg))
for lab,a,b,n in rg.segments(reg,30): print(f'  {lab:5s} {str(a)[:10]} -> {str(b)[:10]} ({n}d)')

sig={s:st.legacy_smc(d,max_hold_bars=96) for s,d in data.items()}
print('raw signals:',{s:int((v.side!=0).sum()) for s,v in sig.items()},flush=True)

risk=RiskConfig(starting_equity=1000,risk_per_trade=0.01,leverage=10,max_open=6,
                max_gross_notional=2.5,max_notional_per_trade=1.5,
                max_daily_loss=1.0,max_drawdown_halt=0.99)
MODES={
 'A_repo_assumptions':dict(taker_fee=0.0004,slip_entry_bps=2.0,slip_stop_bps=0.0,
    use_funding=False,fill_at_signal_close=True,exits_on_close_only=True,model_liquidation=False),
 'B_wicks_visible':dict(taker_fee=0.0004,slip_entry_bps=2.0,slip_stop_bps=0.0,
    use_funding=False,fill_at_signal_close=True,pessimistic_ambiguous_bar=True),
 'C_honest':dict(taker_fee=0.0005,slip_entry_bps=2.0,slip_stop_bps=5.0,
    use_funding=True,pessimistic_ambiguous_bar=True),
}
for name,kw in MODES.items():
    bt=Backtester(data,sig,risk=risk,ex=ExecConfig(**kw),funding=fund,bar_minutes=15)
    r=bt.run(); print(); print(fmt(r,f'LEGACY SMC 15m · {name}'),flush=True)
    if name=='C_honest':
        t=bt.trades_df(); t.to_csv('out_legacy15.csv',index=False)
        lab=reg['regime'].reindex(pd.DatetimeIndex(t.entry_ts).normalize(),method='ffill')
        t['mkt']=lab.values
        rows=[]
        for k,g in t.groupby('mkt'):
            w=g[g.pnl>0]; l=g[g.pnl<=0]; gl=abs(l.pnl.sum())
            rows.append({'regime':k,'n':len(g),'win%':round(100*len(w)/len(g),1),
                         'PF':round(w.pnl.sum()/gl,2) if gl>0 else 999,
                         'expR':round(g.r_multiple.mean(),3),'pnl':round(g.pnl.sum(),1)})
        print('\n  by market regime:'); print(pd.DataFrame(rows).to_string(index=False))
        print('\n  per symbol:')
        print(t.groupby('symbol').agg(n=('pnl','size'),wr=('pnl',lambda x:round(100*(x>0).mean(),1)),
              pnl=('pnl','sum'),expR=('r_multiple','mean')).round(2).to_string())
