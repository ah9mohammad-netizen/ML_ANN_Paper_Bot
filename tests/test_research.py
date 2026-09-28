import numpy as np
import pandas as pd
import pytest
from research.engine import Backtester, RiskConfig, ExecConfig
from engine_lite import run_portfolio
from conftest import signal
import strategies


def market(n=4):
    ix=pd.date_range('2024-01-01',periods=n,freq='8h',tz='UTC')
    df=pd.DataFrame(dict(open=100.,high=101.,low=99.,close=100.,volume=100.),index=ix)
    sg=pd.DataFrame(dict(side=0,sl=95.,tp=120.,ref_price=100.,max_hold_bars=0,
        atr=1.,risk_mult=1.,be_at_r=0.,trail_atr=0.,trail_start_r=0.,tag='TEST',regime=''),index=ix)
    sg.iloc[0,sg.columns.get_loc('side')]=1
    return df,sg


def engine(df,sg,**kw):
    return Backtester({'BTC':df},{'BTC':sg},
        risk=RiskConfig(starting_equity=1000,risk_per_trade=.01,
                        leverage=10,max_notional_per_trade=.35),
        ex=ExecConfig(slip_entry_bps=0,slip_stop_bps=0,chandelier=True),
        bar_minutes=480,**kw)


def test_entry_bar_stop_is_not_skipped():
    df,sg=market(); df.iloc[1,df.columns.get_loc('low')]=94
    bt=engine(df,sg); rep=bt.run(); t=bt.trades[0]
    assert t.reason=='SL' and t.entry_ts==t.exit_ts==df.index[1]
    assert rep['net_pnl']==pytest.approx(bt.equity-1000)


def test_entry_bar_close_does_not_influence_sizing_or_entry_gate():
    df,sg=market(); df2=df.copy()
    # Same open; these later closes must not change quantity at that open.
    df2.iloc[1,df2.columns.get_loc('close')]=119
    df2.iloc[1,df2.columns.get_loc('high')]=119
    a=engine(df,sg); b=engine(df2,sg); a.run(); b.run()
    assert a.trades[0].qty==b.trades[0].qty


def test_eod_cash_curve_and_net_profit_reconcile():
    df,sg=market(); bt=engine(df,sg); rep=bt.run()
    assert bt.trades[0].reason=='EOD'
    assert bt.equity==pytest.approx(999.8)
    assert rep['final_equity']==pytest.approx(bt.equity)
    assert bt.trades[0].pnl==pytest.approx(-.2)
    assert 'funding_warning' in rep


def test_zero_trail_activation_remains_zero():
    df,sg=market(); sg['trail_atr']=1.
    bt=engine(df,sg); bt.run()
    assert bt.trades[0].meta['trail_start_r']==0


def test_paper_and_backtest_same_exit_accounting_on_same_bar(account):
    from conftest import candles
    cfg,s,f,b=account; b.open(signal(),1,1000,0)
    f.minute=candles([('2024-01-01T00:01Z',100,101,89,100)])
    b.manage(480)
    df,sg=market(); df.iloc[1,df.columns.get_loc('low')]=89
    bt=engine(df,sg); bt.run()
    assert bt.trades[0].reason==s.closed()[0]['exit_reason']=='SL'
    assert bt.trades[0].pnl==pytest.approx(s.closed()[0]['pnl'])


def test_refit_uses_shared_engine_not_double_entry_fee(account):
    cfg,s,f,b=account; df,sg=market()
    rep=run_portfolio({'BTC':df},{'BTC':sg},cfg,480)
    assert rep['net_pnl']==pytest.approx(-.2)
    assert rep['final_equity']==pytest.approx(999.8)
    assert rep['funding_history_supplied'] is False


@pytest.mark.parametrize('bar_unit', ['s', 'ms', 'us', 'ns'])
@pytest.mark.parametrize('fund_unit', ['s', 'ms', 'us', 'ns'])
def test_funding_does_not_charge_before_entry(bar_unit, fund_unit):
    df,sg=market()
    df.index = df.index.as_unit(bar_unit)
    sg.index = sg.index.as_unit(bar_unit)
    fund=pd.DataFrame({'rate':[.01,.001]},index=[df.index[0],df.index[2]])
    fund.index = fund.index.as_unit(fund_unit)
    bt=engine(df,sg,funding={'BTC':fund}); bt.run()
    assert bt.trades[0].funding==pytest.approx(.2)
    assert bt.trades[0].pnl==pytest.approx(-.4)
    assert bt.equity==pytest.approx(999.6)


@pytest.mark.parametrize('cut',[260,380,500,650])
def test_donchian_signal_is_causal_on_synthetic_history(cut):
    rng=np.random.default_rng(4)
    close=100*np.exp(np.cumsum(rng.normal(0,.01,800)))
    ix=pd.date_range('2020-01-01',periods=800,freq='8h',tz='UTC')
    df=pd.DataFrame({'open':close,'high':close*1.02,'low':close*.98,
                     'close':close,'volume':100.},index=ix)
    full=strategies.donchian_trend_v2(df)
    part=strategies.donchian_trend_v2(df.iloc[:cut+1])
    pd.testing.assert_series_equal(full.iloc[cut],part.iloc[-1])
