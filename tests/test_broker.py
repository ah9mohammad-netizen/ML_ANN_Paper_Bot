import json
import sqlite3
import pandas as pd
import pytest
from conftest import signal, candles
from execution import bar_exit, trailing_stop, trade_pnl
from tg import Telegram


@pytest.mark.parametrize('side,old', [(1,(100,130,80,100)),(-1,(100,130,80,100))])
def test_pre_entry_and_straddling_candles_never_exit(account, side, old):
    cfg,s,f,b = account
    pid,err=b.open(signal(side),1,1000,0); assert not err
    f.minute=candles([('2023-12-31T23:59Z',*old),('2024-01-01T00:00Z',*old),
                      ('2024-01-01T00:01Z',100,101,99,100)])
    assert b.manage(480)==[]
    p=s.open_positions()[0]
    assert p['sl']==(95 if side>0 else 105)
    assert not s.get('execution_halt_reason')
    assert json.loads(p['extra'])['last_1m_seen'].startswith('2024-01-01 00:01')


def test_pre_entry_extremes_do_not_move_trail(account):
    cfg,s,f,b=account
    b.open(signal(trail_atr=1),1,1000,0)
    f.minute=candles([('2024-01-01T00:00Z',100,119,96,118),
                      ('2024-01-01T00:01Z',100,101,99,100)])
    b.manage(480)
    p=s.open_positions()[0]
    assert p['sl']==95
    assert json.loads(p['extra'])['peak']==101


@pytest.mark.parametrize('rows,reason', [
    ([('2024-01-01T00:01Z',100,101,94,100)], 'SL'),
    ([('2024-01-01T00:01Z',100,101,89,100)], 'SL'),
    ([('2024-01-01T00:01Z',89,100,85,90)], 'LIQ_GAP'),
    ([('2024-01-01T00:01Z',94,101,93,100)], 'SL_GAP'),
    ([('2024-01-01T00:01Z',100,121,94,100)], 'SL_AMBIG'),
    ([('2024-01-01T00:01Z',100,121,99,120)], 'TP'),
])
def test_post_entry_exit_order(account, rows, reason):
    cfg,s,f,b=account; b.open(signal(),1,1000,0); f.minute=candles(rows)
    result=b.manage(480)
    assert result[0][1]==reason
    p=s.closed()[0]
    assert p['pnl']==pytest.approx(s.equity()-1000)
    assert p['pnl']==pytest.approx(p['gross']-p['fees']-p['funding'])


@pytest.mark.parametrize('side', [1,-1])
def test_reject_stop_beyond_liquidation(account,side):
    cfg,s,f,b=account
    pid,err=b.open(signal(side,sl=80 if side>0 else 120),1,1000,0)
    assert pid is None and err=='stop_beyond_liquidation'
    assert not s.open_positions() and s.equity()==1000


def test_gap_in_exit_tape_flags_trade_and_blocks_entries(account):
    cfg,s,f,b=account; b.open(signal(),1,1000,0)
    f.minute=candles([('2024-01-01T00:04Z',100,101,99,100)])
    assert b.manage(480)==[]
    assert s.get('execution_halt_reason')
    assert json.loads(s.open_positions()[0]['extra'])['history_warning']


def test_no_duplicate_processing(account):
    cfg,s,f,b=account; b.open(signal(),1,1000,0)
    f.minute=candles([('2024-01-01T00:01Z',100,101,99,100)])
    b.manage(480)
    # Editing an already consumed bar cannot trigger a second historical exit.
    f.minute.iloc[0,f.minute.columns.get_loc('low')]=80
    assert b.manage(480)==[]
    assert len(s.open_positions())==1


def test_ignores_unclosed_future_candle(account):
    cfg,s,f,b=account; b.open(signal(),1,1000,0)
    f.minute=candles([('2100-01-01T00:01Z',100,101,1,2)])
    assert b.manage(480)==[]


@pytest.mark.parametrize('funding',[1.0,-1.0,0.0])
def test_cash_net_fees_and_funding_exactly_once(account,funding):
    cfg,s,f,b=account; pid,_=b.open(signal(),1,1000,0)
    assert s.equity()==pytest.approx(999.9)
    s.update_position(pid,funding=funding)
    p=s.open_positions()[0]
    result=b.close(p,110,'MANUAL')
    expected=20-.1-.11-funding
    assert result[2]==pytest.approx(expected)
    assert s.closed()[0]['pnl']==pytest.approx(expected)
    assert s.get('realized')==pytest.approx(expected)
    assert s.equity()==pytest.approx(1000+expected)
    assert b.close(p,110,'MANUAL') is None
    assert s.equity()==pytest.approx(1000+expected)


def test_open_cash_and_position_transaction_roll_back(account):
    cfg,s,f,b=account
    s.conn.execute("CREATE TRIGGER fail_cash BEFORE UPDATE ON state WHEN NEW.key='equity' BEGIN SELECT RAISE(ABORT, 'test'); END")
    # INSERT OR REPLACE isn't UPDATE: use an INSERT trigger too.
    s.conn.execute("CREATE TRIGGER fail_cash_insert BEFORE INSERT ON state WHEN NEW.key='equity' BEGIN SELECT RAISE(ABORT, 'test'); END")
    with pytest.raises(sqlite3.IntegrityError): b.open(signal(),1,1000,0)
    assert not s.open_positions() and s.equity()==1000


def test_close_transaction_rolls_back(account):
    cfg,s,f,b=account; b.open(signal(),1,1000,0); p=s.open_positions()[0]
    s.conn.execute("CREATE TRIGGER fail_close BEFORE UPDATE ON positions BEGIN SELECT RAISE(ABORT, 'test'); END")
    with pytest.raises(sqlite3.IntegrityError): b.close(p,110,'MANUAL')
    assert len(s.open_positions())==1 and not s.closed()
    assert s.equity()==pytest.approx(999.9)


def test_manual_flat_uses_same_net_accounting(account):
    cfg,s,f,b=account; b.open(signal(),1,1000,0); f.px=110
    tg=Telegram('','',s,b,None,cfg,b.log)
    tg.handle('/flat')
    assert s.closed()[0]['pnl']==pytest.approx(s.equity()-1000)


def test_accrued_funding_in_marked_equity(account):
    cfg,s,f,b=account; pid,_=b.open(signal(),1,1000,0)
    s.update_position(pid,funding=1.0)
    marked,unreal=b.mark_equity()
    assert marked==pytest.approx(998.9) and unreal==-1
    assert s.equity()==pytest.approx(999.9)  # settled once at close


def test_size_caps_actual_risk(account):
    cfg,s,f,b=account
    q,n,r=b.size(1000,100,98,0)
    assert n==350 and r==7
    assert b.size(1000,100,98,1990)==pytest.approx((.1,10,.2))


def test_chandelier_updates_at_strategy_close_using_current_atr(account):
    cfg,s,f,b=account; pid,_=b.open(signal(trail_atr=2),1,1000,0)
    p=s.open_positions()[0]; extra=json.loads(p['extra'])
    ix=pd.date_range('2023-12-20',periods=38,freq='8h',tz='UTC')
    f.strategy=pd.DataFrame(dict(open=100.,high=102.,low=98.,close=100.),index=ix)
    # Include exact Jan 1 00:00 bar, ATR=4 and later data with extreme values.
    ix=pd.date_range('2023-12-20','2024-01-01T08:00',freq='8h',tz='UTC')
    f.strategy=pd.DataFrame(dict(open=100.,high=102.,low=98.,close=100.),index=ix)
    f.strategy.loc[pd.Timestamp('2024-01-01T08:00Z'),'high']=10000
    bar=pd.Series(dict(open=110.,high=115.,low=110.,close=112.))
    sl=b._trail(p,extra,95,pd.Timestamp('2024-01-01T07:58Z'),bar,480)
    assert sl==95
    sl=b._trail(p,extra,sl,pd.Timestamp('2024-01-01T07:59Z'),bar,480)
    assert sl==pytest.approx(115-2*4)  # current ATR, not entry ATR=1


def test_break_even_cannot_loosen_a_trailing_stop():
    sl,_,done=trailing_stop(side=1,entry=100,sl0=95,sl=108,close=110,
        high=111,low=109,atr=1,peak=111,trail_atr=0,trail_start_r=0,
        be_at_r=1,fee_rate=.0005)
    assert sl==108 and done


@pytest.mark.parametrize('side,o,h,l,sl,tp,liq,want',[
    (1,100,101,89,95,120,90.5,'SL'),
    (-1,100,111,99,105,80,109.5,'SL'),
    (1,100,101,85,80,120,90.5,'LIQ'),
    (-1,100,115,99,120,80,109.5,'LIQ'),
    (1,89,101,80,95,120,90.5,'LIQ_GAP'),
    (-1,111,115,99,105,80,109.5,'LIQ_GAP'),
    (1,121,125,89,95,120,90.5,'TP_GAP'),
    (-1,79,111,75,105,80,109.5,'TP_GAP'),
])
def test_shared_liquidation_rules(side,o,h,l,sl,tp,liq,want):
    assert bar_exit(side,o,h,l,sl,tp,liq)[1]==want


def test_restart_retains_exit_cursor(account,tmp_path):
    from broker import PaperBroker
    from store import Store
    cfg,s,f,b=account; b.open(signal(),1,1000,0)
    f.minute=candles([('2024-01-01T00:01Z',100,101,99,100)])
    b.manage(480); s.backup(tmp_path/'restart.db')
    restarted=Store(tmp_path/'restart.db')
    try:
        b2=PaperBroker(cfg,restarted,f,b.log)
        f.minute.iloc[0,f.minute.columns.get_loc('low')]=80
        assert b2.manage(480)==[]
        assert len(restarted.open_positions())==1
    finally:
        restarted.conn.close()


def test_time_barrier_precedes_later_historical_stop(account):
    cfg,s,f,b=account; pid,_=b.open(signal(max_hold_bars=1),1,1000,0)
    # A one-minute hold for this synthetic broker call, deadline at 00:01:30.
    f.minute=candles([('2024-01-01T00:01Z',100,101,99,100),
                      ('2024-01-01T00:02Z',100,101,90,95)])
    assert b.manage(1)[0][1]=='TIME'
