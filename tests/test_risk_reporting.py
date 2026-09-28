import json
from types import SimpleNamespace
import pandas as pd
import pytest
from conftest import signal
from risk import RiskManager
from runner import Runner
from store import Store
from tg import Telegram
import metrics


def runner_stub(cfg,s,b,stale=None):
    r=Runner.__new__(Runner); r.cfg=cfg; r.store=s; r.log=b.log
    r.risk=RiskManager(cfg,s,b.log); r.broker=b
    stale=stale or []
    r.feed=SimpleNamespace(bars_many=lambda *a,**k: None,
        stale_symbols=lambda *a: stale,recover=lambda *a: None,
        mark_refreshed=lambda *a: None)
    return r


@pytest.mark.parametrize('halt',['manual halt','drawdown 30% — halted',
                                 'daily loss -6%','8 consecutive losses'])
def test_refresh_does_not_clear_risk_halts(account,halt):
    cfg,s,f,b=account; s.set('halt_reason',halt)
    r=runner_stub(cfg,s,b); scans=[]; r.scan=lambda *a: scans.append(a)
    r.refresh_and_scan(1000)
    assert s.get('halt_reason')==halt and not scans
    assert r.risk.can_open(signal(),[],1000)==(False,'risk_halt')


def test_data_staleness_recovers_without_manual_resume(account):
    cfg,s,f,b=account; r=runner_stub(cfg,s,b,stale=cfg.pairs)
    scans=[]; r.scan=lambda *a: scans.append(a)
    r.refresh_and_scan(1000)
    assert s.get('data_halt_reason') and not scans
    assert not s.get('halt_reason')
    r.feed.stale_symbols=lambda *a: []
    r.refresh_and_scan(1000)
    assert not s.get('data_halt_reason') and len(scans)==1


def test_new_drawdown_halt_prevents_scan(account):
    cfg,s,f,b=account; r=runner_stub(cfg,s,b); scans=[]
    r.scan=lambda *a: scans.append(a); r.refresh_and_scan(700)
    assert not scans and s.get('halt_reason').startswith('drawdown')


def test_mark_to_market_peak_drives_drawdown(account):
    cfg,s,f,b=account; risk=RiskManager(cfg,s,b.log)
    risk.halt_check(1200)
    s.set('day_start_equity',1000)
    assert s.get('peak_equity')==1200
    assert risk.halt_check(899).startswith('drawdown')


def test_daily_halt_rolls_but_manual_halt_persists(account):
    cfg,s,f,b=account; r=RiskManager(cfg,s,b.log)
    s.set('halt_reason','daily loss -6%'); s.set('day','1900-01-01')
    r.roll_day(1000); assert not s.get('halt_reason')
    s.set('halt_reason','manual halt'); s.set('day','1900-01-01')
    r.roll_day(1000); assert s.get('halt_reason')=='manual halt'


@pytest.mark.parametrize('key,why',[('execution_halt_reason','execution_history_halt'),
                                    ('data_halt_reason','stale_data')])
def test_entry_gate_rejects_quality_halts(account,key,why):
    cfg,s,f,b=account; s.set(key,'missing history')
    r=RiskManager(cfg,s,b.log)
    assert r.can_open(signal(),[],1000)==(False,why)


def test_resume_does_not_override_breached_limits(account):
    cfg,s,f,b=account; s.set('equity',700); s.set('halt_reason','manual halt')
    risk=RiskManager(cfg,s,b.log); tg=Telegram('','',s,b,risk,cfg,b.log)
    messages=[]; tg.send=lambda msg,**kw: messages.append(msg)
    tg.handle('/resume')
    assert s.get('halt_reason').startswith('drawdown')
    assert 'still blocked' in messages[0]


def test_legacy_normalization_preserves_raw_db(account):
    cfg,s,f,b=account; pid,_=b.open(signal(),1,1000,0)
    s.update_position(pid,status='CLOSED',gross=20.,fees=.21,funding=1.,
                      pnl=18.89,r_multiple=1.889,extra='{}')
    p=s.closed()[0]
    assert p['pnl']==pytest.approx(18.79) and p['legacy_execution']
    raw=s.conn.execute('SELECT pnl,extra FROM positions WHERE id=?',(pid,)).fetchone()
    assert raw=={'pnl':18.89,'extra':'{}'}
    m=metrics.from_store(s)
    assert metrics.verdict(m)[0]=='UNVALIDATED EXECUTION DATA'
    assert 'funding cost +1.00 (+paid/-received)' in metrics.fmt(m)


def test_backup_includes_wal(account,tmp_path):
    cfg,s,f,b=account
    on_disk=Store(tmp_path/'live.db'); on_disk.init_account(1234)
    on_disk.backup(tmp_path/'backup.db')
    copy=Store(tmp_path/'backup.db')
    assert copy.equity()==1234
    copy.conn.close(); on_disk.conn.close()


def test_report_no_hardcoded_frequency_or_edge_probability(account):
    cfg,s,f,b=account
    for i in range(12):
        pid,_=b.open(signal(),i,1000,0)
        b.close(s.open_positions()[0],110 if i%2 else 95,'TEST')
    m=metrics.from_store(s); text=metrics.fmt(m)
    assert m['legacy_trades']==0
    assert '1.3 trades/week' not in text and '16 more months' not in text
    assert 'positive bootstrap means' in text
    assert 'CI method: IID' in text
    assert metrics.verdict(m)[0]=='INSUFFICIENT DATA'


def test_all_winner_bootstrap_preserves_infinity(account):
    cfg,s,f,b=account
    for i in range(10):
        b.open(signal(),i,1000,0); b.close(s.open_positions()[0],110,'TEST')
    m=metrics.from_store(s)
    assert m['pf_ci95']==(float('inf'),float('inf'))
    assert m['bootstrap_positive_fraction']==1


def test_post_fix_report_excludes_legacy_without_deleting(account):
    cfg,s,f,b=account
    pid,_=b.open(signal(),1,1000,0)
    b.close(s.open_positions()[0],110,'TEST')
    s.update_position(pid,extra='{}')
    b.open(signal(),2,1000,0); b.close(s.open_positions()[0],110,'TEST')
    m=metrics.from_store(s,post_fix_only=True)
    assert m['n']==1 and m['excluded_trades']==1 and m['legacy_trades']==0
    assert len(s.closed())==2
    assert 'Post-fix trades only' in metrics.fmt(m)


def test_market_gate_missing_data_fails_closed(account):
    cfg,s,f,b=account
    r=runner_stub(cfg,s,b); r._mkt_daily=None; r._mkt_ts=0
    r.feed.bars=lambda *a,**kw: None
    ix=pd.date_range('2024-01-01',periods=3,tz='UTC')
    assert r.market_filter(ix).eq(0).all()


def test_halt_does_not_prevent_exit_management(account):
    from conftest import candles
    cfg,s,f,b=account; b.open(signal(),1,1000,0)
    s.set('halt_reason','manual halt')
    f.minute=candles([('2024-01-01T00:01Z',100,101,94,99)])
    assert b.manage(480)[0][1]=='SL'
    assert not s.open_positions()
