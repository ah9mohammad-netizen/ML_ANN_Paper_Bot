"""
Phase 0 test suite — real assertions replacing the previous smoke test that
"validated" the engine by feeding it a synthetic 20-sigma spike and printing
SUCCESS regardless of behavior.

Run: python test_strategy.py

Covers the Phase 0 fixes:
  1. Pair routing: every configured pair has an explicit category + timeframe;
     uncategorized pairs are REJECTED (no silent inverted-R:R fallback).
  2. Scalper config integrity: every CCi_BB pair/side has a tuned config with
     sane (non-inverted) risk/reward.
  3. Engine behavior on synthetic forced setups (SMC sweep, CCI+BB break).
  4. Funding accrual math.
  5. Position sizing caps.
"""
import os
import tempfile
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from config import BotConfig
from strategy import (
    StrategyBrain, SCALPER_ASSET_CONFIG, CATEGORY_SMC, CATEGORY_SCALPER,
    timeframe_for_pair,
)
from paper_engine import PaperEngine, count_funding_events
from storage import Store

PASS = 0
FAIL = 0

def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        PASS += 1
        print(f"   ✅ {name}")
    else:
        FAIL += 1
        print(f"   ❌ {name} {detail}")

def flat_df(n=300, tf='15min', price=100.0, volume=100.0):
    """Perfectly flat synthetic candles: high/low straddle the close by 1."""
    ts = pd.date_range(end=pd.Timestamp.now(tz='UTC'), periods=n, freq=tf)
    close = np.full(n, price, dtype=float)
    df = pd.DataFrame({
        'open': close.copy(),
        'high': close + 1.0,
        'low': close - 1.0,
        'close': close.copy(),
        'volume': np.full(n, volume, dtype=float),
    }, index=ts)
    return df

# ------------------------------------------------------------------
# 1. Routing integrity (the SUI/APT/DOGE inverted-R:R bug)
# ------------------------------------------------------------------
def test_routing():
    print("👉 Test 1: pair routing & categories")
    cfg_pairs = [p.strip().upper() for p in BotConfig().pairs if p.strip()]
    for p in cfg_pairs:
        categorized = p in CATEGORY_SMC or p in CATEGORY_SCALPER
        check(f"{p} has an explicit category", categorized)
        check(f"{p} maps to a candle timeframe", timeframe_for_pair(p) in ('15m', '5m'))
    check("SUI routes to SMC (15m)", 'SUI' in CATEGORY_SMC and timeframe_for_pair('SUI') == '15m')
    check("APT routes to SMC (15m)", 'APT' in CATEGORY_SMC and timeframe_for_pair('APT') == '15m')
    check("DOGE routes to scalper (5m)", 'DOGE' in CATEGORY_SCALPER and timeframe_for_pair('DOGE') == '5m')

    brain = StrategyBrain()
    sig = brain.latest_signal('XYZ123', flat_df(300))
    check("uncategorized pair is REJECTED (None), not silently traded", sig is None)
    sig = brain.calculate_cci_bb_signal('XYZ123', flat_df(200))
    check("scalper refuses unknown pair (no fallback config)", sig is None)

# ------------------------------------------------------------------
# 2. Scalper config sanity — no inverted risk/reward anywhere
# ------------------------------------------------------------------
def test_scalper_configs():
    print("👉 Test 2: scalper config risk/reward sanity")
    for (pair, side), c in SCALPER_ASSET_CONFIG.items():
        rr = c['tp_atr'] / c['sl_atr']
        check(f"{pair} {side} reward >= risk (RR={rr:.2f})", rr >= 1.0)
    for p in CATEGORY_SCALPER:
        check(f"{p} has LONG+SHORT scalper configs",
              (p, 'LONG') in SCALPER_ASSET_CONFIG and (p, 'SHORT') in SCALPER_ASSET_CONFIG)

# ------------------------------------------------------------------
# 3. Engine behavior on forced setups
# ------------------------------------------------------------------
def test_smc_signal():
    print("👉 Test 3: SMC sweep-reclaim fires on a forced sweep")
    df = flat_df(300)  # flat, low=99 close=100 everywhere
    # Forced sweep candle: wicks below the 36-bar swing low, closes back above,
    # with 2x volume capitulation.
    df.iloc[-1, df.columns.get_loc('low')] = 98.0
    df.iloc[-1, df.columns.get_loc('close')] = 100.5
    df.iloc[-1, df.columns.get_loc('open')] = 100.0
    df.iloc[-1, df.columns.get_loc('volume')] = 200.0

    brain = StrategyBrain()
    sig = brain.calculate_smc_signal('SOL', df, lookback=36, fvg_required=False, trend_filter=True)
    check("LONG signal generated", sig is not None and sig['side'] == 'LONG')
    if sig:
        risk = sig['entry'] - sig['sl']
        reward = sig['tp'] - sig['entry']
        check("1.33:1 R:R geometry", abs(reward / risk - (2.0 / 1.5)) < 1e-6,
              f"(got {reward/risk:.3f})")
        check("SL below entry, TP above entry", sig['sl'] < sig['entry'] < sig['tp'])

def test_cci_bb_signal():
    print("👉 Test 4: CCI+BB scalper fires on a forced extreme dump")
    df = flat_df(200, tf='5min')  # flat at 100
    # Last candle: violent dump closes at 90 (close < lower BB, CCI extremely negative)
    df.iloc[-1, df.columns.get_loc('open')] = 95.0
    df.iloc[-1, df.columns.get_loc('high')] = 95.0
    df.iloc[-1, df.columns.get_loc('low')] = 89.5
    df.iloc[-1, df.columns.get_loc('close')] = 90.0

    brain = StrategyBrain()
    # Use LINK: the one pair tuned WITHOUT an EMA trend filter, so the deep
    # dump below EMA-100 is still tradable (that filter blocking dip-buys is
    # intentional behavior, verified here separately with DOGE below).
    sig = brain.calculate_cci_bb_signal('LINK', df)
    check("LONG signal generated for LINK (trend_filter=False)", sig is not None and sig['side'] == 'LONG')
    if sig:
        risk = sig['entry'] - sig['sl']
        reward = sig['tp'] - sig['entry']
        check("1.33:1 R:R geometry", abs(reward / risk - (2.0 / 1.5)) < 1e-6)
    blocked = brain.calculate_cci_bb_signal('DOGE', df)
    check("DOGE LONG correctly blocked by EMA trend filter after a dump", blocked is None)

# ------------------------------------------------------------------
# 5. Funding accrual math
# ------------------------------------------------------------------
def test_funding_math():
    print("👉 Test 5: funding accrual")
    t0 = datetime(2026, 8, 22, 1, 0, tzinfo=timezone.utc)  # 01:00 UTC
    check("6h hold (no boundary crossed) -> 0 funding events",
          count_funding_events(t0, t0 + timedelta(hours=6)) == 0)
    check("9h hold crossing 08:00 UTC -> 1 event",
          count_funding_events(t0, t0 + timedelta(hours=9)) == 1)
    check("25h hold -> 3 events", count_funding_events(t0, t0 + timedelta(hours=25)) == 3)
    check("zero/negative duration -> 0", count_funding_events(t0, t0) == 0)

# ------------------------------------------------------------------
# 6. Position sizing caps
# ------------------------------------------------------------------
def test_sizing():
    print("👉 Test 6: position sizing caps")
    with tempfile.TemporaryDirectory() as tmp:
        cfg = BotConfig()
        cfg.starting_balance = 1000.0
        store = Store(os.path.join(tmp, 't.db'))
        engine = PaperEngine(cfg, store, StrategyBrain(cfg))
        sig = {
            'pair': 'BTC', 'entry': 100000.0,
            'risk_pct': 0.01,    # 1% stop
            'risk_mult': 1.0,
        }
        notional, margin, qty = engine.calc_size(sig)
        # risk 1% of 1000 = 10 USDT / 1% stop = 1000 notional; margin 100
        # (under the 15% margin cap of 150, so uncapped fixed-fractional).
        check("fixed-fractional sizing", abs(notional - 1000.0) < 1e-6, f"(got {notional})")
        check("margin = notional/leverage", abs(margin - 100.0) < 1e-6)
        # Tighter 0.5% stop -> raw notional 2000, but margin cap 150 binds
        # first -> notional de-levered to 1500.
        sig1 = {'pair': 'BTC', 'entry': 100000.0, 'risk_pct': 0.005, 'risk_mult': 1.0}
        notional1, margin1, _ = engine.calc_size(sig1)
        check("margin cap de-levers oversized positions",
              abs(margin1 - 150.0) < 1e-6 and abs(notional1 - 1500.0) < 1e-6,
              f"(got margin={margin1}, notional={notional1})")
        # Extremely tight stop would explode notional -> capped
        sig2 = {'pair': 'BTC', 'entry': 100000.0, 'risk_pct': 0.0001, 'risk_mult': 1.0}
        notional2, margin2, _ = engine.calc_size(sig2)
        check("notional cap respected (<= margin cap * leverage)",
              notional2 <= (1000 * cfg.max_margin_per_position_pct / 100) * cfg.leverage + 1e-6)

# ------------------------------------------------------------------
# 7. Intrabar exit engine (Phase 0's core honesty fix): SL/TP must trigger on
#    1m candle high/low, with pessimistic tie-break when one candle spans both.
# ------------------------------------------------------------------
def _run_exit_scenario(tmp, last_candle_high, last_candle_low, expect_reason):
    import paper_engine as pe

    cfg = BotConfig()
    cfg.starting_balance = 1000.0
    store = Store(os.path.join(tmp, 't.db'))
    engine = PaperEngine(cfg, store, StrategyBrain(cfg))

    pid = store.add_position({
        'pair': 'BTC', 'side': 'LONG', 'setup': 'TEST',
        'probability': 0.5, 'entry': 100.0, 'sl': 97.0, 'tp': 106.0,
        'notional': 1000.0, 'margin': 100.0, 'leverage': 10.0, 'qty': 10.0,
        'meta': {'max_hold_hours': 24},
    })
    # Backdate the open by 3 minutes so the mocked 1m candles post-date entry.
    opened = pd.Timestamp.now(tz='UTC') - pd.Timedelta(minutes=3)
    store.conn.execute('UPDATE positions SET opened_at=? WHERE id=?', (opened.isoformat(), pid))
    store.conn.commit()

    ts = pd.date_range(end=pd.Timestamp.now(tz='UTC'), periods=3, freq='1min')
    fake_1m = pd.DataFrame({
        'open': [100.0, 100.0, 100.0],
        'high': [100.5, 100.5, last_candle_high],
        'low':  [99.5, 99.5, last_candle_low],
        'close': [100.0, 100.0, 100.0],
        'volume': [10.0, 10.0, 10.0],
    }, index=ts)

    orig_okx, orig_tick = pe.fetch_okx, pe.fetch_ticker
    pe.fetch_okx = lambda pair, tf, limit=300: fake_1m if tf == '1m' else pd.DataFrame()
    pe.fetch_ticker = lambda pair: 100.0  # neutral mark between SL and TP
    try:
        closed = engine.update_positions()
    finally:
        pe.fetch_okx, pe.fetch_ticker = orig_okx, orig_tick

    if expect_reason is None:
        return (len(closed) == 0, None, store.balance())
    if len(closed) != 1:
        return (False, None, store.balance())
    pos, reason, pnl = closed[0]
    return (reason == expect_reason, pnl, store.balance())

def test_intrabar_exits():
    print("👉 Test 7: intrabar exit engine (1m high/low, pessimistic tie-break)")
    with tempfile.TemporaryDirectory() as tmp:
        # (a) 1m candle low pierces SL at 97 -> closes at SL, LONG loses ~3% + costs
        ok, pnl, bal = _run_exit_scenario(tmp, 100.8, 96.5, 'SL')
        check("SL hit on 1m low -> exits at stop", ok)
        check("SL economics: notional 1000, -3% gross, -0.12% costs, no funding",
              pnl is not None and abs(pnl - (-31.2)) < 0.01, f"(pnl={pnl})")
    with tempfile.TemporaryDirectory() as tmp:
        # (b) 1m candle high touches TP at 106 -> closes at TP
        ok, pnl, bal = _run_exit_scenario(tmp, 107.0, 99.0, 'TP')
        check("TP hit on 1m high -> exits at target", ok)
        check("TP economics: +6% gross - 0.12% costs",
              pnl is not None and abs(pnl - 58.8) < 0.01, f"(pnl={pnl})")
    with tempfile.TemporaryDirectory() as tmp:
        # (c) ONE candle spans both SL and TP -> pessimistic: stop must win
        ok, pnl, bal = _run_exit_scenario(tmp, 107.0, 96.5, 'SL')
        check("both levels in one candle -> SL assumed first (pessimistic)", ok)
    with tempfile.TemporaryDirectory() as tmp:
        # (d) neither level touched -> position survives
        ok, _, _ = _run_exit_scenario(tmp, 103.0, 98.0, None)
        check("no level touched -> position stays open", ok)

if __name__ == '__main__':
    print("=== PHASE 0 VERIFICATION SUITE ===")
    test_routing()
    test_scalper_configs()
    test_smc_signal()
    test_cci_bb_signal()
    test_funding_math()
    test_sizing()
    test_intrabar_exits()
    print(f"\n{'🎉 ALL TESTS PASSED' if FAIL == 0 else '🔴 FAILURES PRESENT'}: {PASS} passed, {FAIL} failed")
    raise SystemExit(1 if FAIL else 0)
