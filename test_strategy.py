import sys
import numpy as np
import pandas as pd
from config import BotConfig
from strategy import StrategyBrain
from data_client import fetch_okx

def generate_mock_data(tf, limit=300):
    print(f"⚠️ Network error or sandbox restriction. Generating {limit} realistic mock candles...")
    np.random.seed(42)
    # Generate timestamp index
    freq_map = {'1m': '1min', '5m': '5min', '15m': '15min', '30m': '30min', '1h': '1h', '4h': '4h'}
    freq = freq_map.get(tf, '5min')
    ts = pd.date_range(end=pd.Timestamp.now(tz='UTC'), periods=limit, freq=freq)
    
    # Generate synthetic price path (random walk)
    close = 100.0 + np.cumsum(np.random.normal(0.0, 0.5, limit))
    open_px = close - np.random.normal(0.0, 0.2, limit)
    high = np.maximum(open_px, close) + np.random.exponential(0.15, limit)
    low = np.minimum(open_px, close) - np.random.exponential(0.15, limit)
    volume = np.random.exponential(1000, limit)
    
    # For CCI_BB_SCALPER, let's make the last bar hit an extreme oversold so we trigger a signal
    # CCI <= -134, close < lower bollinger band
    # Let's force a dump on the last few candles
    if limit > 20:
        close[-1] = close[-20:].min() - 10.0
        open_px[-1] = close[-1] + 1.0
        low[-1] = close[-1] - 0.5
        high[-1] = open_px[-1] + 0.1
    
    df = pd.DataFrame({
        'open': open_px,
        'high': high,
        'low': low,
        'close': close,
        'volume': volume
    }, index=ts)
    return df

def test_mode(mode, tf, pair='BTC'):
    print(f"\n--- Testing Mode: {mode} (Timeframe: {tf}) for {pair} ---")
    cfg = BotConfig()
    cfg.strategy_mode = mode
    cfg.timeframe = tf
    
    brain = StrategyBrain(cfg)
    
    print(f"Fetching OKX {tf} candles for {pair}...")
    try:
        df = fetch_okx(pair, tf, limit=300)
        if df.empty:
            df = generate_mock_data(tf)
    except Exception as e:
        print(f"Network error: {e}")
        df = generate_mock_data(tf)
        
    print(f"Using {len(df)} candles. Columns: {list(df.columns)}")
    print(f"Latest candles price range: Low={df.low.iloc[-1]:.2f}, High={df.high.iloc[-1]:.2f}, Close={df.close.iloc[-1]:.2f}")
    
    print("Running Strategy Brain latest_signal...")
    sig = brain.latest_signal(pair, None, None, df)
    
    if sig:
        print(f"🚀 SIGNAL GENERATED!")
        print(f"  Side: {sig['side']}")
        print(f"  Setup: {sig['setup']}")
        print(f"  Probability: {sig['probability']:.2f}")
        print(f"  Predicted R: {sig['predicted_R']:.2f}")
        print(f"  Entry: {sig['entry']:.4f}")
        print(f"  SL: {sig['sl']:.4f} | TP: {sig['tp']:.4f}")
        print(f"  Risk %: {sig['risk_pct']*100:.2f}%")
        print(f"  Hold time: {sig['max_hold_hours']} hours")
    else:
        print("ℹ️ No active signal at the latest candle close (neutral market state).")

if __name__ == '__main__':
    print("Starting strategy dry-run tests...")
    
    # 1. Test CCI_BB_SCALPER on 5m
    test_mode('CCI_BB_SCALPER', '5m', 'SOL')
    
    # 2. Test HMA_TREND_CROSSOVER on 5m
    test_mode('HMA_TREND_CROSSOVER', '5m', 'SOL')
    
    # 3. Test SMART_MONEY_PULLBACK on 30m
    test_mode('SMART_MONEY_PULLBACK', '30m', 'SOL')
    
    # 4. Test MTF_LOCAL_OPT on 4h
    test_mode('MTF_LOCAL_OPT', '4h', 'BTC')
