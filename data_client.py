import time
import requests
import pandas as pd
import numpy as np

BAR_MAP = {
    '1m': '1m',
    '3m': '3m',
    '5m': '5m',
    '15m': '15m',
    '30m': '30m',
    '1h': '1H',
    '2h': '2H',
    '4h': '4H',
    '1D': '1D'
}

def fetch_binance(sym, tf, limit=300):
    binance_tf = tf  # '5m' and '15m' are identical on Binance
    symbol = f"{sym.upper()}USDT"
    
    # Redundant public endpoints for Binance API
    gateways = [
        'https://api.binance.com',
        'https://api1.binance.com',
        'https://api3.binance.com'
    ]
    
    for base_url in gateways:
        try:
            url = f"{base_url}/api/v3/klines"
            params = {
                'symbol': symbol,
                'interval': binance_tf,
                'limit': min(limit, 1000)
            }
            r = requests.get(url, params=params, timeout=8)
            js = r.json()
            if isinstance(js, list) and len(js) > 0:
                # 0: Open time, 1: Open, 2: High, 3: Low, 4: Close, 5: Volume
                df = pd.DataFrame(js, columns=[
                    'ts', 'open', 'high', 'low', 'close', 'volume', 
                    'close_time', 'asset_vol', 'trades', 'buy_base', 'buy_asset', 'ignored'
                ])
                df = df[['ts', 'open', 'high', 'low', 'close', 'volume']].copy()
                df['ts'] = pd.to_datetime(df.ts.astype(np.int64), unit='ms', utc=True)
                for c in ['open', 'high', 'low', 'close', 'volume']:
                    df[c] = pd.to_numeric(df[c], errors='coerce')
                return df.sort_values('ts').drop_duplicates('ts').set_index('ts')
        except Exception:
            continue
            
    return pd.DataFrame()

def fetch_okx(sym, tf, limit=300):
    pair = f'{sym}-USDT'
    bar = BAR_MAP[tf]
    all_c = []; after = None
    pages = max(1, int(np.ceil(limit/100)))
    
    # Redundant public endpoints for OKX API (bypasses local cloud server blocks)
    endpoints = [
        'https://aws.okx.com',
        'https://www.okx.com',
        'https://www.okx.cab',
        'https://www.okx.ceo'
    ]
    
    for _ in range(pages):
        params = {'instId': pair, 'bar': bar, 'limit': '100'}
        if after: 
            params['after'] = after
            
        success = False
        for base_url in endpoints:
            try:
                r = requests.get(f'{base_url}/api/v5/market/history-candles', params=params, timeout=8)
                js = r.json()
                if js.get('code') == '0' and js.get('data'):
                    all_c.extend(js['data'])
                    after = js['data'][-1][0]
                    success = True
                    break
            except Exception:
                continue
                
        if not success:
            break
            
        time.sleep(0.05)
        
    if not all_c: 
        # Trigger the Binance Failover backup!
        print(f"🔄 [Failover Triggered] OKX failed for {sym}. Switching to Binance Public API...")
        df_backup = fetch_binance(sym, tf, limit)
        if not df_backup.empty:
            print(f"✅ [Failover Success] Successfully fetched {sym} candles from Binance.")
            return df_backup
        return pd.DataFrame()
        
    df = pd.DataFrame(all_c, columns=['ts','open','high','low','close','vol','volCcy','volCcyQuote','confirm'])
    df = df[df.confirm != '0'][['ts','open','high','low','close','vol']].copy()
    df.columns = ['ts','open','high','low','close','volume']
    df['ts'] = pd.to_datetime(df.ts.astype(np.int64), unit='ms', utc=True)
    for c in ['open','high','low','close','volume']:
        df[c] = pd.to_numeric(df[c], errors='coerce')
    return df.sort_values('ts').drop_duplicates('ts').set_index('ts')[['open','high','low','close','volume']]
