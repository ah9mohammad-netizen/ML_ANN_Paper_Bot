import time
import requests
import pandas as pd
import numpy as np

BAR_MAP={
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

def fetch_okx(sym, tf, limit=300):
    pair=f'{sym}-USDT'
    bar=BAR_MAP[tf]
    all_c=[]; after=None
    pages=max(1, int(np.ceil(limit/100)))
    for _ in range(pages):
        params={'instId':pair,'bar':bar,'limit':'100'}
        if after: params['after']=after
        r=requests.get('https://www.okx.com/api/v5/market/history-candles',params=params,timeout=12)
        js=r.json()
        if js.get('code')!='0' or not js.get('data'): break
        all_c.extend(js['data']); after=js['data'][-1][0]
        time.sleep(0.04)
    if not all_c: return pd.DataFrame()
    df=pd.DataFrame(all_c,columns=['ts','open','high','low','close','vol','volCcy','volCcyQuote','confirm'])
    df=df[df.confirm!='0'][['ts','open','high','low','close','vol']].copy()
    df.columns=['ts','open','high','low','close','volume']
    df['ts']=pd.to_datetime(df.ts.astype(np.int64),unit='ms',utc=True)
    for c in ['open','high','low','close','volume']:
        df[c]=pd.to_numeric(df[c],errors='coerce')
    return df.sort_values('ts').drop_duplicates('ts').set_index('ts')[['open','high','low','close','volume']]
