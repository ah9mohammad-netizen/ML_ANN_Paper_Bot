"""Frozen-dataset builder — run where exchange APIs are reachable (Railway, laptop).

Downloads from OKX public endpoints (same venue the live paper bot reads):
  - spot candles per pair on its strategy timeframe (5m scalper / 15m SMC)
  - 1m candles optional (--with-1m) for finer exit-granularity studies
  - historical FUNDING RATES for the matching USDT perpetual swaps

Writes parquet files + a sha256 manifest so every reported number is
auditable against an immutable snapshot:

  research/data/frozen/<PAIR>-<TF>.parquet
  research/data/frozen/<PAIR>-SWAP-funding.parquet
  research/data/frozen/manifest.json

Usage:
  python -m research.fetch_data --days 730                 # full multi-regime pull
  python -m research.fetch_data --days 400 --pairs BTC,ETH # subset
"""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from strategy import CATEGORY_SMC, CATEGORY_SCALPER, timeframe_for_pair

OKX_HOSTS = ['https://aws.okx.com', 'https://www.okx.com', 'https://www.okx.cab', 'https://www.okx.ceo']
DATA_DIR = Path(__file__).parent / 'data' / 'frozen'
BAR = {'5m': '5m', '15m': '15m', '1m': '1m'}


def _get(path, params, tries=4):
    for _ in range(tries):
        for host in OKX_HOSTS:
            try:
                r = requests.get(host + path, params=params, timeout=15)
                js = r.json()
                if js.get('code') == '0':
                    return js.get('data') or []
            except Exception:
                continue
        time.sleep(1.0)
    return []


def fetch_candles(symbol, tf, days):
    """Paginate OKX history-candles backwards from now."""
    bar = BAR[tf]
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 86400_000
    rows, after = [], None
    while True:
        params = {'instId': f'{symbol}-USDT', 'bar': bar, 'limit': '100'}
        if after:
            params['after'] = after
        data = _get('/api/v5/market/history-candles', params)
        if not data:
            break
        rows.extend(data)
        after = data[-1][0]
        if int(data[-1][0]) <= start_ms:
            break
        time.sleep(0.12)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=['ts', 'open', 'high', 'low', 'close', 'vol', 'c1', 'c2', 'confirm'])
    df = df[df.confirm != '0']
    df['date'] = pd.to_datetime(df.ts.astype(np.int64), unit='ms', utc=True)
    for c in ('open', 'high', 'low', 'close', 'vol'):
        df[c] = pd.to_numeric(df[c], errors='coerce')
    df = df.rename(columns={'vol': 'volume'})[['date', 'open', 'high', 'low', 'close', 'volume']]
    return df.sort_values('date').drop_duplicates('date').reset_index(drop=True)


def fetch_funding(symbol, days):
    """Historical funding for the USDT-M perpetual swap."""
    end_ms = int(time.time() * 1000)
    start_ms = end_ms - days * 86400_000
    rows, after = [], None
    while True:
        params = {'instId': f'{symbol}-USDT-SWAP', 'limit': '100'}
        if after:
            params['after'] = after
        data = _get('/api/v5/public/funding-rate-history', params)
        if not data:
            break
        rows.extend(data)
        after = data[-1].get('fundingTime')
        if not after or int(after) <= start_ms:
            break
        time.sleep(0.12)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df['date'] = pd.to_datetime(df.fundingTime.astype(np.int64), unit='ms', utc=True)
    df['rate_pct'] = pd.to_numeric(df.fundingRate, errors='coerce') * 100
    return df[['date', 'rate_pct']].sort_values('date').drop_duplicates('date').reset_index(drop=True)


def _sha256(path):
    h = hashlib.sha256()
    h.update(Path(path).read_bytes())
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--days', type=int, default=730)
    ap.add_argument('--pairs', type=str, default='')
    ap.add_argument('--with-1m', action='store_true')
    ap.add_argument('--force', action='store_true',
                    help='re-download pairs that already have parquet files')
    args = ap.parse_args()

    pairs = [p.strip().upper() for p in args.pairs.split(',') if p.strip()] or \
        sorted(CATEGORY_SMC | CATEGORY_SCALPER)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    # resume support: rebuild manifest from existing files, so an interrupted
    # run (flaky shell session) just continues where it stopped
    manifest_path = DATA_DIR / 'manifest.json'
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else \
        {'days': args.days, 'built_at': '', 'files': {}}
    manifest['days'] = args.days

    for pair in pairs:
        tf = timeframe_for_pair(pair)
        if tf is None:
            print(f'skip {pair}: uncategorized')
            continue
        path = DATA_DIR / f'{pair}-{tf}.parquet'
        fpath = DATA_DIR / f'{pair}-SWAP-funding.parquet'
        if path.exists() and not args.force:
            print(f'⏭️  {pair} {tf} already frozen ({path.name}) — skipping (use --force to refresh)')
        else:
            print(f'⬇️  {pair} {tf} ({args.days}d)...', flush=True)
            df = fetch_candles(pair, tf, args.days)
            if df.empty:
                print(f'   ⚠️  no candle data for {pair}')
                continue
            df.to_parquet(path, index=False)
            manifest['files'][path.name] = {'sha256': _sha256(path), 'rows': len(df),
                                            'first': str(df.date.iloc[0]), 'last': str(df.date.iloc[-1])}
            print(f'   {len(df)} candles  {df.date.iloc[0]} → {df.date.iloc[-1]}')
            manifest_path.write_text(json.dumps(manifest, indent=2))  # checkpoint

        if fpath.exists() and not args.force:
            continue
        print(f'⬇️  {pair} funding...', flush=True)
        fdf = fetch_funding(pair, args.days)
        if not fdf.empty:
            fdf.to_parquet(fpath, index=False)
            manifest['files'][fpath.name] = {'sha256': _sha256(fpath), 'rows': len(fdf),
                                             'mean_rate_pct': round(float(fdf.rate_pct.mean()), 5)}
            print(f'   {len(fdf)} funding prints, mean {fdf.rate_pct.mean():.4f}%')
            manifest_path.write_text(json.dumps(manifest, indent=2))  # checkpoint

    manifest['built_at'] = pd.Timestamp.now(tz='UTC').isoformat()
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f'\n✅ frozen dataset: {len(manifest["files"])} files → {DATA_DIR}')


if __name__ == '__main__':
    main()
