"""OKX historical candle downloader with local parquet cache.

Uses /api/v5/market/history-candles (100 bars/page, paginates backwards via `after`).
Only `confirm == '1'` (closed) bars are kept.
"""
import os, time, threading, datetime as dt
import requests
import numpy as np
import pandas as pd

BASE = os.getenv("OKX_BASE", "https://www.okx.com")
CACHE = os.path.dirname(os.path.abspath(__file__)) + "/data"
os.makedirs(CACHE, exist_ok=True)

BAR = {"1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m",
       "1h": "1H", "2h": "2H", "4h": "4H", "1d": "1D"}

_MS = {"1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
       "1h": 3600, "2h": 7200, "4h": 14400, "1d": 86400}

# crude global rate limiter: OKX history-candles = 20 req / 2s per IP
_lock = threading.Lock()
_slots = []


_MAXP2 = int(os.getenv("OKX_MAX_PER_2S", "18"))


def _throttle(max_per_2s=None):
    m = max_per_2s or _MAXP2
    while True:
        with _lock:
            now = time.time()
            while _slots and now - _slots[0] > 2.0:
                _slots.pop(0)
            if len(_slots) < m:
                _slots.append(now)
                return
            wait = 2.0 - (now - _slots[0])
        time.sleep(max(0.005, min(wait, 0.2)))


def _page(inst_id, bar, after=None, retries=5):
    p = {"instId": inst_id, "bar": bar, "limit": "100"}
    if after is not None:
        p["after"] = str(after)
    for i in range(retries):
        _throttle()
        try:
            r = requests.get(f"{BASE}/api/v5/market/history-candles",
                             params=p, timeout=15)
            j = r.json()
            if j.get("code") == "0":
                return j.get("data") or []
            time.sleep(0.4 * (i + 1))
        except Exception:
            time.sleep(0.5 * (i + 1))
    return []


def _to_df(rows):
    """Raw OKX page rows -> compact float frame. Called often so RAM stays flat."""
    df = pd.DataFrame(rows, columns=["ts", "open", "high", "low", "close",
                                     "vol", "volCcy", "volCcyQuote", "confirm"])
    df = df[df.confirm == "1"]
    if df.empty:
        return df
    df = df[["ts", "open", "high", "low", "close", "vol", "volCcyQuote"]].copy()
    df.columns = ["ts", "open", "high", "low", "close", "volume", "quote_volume"]
    df["ts"] = pd.to_datetime(df.ts.astype(np.int64), unit="ms", utc=True)
    for c in ["open", "high", "low", "close", "volume", "quote_volume"]:
        df[c] = pd.to_numeric(df[c], errors="coerce", downcast="float")
    return df.set_index("ts").dropna()


def _merge_save(path, frames, existing=None):
    parts = [f for f in frames if f is not None and len(f)]
    if existing is not None and len(existing):
        parts.append(existing)
    if not parts:
        return None
    df = pd.concat(parts)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    tmp = path + ".tmp"
    df.to_parquet(tmp)
    os.replace(tmp, path)
    return df


def download(symbol, tf, start, inst="SWAP", verbose=False, path=None,
             chunk_pages=150, existing=None):
    """Page backwards to `start`, checkpointing to `path` every `chunk_pages`.

    Memory stays flat: raw JSON rows are converted to a float frame and dropped
    every chunk instead of being buffered for the whole run. Safe to kill and
    re-run — it resumes from the oldest bar already on disk.
    """
    inst_id = f"{symbol}-USDT-SWAP" if inst == "SWAP" else f"{symbol}-USDT"
    bar = BAR[tf]
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    after = None
    if existing is not None and len(existing):
        after = int(existing.index[0].timestamp() * 1000)   # resume at oldest
    rows, frames, guard, last, t0 = [], [], 0, None, time.time()
    stop = False
    while guard < 20000 and not stop:
        guard += 1
        d = _page(inst_id, bar, after)
        if not d:
            stop = True
        else:
            rows.extend(d)
            after = d[-1][0]
            if last is not None and int(after) >= last:
                stop = True
            last = int(after)
            if int(after) <= start_ms:
                stop = True
        if rows and (len(rows) >= chunk_pages * 100 or stop):
            frames.append(_to_df(rows)); rows = []
            if path:
                existing = _merge_save(path, frames, existing)
                frames = []
            if verbose:
                print(f"    {symbol} {tf}: {guard}p "
                      f"back to {pd.Timestamp(last, unit='ms') if last else '?'} "
                      f"({len(existing) if existing is not None else 0} bars, "
                      f"{time.time()-t0:.0f}s)", flush=True)
    if rows:
        frames.append(_to_df(rows))
    out = _merge_save(path, frames, existing) if path else (
        pd.concat(frames).sort_index() if frames else pd.DataFrame())
    if out is None or not len(out):
        return pd.DataFrame()
    return out.loc[pd.Timestamp(start, tz="UTC"):]


def get(symbol, tf, start="2022-01-01", inst="SWAP", refresh=False, verbose=True):
    f = f"{CACHE}/{inst}_{symbol}_{tf}.parquet"
    existing = None
    if os.path.exists(f) and not refresh:
        try:
            existing = pd.read_parquet(f)
        except Exception:
            existing = None
        if existing is not None and len(existing):
            want = pd.Timestamp(start, tz="UTC")
            fresh = (pd.Timestamp.utcnow().tz_localize(None) -
                     existing.index[-1].tz_localize(None)) < pd.Timedelta(days=1)
            if existing.index[0] <= want + pd.Timedelta(days=3) and fresh:
                return existing.loc[want:]
    return download(symbol, tf, start, inst, verbose=verbose, path=f,
                    existing=existing)


def funding(symbol, start="2022-01-01"):
    """Historical funding rates for a perp (8h settlements)."""
    f = f"{CACHE}/funding_{symbol}.parquet"
    if os.path.exists(f):
        return pd.read_parquet(f)
    inst_id = f"{symbol}-USDT-SWAP"
    start_ms = int(pd.Timestamp(start, tz="UTC").timestamp() * 1000)
    rows, after, guard = [], None, 0
    while guard < 4000:
        guard += 1
        p = {"instId": inst_id, "limit": "100"}
        if after:
            p["after"] = str(after)
        _throttle()
        try:
            j = requests.get(f"{BASE}/api/v5/public/funding-rate-history",
                             params=p, timeout=15).json()
        except Exception:
            time.sleep(0.5); continue
        d = j.get("data") or []
        if not d:
            break
        rows.extend(d)
        after = d[-1]["fundingTime"]
        if int(after) <= start_ms:
            break
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df["ts"] = pd.to_datetime(df.fundingTime.astype(np.int64), unit="ms", utc=True)
    df["rate"] = pd.to_numeric(df.fundingRate, errors="coerce")
    df = df[["ts", "rate"]].sort_values("ts").drop_duplicates("ts").set_index("ts")
    df.to_parquet(f)
    return df


if __name__ == "__main__":
    import sys, concurrent.futures as cf
    tf = sys.argv[1] if len(sys.argv) > 1 else "15m"
    start = sys.argv[2] if len(sys.argv) > 2 else "2022-01-01"
    syms = (sys.argv[3].split(",") if len(sys.argv) > 3 else
            ["BTC", "ETH", "SOL", "LINK", "NEAR", "SUI", "APT", "HYPE",
             "PEPE", "WIF", "FET", "DOGE", "AVAX", "BNB", "XRP", "ADA"])

    def job(s):
        try:
            d = get(s, tf, start)
            return s, len(d), (str(d.index[0])[:10] if len(d) else "-")
        except Exception as e:
            return s, -1, str(e)[:60]

    with cf.ThreadPoolExecutor(int(os.getenv("OKX_THREADS","8"))) as ex:
        for s, n, a in ex.map(job, syms):
            print(f"{s:6s} {tf:4s} bars={n:>7} from={a}", flush=True)
