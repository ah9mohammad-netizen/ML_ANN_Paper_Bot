"""Market data feed.

Guarantees the rest of the bot depends on:
  * only CLOSED bars are ever returned — no repainting, on any venue;
  * bars are cached and extended incrementally, so one new bar costs one small
    request instead of re-downloading 1,000 candles per symbol per minute;
  * every venue adapter applies the same closed-bar rule, so a failover cannot
    silently change the semantics (this is the bug in the old data_client);
  * staleness is measurable, so the runner can halt instead of trading blind.
"""
import time
import threading
from datetime import datetime, timezone

import numpy as np
import pandas as pd
import requests

TF_MS = {"1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
         "30m": 1_800_000, "1h": 3_600_000, "2h": 7_200_000,
         "4h": 14_400_000, "8h": 28_800_000, "12h": 43_200_000,
         "1d": 86_400_000}

# Timeframes no venue serves natively: built by resampling a base timeframe.
# The resample is left-labelled and left-closed, and the final (still forming)
# bucket is dropped, so only genuinely closed bars are ever returned.
SYNTHETIC = {"8h": ("4h", "8h", 2), "12h": ("4h", "12h", 3)}

_OKX_BAR = {"1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m",
            "1h": "1H", "4h": "4H", "1d": "1D"}
_BINANCE_BAR = {k: k for k in TF_MS}
_BYBIT_BAR = {"1m": "1", "3m": "3", "5m": "5", "15m": "15", "30m": "30",
              "1h": "60", "4h": "240", "1d": "D"}

COLS = ["open", "high", "low", "close", "volume"]


def _frame(rows):
    df = pd.DataFrame(rows, columns=["ts"] + COLS)
    df["ts"] = pd.to_datetime(df.ts.astype(np.int64), unit="ms", utc=True)
    for c in COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna().sort_values("ts").drop_duplicates("ts").set_index("ts")


# ── venue adapters: each returns CLOSED bars only ──────────────────────

def _okx(symbol, tf, limit, inst="SWAP", hosts=("https://www.okx.com",
                                                "https://aws.okx.com")):
    inst_id = f"{symbol}-USDT-SWAP" if inst == "SWAP" else f"{symbol}-USDT"
    for h in hosts:
        try:
            r = requests.get(f"{h}/api/v5/market/candles",
                             params={"instId": inst_id, "bar": _OKX_BAR[tf],
                                     "limit": str(min(limit, 300))}, timeout=10)
            j = r.json()
            if j.get("code") != "0" or not j.get("data"):
                continue
            rows = [c[:6] for c in j["data"] if c[8] == "1"]   # confirm == 1
            if rows:
                return _frame(rows)
        except Exception:
            continue
    return pd.DataFrame()


def _binance(symbol, tf, limit, inst="SWAP",
             hosts=("https://fapi.binance.com", "https://api.binance.com")):
    for h in hosts:
        try:
            path = "/fapi/v1/klines" if "fapi" in h else "/api/v3/klines"
            r = requests.get(h + path,
                             params={"symbol": f"{symbol}USDT",
                                     "interval": _BINANCE_BAR[tf],
                                     "limit": min(limit, 1000)}, timeout=10)
            j = r.json()
            if not isinstance(j, list) or not j:
                continue
            now_ms = int(time.time() * 1000)
            # drop the in-progress bar: close_time (index 6) must be in the past
            rows = [[c[0], c[1], c[2], c[3], c[4], c[5]] for c in j
                    if int(c[6]) < now_ms]
            if rows:
                return _frame(rows)
        except Exception:
            continue
    return pd.DataFrame()


def _bybit(symbol, tf, limit, inst="SWAP"):
    try:
        r = requests.get("https://api.bybit.com/v5/market/kline",
                         params={"category": "linear", "symbol": f"{symbol}USDT",
                                 "interval": _BYBIT_BAR[tf],
                                 "limit": min(limit, 1000)}, timeout=10)
        j = r.json()
        lst = (j.get("result") or {}).get("list") or []
        if not lst:
            return pd.DataFrame()
        step = TF_MS[tf]
        now_ms = int(time.time() * 1000)
        rows = [[c[0], c[1], c[2], c[3], c[4], c[5]] for c in lst
                if int(c[0]) + step <= now_ms]
        return _frame(rows) if rows else pd.DataFrame()
    except Exception:
        return pd.DataFrame()


ADAPTERS = {"okx": _okx, "binance": _binance, "bybit": _bybit}


def _okx_backfill(symbol, tf, need, inst="SWAP", host="https://www.okx.com"):
    """Page backwards through /history-candles until `need` closed bars are in
    hand. The plain /candles endpoint caps at 300, which is not enough warmup
    for a 200-period EMA — this is what fills the gap at boot."""
    inst_id = f"{symbol}-USDT-SWAP" if inst == "SWAP" else f"{symbol}-USDT"
    rows, after, guard, last = [], None, 0, None
    while len(rows) < need and guard < 60:
        guard += 1
        p = {"instId": inst_id, "bar": _OKX_BAR[tf], "limit": "100"}
        if after:
            p["after"] = str(after)
        try:
            j = requests.get(f"{host}/api/v5/market/history-candles",
                             params=p, timeout=12).json()
        except Exception:
            break
        d = j.get("data") or []
        if not d:
            break
        rows.extend([c[:6] for c in d if c[8] == "1"])
        after = d[-1][0]
        if last is not None and int(after) >= last:
            break
        last = int(after)
        time.sleep(0.06)
    return _frame(rows) if rows else pd.DataFrame()


class Feed:
    def __init__(self, venue="okx", inst="SWAP", warmup=400, fallbacks=("binance", "bybit")):
        self.primary = venue
        self.order = [venue] + [v for v in fallbacks if v != venue]
        self.inst = inst
        self.warmup = warmup
        self.cache = {}                      # (symbol, tf) -> df
        self.last_ok = {}                    # (symbol, tf) -> epoch seconds
        self.venue_used = {}
        self.lock = threading.Lock()

    def _fetch(self, symbol, tf, limit, deep=False):
        if deep and self.primary == "okx":
            df = _okx_backfill(symbol, tf, limit, self.inst)
            if len(df) >= min(limit, 250):
                self.venue_used[(symbol, tf)] = "okx(history)"
                return df
        for v in self.order:
            df = ADAPTERS[v](symbol, tf, limit, self.inst)
            if len(df):
                self.venue_used[(symbol, tf)] = v
                return df
        return pd.DataFrame()

    @staticmethod
    def _resample(df, rule):
        o = df.resample(rule, label="left", closed="left").agg(
            {"open": "first", "high": "max", "low": "min",
             "close": "last", "volume": "sum"}).dropna()
        # drop the bucket that has not finished yet
        step = pd.Timedelta(rule)
        now = pd.Timestamp.now(tz="UTC")
        return o[o.index + step <= now]

    def bars(self, symbol, tf, need=None):
        """Return the cached closed-bar history, topping it up incrementally.

        For a synthetic timeframe (8h, 12h) the base timeframe is fetched and
        resampled here, so callers never see a partially formed bucket."""
        if tf in SYNTHETIC:
            base, rule, mult = SYNTHETIC[tf]
            b = self.bars(symbol, base, need=(need or self.warmup) * mult + 20)
            if b is None or not len(b):
                return pd.DataFrame()
            out = self._resample(b, rule)
            self.cache[(symbol, tf)] = out
            self.last_ok[(symbol, tf)] = time.time()
            return out

        key = (symbol, tf)
        need = need or self.warmup
        with self.lock:
            cur = self.cache.get(key)
            short = cur is None or len(cur) < need
            limit = need + 10 if short else 60
            # deep backfill only when we genuinely lack warmup history
            new = self._fetch(symbol, tf, limit, deep=short and need > 250)
            if not len(new):
                return cur if cur is not None else pd.DataFrame()
            df = new if cur is None else pd.concat([cur, new])
            df = df[~df.index.duplicated(keep="last")].sort_index()
            if len(df) > need * 3:
                df = df.iloc[-need * 2:]
            self.cache[key] = df
            self.last_ok[key] = time.time()
            return df

    def last_closed_ts(self, symbol, tf):
        d = self.cache.get((symbol, tf))
        return None if d is None or not len(d) else d.index[-1]

    def staleness_minutes(self, symbol, tf):
        """Minutes behind the bar that SHOULD be the latest closed one."""
        ts = self.last_closed_ts(symbol, tf)
        if ts is None:
            return 1e9
        step_ms = TF_MS[tf]
        now_ms = int(pd.Timestamp.now(tz="UTC").timestamp() * 1000)
        expected = (now_ms // step_ms) * step_ms - step_ms   # last closed bar open
        have = int(ts.timestamp() * 1000)
        return max(0.0, (expected - have) / 60000)

    def price(self, symbol):
        d = self.cache.get((symbol, "1m"))
        if d is not None and len(d):
            return float(d.close.iloc[-1])
        d = self.bars(symbol, "1m", need=5)
        return float(d.close.iloc[-1]) if len(d) else None
