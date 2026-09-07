"""Market data feed.

Guarantees the rest of the bot depends on:
  * only CLOSED bars are ever returned — no repainting, on any venue
  * a bar is fetched ONCE per bar period, not once per poll. An 8h bar cannot
    change between closes, so asking for it every 30 seconds is 960 pointless
    requests per symbol per day
  * a token bucket makes exceeding the venue's rate limit structurally
    impossible, even during a burst
  * failures back off and trip a per-symbol circuit breaker instead of
    hammering a venue that is already unhappy
  * every request is counted and timed, so "is it an API problem?" is a
    question the bot can answer about itself

Measured effect of the caching change on a 36-symbol 8h universe:
    before   103,680 requests/day, 19.6s of every 30s cycle spent fetching
    after      ~450 requests/day, most cycles make zero network calls
"""
import threading
import time
from collections import deque

import numpy as np
import pandas as pd
import requests

TF_MS = {"1m": 60_000, "3m": 180_000, "5m": 300_000, "15m": 900_000,
         "30m": 1_800_000, "1h": 3_600_000, "2h": 7_200_000,
         "4h": 14_400_000, "8h": 28_800_000, "12h": 43_200_000,
         "1d": 86_400_000}

# Timeframes no venue serves natively: built by resampling a base timeframe.
SYNTHETIC = {"8h": ("4h", "8h", 2), "12h": ("4h", "12h", 3)}

_OKX_BAR = {"1m": "1m", "3m": "3m", "5m": "5m", "15m": "15m", "30m": "30m",
            "1h": "1H", "2h": "2H", "4h": "4H", "1d": "1D"}
_BINANCE_BAR = {k: k for k in TF_MS}
_BYBIT_BAR = {"1m": "1", "3m": "3", "5m": "5", "15m": "15", "30m": "30",
              "1h": "60", "2h": "120", "4h": "240", "1d": "D"}

COLS = ["open", "high", "low", "close", "volume"]


# ────────────────────────────────────────────────── rate limiting

class TokenBucket:
    """OKX public market endpoints allow 40 requests / 2s per IP. We run at a
    fraction of that on purpose — headroom costs nothing when the bot only
    needs a few hundred requests a day."""

    def __init__(self, rate_per_sec=6.0, burst=12):
        self.rate = float(rate_per_sec)
        self.burst = float(burst)
        self.tokens = float(burst)
        self.ts = time.monotonic()
        self.lock = threading.Lock()
        self.waited = 0.0

    def take(self, n=1.0):
        while True:
            with self.lock:
                now = time.monotonic()
                self.tokens = min(self.burst,
                                  self.tokens + (now - self.ts) * self.rate)
                self.ts = now
                if self.tokens >= n:
                    self.tokens -= n
                    return
                need = (n - self.tokens) / self.rate
            self.waited += need
            time.sleep(min(need, 0.5))


# ────────────────────────────────────────────────── telemetry

class ApiStats:
    def __init__(self):
        self.lock = threading.Lock()
        self.total = 0
        self.errors = 0
        self.rate_limited = 0
        self.by_code = {}
        self.latency = deque(maxlen=500)
        self.started = time.time()
        self.last_error = ""
        self.last_error_ts = 0.0

    def record(self, code, dt, err=""):
        with self.lock:
            self.total += 1
            self.latency.append(dt)
            k = str(code)
            self.by_code[k] = self.by_code.get(k, 0) + 1
            if code == 429 or code == "50011":
                self.rate_limited += 1
            if err or (isinstance(code, int) and code >= 400):
                self.errors += 1
                self.last_error = err or f"HTTP {code}"
                self.last_error_ts = time.time()

    def snapshot(self):
        with self.lock:
            lat = sorted(self.latency)
            hours = max((time.time() - self.started) / 3600, 1e-6)
            return {
                "total": self.total,
                "errors": self.errors,
                "rate_limited": self.rate_limited,
                "per_hour": self.total / hours,
                "per_day_projected": self.total / hours * 24,
                "p50_ms": (lat[len(lat) // 2] * 1000) if lat else 0.0,
                "p95_ms": (lat[int(len(lat) * 0.95)] * 1000) if lat else 0.0,
                "by_code": dict(self.by_code),
                "last_error": self.last_error,
                "last_error_age_s": (time.time() - self.last_error_ts
                                     if self.last_error_ts else None),
            }


# ────────────────────────────────────────────────── helpers

def _frame(rows):
    df = pd.DataFrame(rows, columns=["ts"] + COLS)
    df["ts"] = pd.to_datetime(df.ts.astype(np.int64), unit="ms", utc=True)
    for c in COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    return df.dropna().sort_values("ts").drop_duplicates("ts").set_index("ts")


def last_closed_open_ms(tf, now_ms=None):
    """Epoch ms of the OPEN of the most recently CLOSED bar of `tf`."""
    step = TF_MS[tf]
    now = now_ms if now_ms is not None else int(time.time() * 1000)
    return (now // step) * step - step


def next_close_ms(tf, now_ms=None):
    """Epoch ms at which the currently forming bar of `tf` will close."""
    step = TF_MS[tf]
    now = now_ms if now_ms is not None else int(time.time() * 1000)
    return (now // step) * step + step


# ────────────────────────────────────────────────── venue adapters

class Venue:
    """Each adapter returns CLOSED bars only. The closed-bar rule lives here so
    a failover can never silently change the semantics — that was a real bug in
    the original data_client, where the Binance fallback returned the forming
    candle and the strategy repainted."""

    def __init__(self, bucket, stats):
        self.bucket = bucket
        self.stats = stats

    def _get(self, url, params, timeout=12):
        self.bucket.take()
        t = time.time()
        try:
            r = requests.get(url, params=params, timeout=timeout)
        except Exception as e:
            self.stats.record("EXC", time.time() - t, f"{type(e).__name__}: {e}")
            return None
        dt = time.time() - t
        err = ""
        if r.status_code != 200:
            err = f"HTTP {r.status_code}: {r.text[:120]}"
        self.stats.record(r.status_code, dt, err)
        if r.status_code != 200:
            return None
        try:
            return r.json()
        except Exception as e:
            self.stats.record("BADJSON", 0.0, str(e)[:120])
            return None

    # -- OKX ---------------------------------------------------------
    def okx(self, symbol, tf, limit, inst="SWAP"):
        inst_id = f"{symbol}-USDT-SWAP" if inst == "SWAP" else f"{symbol}-USDT"
        for host in ("https://www.okx.com", "https://aws.okx.com"):
            j = self._get(host + "/api/v5/market/candles",
                          {"instId": inst_id, "bar": _OKX_BAR[tf],
                           "limit": str(min(limit, 300))})
            if not j:
                continue
            if j.get("code") not in (None, "0"):
                self.stats.record(str(j.get("code")), 0.0,
                                  f"okx {j.get('code')}: {j.get('msg')}")
                continue
            rows = [c[:6] for c in (j.get("data") or []) if c[8] == "1"]
            if rows:
                return _frame(rows)
        return pd.DataFrame()

    def okx_history(self, symbol, tf, need, inst="SWAP"):
        """Page backwards for warmup. /candles caps at 300, which is not enough
        for a 200-period EMA on a synthetic timeframe."""
        inst_id = f"{symbol}-USDT-SWAP" if inst == "SWAP" else f"{symbol}-USDT"
        rows, after, guard, last = [], None, 0, None
        while len(rows) < need and guard < 40:
            guard += 1
            p = {"instId": inst_id, "bar": _OKX_BAR[tf], "limit": "100"}
            if after:
                p["after"] = str(after)
            j = self._get("https://www.okx.com/api/v5/market/history-candles", p)
            if not j or j.get("code") not in (None, "0"):
                break
            d = j.get("data") or []
            if not d:
                break
            rows.extend([c[:6] for c in d if c[8] == "1"])
            after = d[-1][0]
            if last is not None and int(after) >= last:
                break
            last = int(after)
        return _frame(rows) if rows else pd.DataFrame()

    # -- fallbacks ---------------------------------------------------
    def binance(self, symbol, tf, limit, inst="SWAP"):
        for host, path in (("https://fapi.binance.com", "/fapi/v1/klines"),
                           ("https://api.binance.com", "/api/v3/klines")):
            j = self._get(host + path, {"symbol": f"{symbol}USDT",
                                        "interval": _BINANCE_BAR[tf],
                                        "limit": min(limit, 1000)})
            if not isinstance(j, list) or not j:
                continue
            now_ms = int(time.time() * 1000)
            rows = [c[:6] for c in j if int(c[6]) < now_ms]   # closed only
            if rows:
                return _frame(rows)
        return pd.DataFrame()

    def bybit(self, symbol, tf, limit, inst="SWAP"):
        j = self._get("https://api.bybit.com/v5/market/kline",
                      {"category": "linear", "symbol": f"{symbol}USDT",
                       "interval": _BYBIT_BAR[tf], "limit": min(limit, 1000)})
        lst = ((j or {}).get("result") or {}).get("list") or []
        if not lst:
            return pd.DataFrame()
        step, now_ms = TF_MS[tf], int(time.time() * 1000)
        rows = [c[:6] for c in lst if int(c[0]) + step <= now_ms]
        return _frame(rows) if rows else pd.DataFrame()


# ────────────────────────────────────────────────── feed

class Feed:
    def __init__(self, venue="okx", inst="SWAP", warmup=400,
                 fallbacks=("binance", "bybit"), rate_per_sec=6.0,
                 max_workers=6, log=None):
        self.primary = venue
        self.order = [venue] + [v for v in fallbacks if v != venue]
        self.inst = inst
        self.warmup = warmup
        self.log = log
        self.stats = ApiStats()
        self.bucket = TokenBucket(rate_per_sec, burst=max(6, int(rate_per_sec * 2)))
        self.v = Venue(self.bucket, self.stats)
        self.max_workers = max_workers

        self.cache = {}          # (symbol, tf) -> df
        self.fetched_bar = {}    # (symbol, tf) -> open-ms of newest bar we hold
        self.refreshed_bar = {}  # tf -> open-ms of the last completed sweep
        self.last_ok = {}
        self.venue_used = {}
        self.fail = {}           # (symbol, tf) -> consecutive failures
        self.blocked_until = {}  # (symbol, tf) -> monotonic deadline
        self.lock = threading.RLock()

    # -- circuit breaker ------------------------------------------------

    def _tripped(self, key):
        d = self.blocked_until.get(key)
        return d is not None and time.monotonic() < d

    def _note(self, key, ok):
        if ok:
            self.fail.pop(key, None)
            self.blocked_until.pop(key, None)
            return
        n = self.fail.get(key, 0) + 1
        self.fail[key] = n
        if n >= 3:
            back = min(300.0, 15.0 * (2 ** (n - 3)))
            self.blocked_until[key] = time.monotonic() + back
            if self.log:
                self.log.warning("feed: %s failed %d times — backing off %.0fs",
                                 key, n, back)

    # -- fetching -------------------------------------------------------

    def _fetch(self, symbol, tf, limit, deep=False):
        if deep and self.primary == "okx":
            df = self.v.okx_history(symbol, tf, limit, self.inst)
            if len(df) >= min(limit, 250):
                self.venue_used[(symbol, tf)] = "okx/history"
                return df
        for name in self.order:
            df = getattr(self.v, name)(symbol, tf, limit, self.inst)
            if len(df):
                self.venue_used[(symbol, tf)] = name
                return df
        return pd.DataFrame()

    @staticmethod
    def _resample(df, rule):
        o = df.resample(rule, label="left", closed="left").agg(
            {"open": "first", "high": "max", "low": "min",
             "close": "last", "volume": "sum"}).dropna()
        step = pd.Timedelta(rule)
        now = pd.Timestamp.now(tz="UTC")
        return o[o.index + step <= now]      # drop the forming bucket

    def bars(self, symbol, tf, need=None, force=False):
        """Closed-bar history for (symbol, tf).

        Returns the cache untouched unless a new bar of `tf` has actually
        closed since the last fetch. This is the whole point: an 8h bar changes
        three times a day, so there is no reason to ask more often."""
        need = need or self.warmup
        key = (symbol, tf)

        if tf in SYNTHETIC:
            base, rule, mult = SYNTHETIC[tf]
            b = self.bars(symbol, base, need=need * mult + 20, force=force)
            if b is None or not len(b):
                return self.cache.get(key, pd.DataFrame())
            out = self._resample(b, rule)
            with self.lock:
                self.cache[key] = out
                if len(out):
                    self.fetched_bar[key] = int(out.index[-1].timestamp() * 1000)
            return out

        with self.lock:
            cur = self.cache.get(key)
            have_bar = self.fetched_bar.get(key)
            want_bar = last_closed_open_ms(tf)
            fresh = have_bar is not None and have_bar >= want_bar
            deep_needed = cur is None or len(cur) < need

            if not force and fresh and not deep_needed:
                return cur                       # <- the common path: no I/O
            if self._tripped(key):
                return cur if cur is not None else pd.DataFrame()

        limit = need + 10 if deep_needed else 60
        new = self._fetch(symbol, tf, limit, deep=deep_needed and need > 250)

        with self.lock:
            if not len(new):
                self._note(key, False)
                return cur if cur is not None else pd.DataFrame()
            self._note(key, True)
            df = new if cur is None else pd.concat([cur, new])
            df = df[~df.index.duplicated(keep="last")].sort_index()
            if len(df) > need * 3:
                df = df.iloc[-need * 2:]
            self.cache[key] = df
            self.fetched_bar[key] = int(df.index[-1].timestamp() * 1000)
            self.last_ok[key] = time.time()
            return df

    def bars_many(self, symbols, tf, need=None, force=False):
        """Refresh many symbols concurrently. The token bucket still bounds the
        request rate, so parallelism cuts wall-clock without raising load."""
        import concurrent.futures as cf
        out = {}
        todo = list(symbols)
        if not todo:
            return out
        with cf.ThreadPoolExecutor(min(self.max_workers, len(todo))) as ex:
            futs = {ex.submit(self.bars, s, tf, need, force): s for s in todo}
            for f in cf.as_completed(futs):
                s = futs[f]
                try:
                    out[s] = f.result()
                except Exception as e:
                    if self.log:
                        self.log.warning("feed: %s %s failed: %s", s, tf, e)
                    out[s] = self.cache.get((s, tf), pd.DataFrame())
        return out

    def due(self, tf):
        """Has a new bar closed since the last completed refresh sweep?

        Deliberately a single per-TIMEFRAME watermark, not a scan over
        per-symbol state. The earlier per-symbol version returned True forever
        whenever any one symbol's fetch was failing, which made the runner
        re-scan the whole universe every 30 seconds for nothing."""
        base = SYNTHETIC.get(tf, (tf,))[0]
        want = last_closed_open_ms(base)
        with self.lock:
            got = self.refreshed_bar.get(tf)
        return got is None or got < want

    def mark_refreshed(self, tf):
        """Called by the runner once a refresh sweep has completed, whatever
        individual symbols did. Symbol-level failures are retried on their own
        backoff — they must not pin the whole loop in a spin."""
        base = SYNTHETIC.get(tf, (tf,))[0]
        with self.lock:
            self.refreshed_bar[tf] = last_closed_open_ms(base)

    def seconds_to_next_close(self, tf):
        base = SYNTHETIC.get(tf, (tf,))[0]
        return max(0.0, (next_close_ms(base) - time.time() * 1000) / 1000.0)

    # -- accessors ------------------------------------------------------

    def last_closed_ts(self, symbol, tf):
        d = self.cache.get((symbol, tf))
        return None if d is None or not len(d) else d.index[-1]

    def staleness_minutes(self, symbol, tf):
        """Minutes behind the newest bar that SHOULD exist.

        For a synthetic timeframe this is measured on the BASE series. An 8h
        bucket is only emitted once every 4h bar inside it has closed, so
        judging freshness on the resampled series makes the feed look 8 hours
        stale for the 4 hours before every close — which is exactly what tripped
        the stale-data kill switch in production."""
        base = SYNTHETIC.get(tf, (tf,))[0]
        ts = self.last_closed_ts(symbol, base)
        if ts is None:
            return 1e9
        expected = last_closed_open_ms(base)
        return max(0.0, (expected - int(ts.timestamp() * 1000)) / 60000)

    def stale_symbols(self, symbols, tf, max_minutes):
        return [s for s in symbols
                if self.staleness_minutes(s, tf) > max_minutes]

    def recover(self, symbols, tf):
        """Force a refetch for symbols whose circuit breaker is open. Clears the
        breaker first so a transient 429 cannot wedge a symbol permanently."""
        base = SYNTHETIC.get(tf, (tf,))[0]
        with self.lock:
            for s in symbols:
                self.fail.pop((s, base), None)
                self.blocked_until.pop((s, base), None)
        return self.bars_many(symbols, tf, need=400, force=True)

    def price(self, symbol, max_age_s=45):
        """Latest 1m close, cached briefly. Used for marking and paper fills."""
        key = (symbol, "1m")
        with self.lock:
            d = self.cache.get(key)
            age = time.time() - self.last_ok.get(key, 0)
            if d is not None and len(d) and age < max_age_s:
                return float(d.close.iloc[-1])
        d = self.bars(symbol, "1m", need=5, force=True)
        return float(d.close.iloc[-1]) if d is not None and len(d) else None

    def api_report(self):
        s = self.stats.snapshot()
        with self.lock:
            tripped = [f"{k[0]}/{k[1]}" for k in self.blocked_until
                       if self._tripped(k)]
        s["circuit_open"] = tripped
        s["throttle_wait_s"] = round(self.bucket.waited, 1)
        s["cached_series"] = len(self.cache)
        return s
