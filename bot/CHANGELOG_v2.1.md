# v2.1 — practical operation

Everything here is about how the bot *runs*. No strategy logic changed, so the
walk-forward result it was validated on still applies unchanged.

## The problem, measured

Instrumenting every outbound HTTP call on the deployed 36-symbol 8h config:

```
BOOT   (verify_universe, need=60)        36 calls   20.1s
CYCLE 1 (cold cache, need=400)          324 calls  186.3s
CYCLE 2 (warm cache — repeats forever)   36 calls   19.6s
        -> 1.20 req/s sustained = 103,680 requests/day
        -> 19.6s of every 30s cycle spent fetching
response codes: 396x HTTP 200, zero OKX error codes, zero rate limiting
```

**It was not being rate limited.** But it re-fetched every symbol every 30
seconds for data that changes three times a day, spent two-thirds of each cycle
doing it, and had no headroom left if OKX slowed down. One bad latency day and
cycles would have overlapped.

Root cause: `signal_for()` called the feed *before* checking whether the bar had
changed. The loop was bar-driven in intent and poll-driven in fact.

## The fix

**`feed.py` — rewritten**

- **Bar-aligned cache.** `bars()` returns the cache untouched unless a new bar
  of that timeframe has actually closed. The common path makes zero network
  calls.
- **Token bucket** (`API_RATE_PER_SEC`, default 6/s against OKX's 20/s limit).
  Exceeding the venue limit is now structurally impossible.
- **Concurrent refresh** via `bars_many()` — the once-per-bar sweep of 36
  symbols takes ~3s instead of ~20s, without raising the request rate.
- **Circuit breaker.** Three consecutive failures on a symbol trigger
  exponential backoff to 300s instead of hammering a struggling venue.
- **Telemetry.** Every request is counted, timed and bucketed by response code.
- `due()` / `seconds_to_next_close()` so the runner can schedule instead of poll.

**`runner.py` — rewritten**

- Genuinely bar-driven: `scan()` runs only when `feed.due(timeframe)` is true.
- Concurrent warmup at boot, then an immediate first scan rather than waiting
  up to 8 hours for the next close.
- `signal_for()` reads from cache and never triggers I/O.
- 1-minute exit data for open positions is prefetched in one concurrent batch.
- Adaptive sleep: `POLL_SECONDS` while holding risk, `IDLE_POLL_SECONDS` when
  flat, never overshooting the next bar close.
- Equity snapshots throttled to `SNAPSHOT_SECONDS` (was one row per 30s =
  2,880 rows/day).
- Open messages no longer advertise a fake "R:R 20.00" for a trailing-exit
  strategy — they say `trailing 4.5 ATR, no fixed target`.

**`tg.py`**

- `/api` — request counts, error counts, throttling, latency percentiles,
  circuit-breaker state, scans completed, time to next scan.
- `/diag` reads from cache and reports data freshness alongside the market gate.

## Measured after

```
BOOT                                324 calls   52.4s   (one-time)
SCAN (per 8h bar close)               5 calls    3.0s
STEADY STATE, 1 position open         4 calls / 240s = 1,440/day
STEADY STATE, flat                    ~0 calls
errors 0 · throttled 0 · p50 496ms · p95 594ms · circuit open: none
```

**103,680 → ~1,440 requests/day. A 72× reduction.**

## New environment variables

| var | default | purpose |
|---|---|---|
| `IDLE_POLL_SECONDS` | 60 | loop interval when flat |
| `API_RATE_PER_SEC` | 6 | token-bucket ceiling (OKX allows 20) |
| `API_WORKERS` | 6 | concurrency for batch refreshes |
| `SNAPSHOT_SECONDS` | 300 | equity-curve write interval |

All optional — the defaults are correct for a 36-symbol 8h universe.
