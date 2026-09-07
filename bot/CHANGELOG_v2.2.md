# v2.2 — the stale-data deadlock

## What happened in production

```
15:59:35 INFO    scan #481: 33 symbols, 0 API calls, 0.1s, 0 opened
16:00:05 WARNING HALTED: stale data for BNB,XRP,LTC,ETH — not trading blind
17:00:05 WARNING HALTED: stale data for BNB,XRP,LTC,ETH — not trading blind
   ... eleven consecutive hours, never recovering ...
```

Three of my bugs, compounding:

**1. A halt could block the refresh that would clear it.** In `run()` the order
was `halt_check()` → `if not halt: scan()`. Once staleness tripped the halt, the
scan never ran, so no data was ever fetched again, so the data stayed stale, so
the halt stayed on. **A permanent deadlock reachable from one transient 429.**
The API report shows exactly that trigger: 1,434 calls, 3 errors, 3 throttled.

**2. Staleness was measured on the wrong series.** For a synthetic timeframe,
`staleness_minutes` read the *resampled* 8h cache. An 8h bucket is only emitted
once both 4h bars inside it have closed, so for the four hours before every
close the resampled series legitimately lags by a whole bar — and the check read
that as "8 hours stale". Freshness is now measured on the **base** series.

**3. `STALE_DATA_HALT_MIN=20` is meaningless on 8h bars.** Twenty minutes is
shorter than the bar. Now `0 = auto`, resolving to 1.5 base bars (6h on an 8h
timeframe).

**4. `due()` spun the loop.** It scanned per-symbol state and returned True if
*any* symbol was behind — so four broken feeds made the runner re-scan all 33
symbols every 30 seconds forever (visible as `scan #470 … #481`, 0 API calls
each). Replaced with a single per-timeframe watermark set by `mark_refreshed()`.

## The fix

**Data is refreshed before any halt is evaluated.** Nothing the risk layer
decides can prevent the bot from seeing the market.

**Stale feeds degrade instead of stopping everything.** Symbols that are
genuinely behind are excluded from that scan; the rest trade normally. A full
halt now requires `STALE_HALT_FRACTION` (default 0.5) of the universe to be
dark. Four lagging feeds out of 33 is not a reason to stop trading the other 29.

**Feeds self-heal.** `recover_stale()` runs every `RECOVER_SECONDS` (default
600), clears the circuit breaker for lagging symbols and forces a refetch. The
old circuit breaker could be opened by a transient 429 and — because only
`scan()` called the feed — was never retried once the halt was on.

## Verified against the production failure

Four feeds forced to return HTTP 429 on every call, cache aged 12h past budget:

```
33-symbol universe (production ratio)
  stale budget = 360 min (auto from 8h)
  WARNING excluding 4 stale feed(s) this scan: BNB, ETH, LTC, XRP
  INFO    scan #1: 29 symbols, 6 API calls, 3.2s, 2 opened
  cycle 1: due=False  halt=NONE  scans=1
  WARNING stale feeds: ETH, XRP, BNB, LTC — forcing refetch
  -> 4/33 stale: traded the healthy 29, opened 2 positions, kept retrying

8-symbol universe (same 4 broken = 50%)
  ERROR   scan skipped: 4/8 feeds stale — tape is dark
  -> correctly halts when half the tape is gone, and still retries
```

`due=False` after the sweep — the 30-second scan spin is gone.

## New environment variables

| var | default | purpose |
|---|---|---|
| `STALE_DATA_HALT_MIN` | `0` (auto) | staleness budget; 0 = 1.5 base bars |
| `STALE_HALT_FRACTION` | `0.5` | fraction of dark feeds that halts everything |
| `RECOVER_SECONDS` | `600` | how often to retry lagging feeds |

All optional. **If you set `STALE_DATA_HALT_MIN=90` earlier, remove it** — leave
it unset so the automatic budget applies.

## After deploying

The bot will clear the halt by itself on the next cycle. If you want it clean
immediately, send `/resume` — but it is no longer required.
