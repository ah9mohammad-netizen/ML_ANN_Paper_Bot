# ML_ANN_Paper_Bot — v2 rebuild

Two things live here.

**`research/`** — an honest backtesting stack: an event-driven portfolio
backtester, a walk-forward optimiser, a look-ahead self-test, and the scripts
that produced every number in `research/FINDINGS.md`.

**`bot/`** — the paper-trading bot, rewritten. It imports `strategies.py` from
the same source the backtester uses, so what you test is what runs.

---

## Start here

Read `research/FINDINGS.md` first. The headline: **the strategy currently
deployed loses money with statistical certainty** (profit factor 0.77 over
2,168 trades and 4.7 years, t-statistic −8.13), and none of the six rebuilt
strategies cleared a pre-declared significance bar either. This repo is
therefore a measurement instrument, not a money printer. Treat it that way.

The old `Quantitive_Audit.md` and `Research_Report.md` should be deleted. They
describe features that were never implemented (ADX-adaptive sizing, Heikin-Ashi
confirmation, market-structure shift), cite win rates of 99.3% that come from
strategies with no stop-loss, and conclude "APPROVED FOR LIVE DEPLOYMENT" on a
sample of 99 trades that netted +0.6%.

---

## Research stack

```
research/
  okx_data.py       OKX downloader, parquet cache, resumable, closed bars only
  indicators.py     strictly causal indicators — no shift(-n) anywhere
  strategies.py     signal library, shared with the live bot
  engine.py         event-driven backtester (fills, wicks, fees, funding, liq)
  portfolio.py      weights-based simulator for continuously-held positions
  walkforward.py    rolling-window optimisation + parameter stability
  regimes.py        causal bull/bear/chop labelling from BTC daily structure
  test_lookahead.py the self-test every strategy must pass
```

### Rules the engine enforces

1. A signal from bar *i* is only computed after bar *i* has closed.
2. It fills at the **open of bar i+1**, plus slippage — never at the close you
   just looked at.
3. Exits are checked against the full high/low of every later bar, so a wick
   through your stop is seen.
4. If one bar touches both the stop and the target, **the stop fills**.
5. A gap through a level fills at the bar's open, not at the level.
6. Taker fees both legs, extra slippage on stops, funding at 8h settlements,
   liquidation checked against the bar extreme at the configured leverage.

### Run it

```bash
pip install -r bot/requirements.txt
cd research

python okx_data.py 15m 2022-01-01 BTC,ETH,SOL,LINK,XRP    # ~15 min, resumable
python test_lookahead.py                                   # must print ALL CLEAN
python run_legacy15.py                                     # the old strategy, honestly
python run_final.py                                        # the rebuilt ones, walk-forward
```

`test_lookahead.py` recomputes each strategy on truncated history and checks the
signal at bar *i* is identical to the one computed on the full series. Any
strategy you add must pass it before its backtest means anything.

---

## The bot

```
bot/
  runner.py     main loop — bar-driven, not poll-driven
  botconfig.py  env config with validation that refuses nonsense
  feed.py       closed-bar-only feed, incremental, multi-venue, staleness aware
  broker.py     paper execution — exits evaluated on closed 1-MINUTE bars
  risk.py       pre-trade gates and kill switches
  store.py      SQLite with schema versioning and equity snapshots
  metrics.py    expectancy in R, bootstrap CIs, and a verdict that says
                "insufficient data" until 100 closed trades
  tg.py         Telegram UI
```

### What changed vs the old bot

| | old | new |
|---|---|---|
| Exit checking | last 15m **close** vs SL/TP | every closed **1m bar's high/low** |
| Entry price | the close the signal fired on | market on the next bar, with slippage |
| Repainting | Binance failover returned unclosed bars | every venue filters to closed bars |
| Data cost | 1,000 candles × 12 pairs every 60s | incremental top-up, cached |
| Gross exposure | up to 9× equity, no correlation cap | 2× cap + max 3 same-side alts |
| Kill switches | none | daily loss, drawdown, consecutive losses, stale data |
| Storage | silent loss on Railway redeploy | boots with a loud warning |
| Reported metrics | win rate and PF | + expectancy in R, bootstrap CI, t-stat, DD |
| Backtest parity | separate, incompatible code | same `strategies.py` |

### Deploy on Railway

Attach a **volume** first, then set `DB_PATH` to a path inside it. Without one
every trade record is destroyed on each redeploy — which is why the old bot's
reports keep restarting at n=15, n=32, n=17. The bot prints a warning at boot if
it detects this.

```
# --- persistence: MUST point inside the mounted Railway volume
DB_PATH=/data/bot.db

# --- universe & strategy (FET has no OKX perpetual; it is dropped at boot)
PAIRS=BTC,ETH,SOL,LINK,NEAR,SUI,APT,HYPE,PEPE,WIF,DOGE
TIMEFRAME=8h
STRATEGIES=donchian
MARKET_FILTER=align
MODE=paper

# --- risk
STARTING_EQUITY=1000
RISK_PER_TRADE=0.0075
LEVERAGE=4
MAX_OPEN=5
MAX_GROSS_NOTIONAL=2.0
MAX_NOTIONAL_PER_TRADE=0.6
MAX_CORRELATED=3

# --- kill switches
MAX_DAILY_LOSS=0.06
MAX_DRAWDOWN_HALT=0.25
MAX_CONSECUTIVE_LOSSES=8
STALE_DATA_HALT_MIN=90

# --- costs
TAKER_FEE=0.0005
SLIP_ENTRY_BPS=2
SLIP_STOP_BPS=5
APPLY_FUNDING=true

# --- ops
POLL_SECONDS=30
HEARTBEAT_HOURS=12
TELEGRAM_BOT_TOKEN=...
TELEGRAM_CHAT_ID=...
```

Expect roughly **2-3 trades per month across the whole universe** and days of
silence between them. That is the strategy working as designed, not a fault.
`/risk` and `/signals` show you it is alive.

`MODE=live` is deliberately not implemented. The config validator rejects it.

### Telegram

`/stats` `/open` `/recent` `/signals` `/risk` `/config` `/pause` `/resume`
`/halt` `/flat` `/backup` `/reset CONFIRM <amount>`

`/stats` reports expectancy in R with a confidence interval alongside win rate,
and refuses to call anything an edge below 100 closed trades.

---

## Honest limits

- Backtests use OKX perpetual data. Binance and Bybit block the machine this was
  built on. Costs are modelled at 5bps taker, which is a retail tier — verify
  against your own venue.
- Funding history from OKX's public endpoint only reaches back ~3 months, so
  funding drag is understated in the long backtests. Holds average 2–6 hours in
  most tests, where funding is close to irrelevant; it matters for the daily
  portfolio results.
- Six strategy families were tested across five timeframes. Every extra test
  raises the chance one looks good by accident. That is why the final bar was
  set before the last run: t > 2.5, bootstrap PF 95% CI lower bound > 1.10, and
  positive expectancy in at least two of three regimes.
- Paper trading for one or two weeks produces ~20 trades. That cannot separate
  an edge from noise, and `metrics.py` will tell you so.
