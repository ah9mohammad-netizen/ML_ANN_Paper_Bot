# MTF Local-Opt Paper Bot

Paper-trading bot for the final pair-direction strategy developed from the research phases.

This repo keeps the same Railway-friendly bot shell as the previous `MTF_ANN_Paper_Bot`, but the strategy has changed:

- No ANN artifact is loaded.
- Pair-direction configs are explicit and transparent.
- Every indicator/parameter acts as a decision maker.
- The strategy accumulates decision-maker effects into:
  - `probability` = probability-like take score
  - `predicted_R` = expected-R proxy
- The bot opens paper trades only when the active pair-direction config passes its thresholds.

## Active paper universe

Default active pairs:

```text
BTC, ETH, SOL, AVAX
```

HYPE is omitted due insufficient data.
BNB remains watch-only and is disabled by default.

## Pair-direction statuses

| Pair | Long | Short |
|---|---|---|
| BTC | Active | Active |
| ETH | Active | Active weak |
| SOL | Active | Active |
| AVAX | Disabled by default | Active |
| BNB | Disabled/watch | Disabled/watch |
| HYPE | Omitted | Omitted |

`AVAX LONG` was a research candidate but underperformed in individual backtests, so it is present in `strategy.py` but disabled.

## Strategy timeframe

```text
Entry/decision timeframe: 4h
Regime timeframe: derived 1D from 4h candles
```

The bot checks repeatedly, but signals only meaningfully update when a new confirmed 4h candle appears.

## Risk profile defaults

These defaults are based on the conservative Phase 17/18 backtests:

```text
STARTING_BALANCE=100
RISK_PER_TRADE_PCT=1.0
LEVERAGE=3
MAX_OPEN_POSITIONS=2
SAME_PAIR_LOCK=true
TAKER_FEE_PCT=0.04
SLIPPAGE_PCT=0.02
```

## Railway deployment

1. Create a new GitHub repo.
2. Copy/push these files.
3. Create a Railway project from the GitHub repo.
4. Add environment variables:

```text
TELEGRAM_BOT_TOKEN=your_token
TELEGRAM_CHAT_ID=your_chat_id
STARTING_BALANCE=100
RISK_PER_TRADE_PCT=1.0
LEVERAGE=3
MAX_OPEN_POSITIONS=2
PAIRS=BTC,ETH,SOL,AVAX
POLL_SECONDS=120
```

5. Railway should detect:

```text
Procfile: worker: python main.py
```

## Optional environment variables

```text
PAUSED=false
DB_PATH=paper_bot.db
MAX_TOTAL_MARGIN_PCT=70
MAX_MARGIN_PER_POSITION_PCT=35
MAX_NOTIONAL_PCT=150
SAME_PAIR_LOCK=true
CLOSE_ON_TIME_EXIT=true
TAKER_FEE_PCT=0.04
SLIPPAGE_PCT=0.02
```

## Telegram commands

The old Telegram UI is preserved. Use commands supported by `telegram_ui.py`, such as stats/open/recent depending on the current implementation.

## Important warning

This is for **paper trading only**.

The backtest was positive under conservative assumptions, but live paper performance can differ due to:

- exchange candle differences
- spread/slippage
- funding rates
- unavailable order-book information
- regime shifts
- Railway restarts / DB persistence issues

Run for at least one month in paper mode before considering live trading.
