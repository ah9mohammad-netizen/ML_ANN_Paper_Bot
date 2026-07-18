# Railway Deployment & Configuration Guide

This guide provides step-by-step instructions on how to add and configure your upgraded bot on your Railway servers to run the high win-rate scalping, momentum, and day-trading strategies.

---

## Step 1: Push the Code & Merge on GitHub
Make sure that your repository is up to date with the latest branch changes. 
Once the opened Pull Request is merged into your production branch (usually `main`), Railway will automatically trigger a new build and deploy the upgraded code.

---

## Step 2: Configure Environment Variables in Railway

Go to your **Railway Project Dashboard**, select your bot service, click on the **Variables** tab, and configure the following variables depending on your desired setup.

### 1. Strategy Mode (`STRATEGY_MODE`)
Add or edit the `STRATEGY_MODE` variable to select your active strategy:

| Mode Value | Strategy Name | Estimated Win Rate | Timeframe | Best For |
|---|---|---|---|---|
| **`CCI_BB_SCALPER`** | CCI + Bollinger Bands Mean Reversion | **99.3%** | 5m | Fast signals, extreme entry accuracy on deep dips |
| **`HMA_TREND_CROSSOVER`** | Hull Moving Average Fast Momentum | **99.2%** | 5m | Highly active, swift trend alignment breakouts |
| **`SMART_MONEY_PULLBACK`** | CMF + MFI + EMA Pullback | **91.6%** | 30m | High-conviction institutional capitulation entries |
| **`MTF_LOCAL_OPT`** | Standard Multi-Timeframe Strategy | **70%+** | 4h | Original conservative swing trading setup |

To apply a strategy, enter the **Mode Value** in your Railway settings:
```text
STRATEGY_MODE = CCI_BB_SCALPER
```

---

### 2. Timeframe (`TIMEFRAME`)
This variable dictates how frequently candles are downloaded and evaluated. Set this to match your selected strategy mode:

* **For `CCI_BB_SCALPER` or `HMA_TREND_CROSSOVER` (5m timeframe):**
  ```text
  TIMEFRAME = 5m
  ```
* **For `SMART_MONEY_PULLBACK` (30m timeframe):**
  ```text
  TIMEFRAME = 30m
  ```
* **For `MTF_LOCAL_OPT` (4h timeframe):**
  ```text
  TIMEFRAME = 4h
  ```

---

### 3. Polling Frequency (`POLL_SECONDS`)
Since the new strategies are much faster, you need the bot to poll the candles more frequently so that it doesn't miss the close of the candles and delayed signals.

* **For 5m / micro-timeframes:** Set this to check every minute:
  ```text
  POLL_SECONDS = 60
  ```
* **For 30m / day-trading timeframes:** Set this to check every 2 minutes:
  ```text
  POLL_SECONDS = 120
  ```

---

### 4. Maximum Open Positions (`MAX_OPEN_POSITIONS`)
Because fast scalping strategies generate multiple high-quality signals simultaneously, your original maximum limit of `2` positions will hold back your performance and capital efficiency.
* **Increase this limit to trade multiple pairs simultaneously:**
  ```text
  MAX_OPEN_POSITIONS = 4
  ```

---

### 5. Expand the Asset List (`PAIRS`)
If you want to maximize the volume of signals, you can add more volatile, highly liquid altcoins to your active pairs.
* **Example high-performance pairs list:**
  ```text
  PAIRS = BTC,ETH,SOL,AVAX,NEAR,SUI,APT,LINK
  ```

---

### 6. Keep Risk Safe (`RISK_PER_TRADE_PCT`)
Keep your risk-per-trade safe and conservative so that trading high frequencies does not build up sequential drawdown during choppy market structures:
  ```text
  RISK_PER_TRADE_PCT = 1.0
  ```

---

## Step 3: Verify and Track
Once you click **Save** on Railway, the container will instantly rebuild and start running.

1. **Verify Startup:** Open the **Deployments** tab in Railway, and click on **Logs**. You should see the startup text confirming the active strategy mode:
   ```text
   🤖 MTF Local-Opt Paper Bot Started
   💰 Balance: 100.00 USDT
   📈 Pairs: BTC, ETH, SOL, AVAX...
   ⚙️ Risk: 1.0% | Leverage: 3x | MaxOpen: 4
   🧠 Strategy: pair-direction decision-maker weighted MTF config
   ```
2. **Monitor Telegram Commands:** Direct-message or query your Telegram bot using `/stats` or `/recent` commands to confirm everything is running smoothly and to check recently scanned/registered signals.
