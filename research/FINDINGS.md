# ML_ANN_Paper_Bot — audit, rebuild and evidence

Everything below comes from OKX perpetual futures data, 2022-01-01 to
2026-08-29: 15-minute bars (163,000 per pair, 10 pairs) plus 4-hour bars for a
wider 18-pair universe. Regime mix over the window: 32% bull, 28% bear, 40%
chop, labelled causally from BTC daily structure.

Cost model unless stated otherwise: 5bps taker each side, 2bps entry slippage,
5bps extra slippage on stop fills, funding charged at 8h settlements,
liquidation checked against the bar extreme.

---

## 1. What the current bot does

`strategy.py` implements two setups. `SMC_SWEEP_RECLAIM` on 15m for
BTC/ETH/SOL/AVAX/BNB/LINK/NEAR: a bar that wicks below the prior 36-bar low and
closes back above it, with volume ≥ 1.25× the 20-bar average, on the correct
side of EMA-200; stop 1.5 ATR, target 2.0 ATR. `CCI_BB_SCALPER` for everything
else: CCI-14 beyond ±135 with a close outside a Bollinger band, filtered by
EMA-100.

### Defects that change the results

**Wicks are invisible.** `paper_engine.update_positions()` fetches the last
closed 15m candle and compares its **close** to the stop and target. A candle
that spikes through the stop and closes back inside is recorded as still open.
Measured cost of this one bug on the full sample: profit factor 0.88 → 0.82.

**Fills are free.** Both stop and target are assumed to fill at exactly their
own price. Stops do not.

**Entry at the observed close.** The signal is computed from `d.iloc[-1]` and
the entry price is that same close. In the backtest that is a free look; the
honest equivalent is the next bar's open plus slippage.

**Inverted fallback reward:risk.** Any pair not named in `SCALPER_ASSET_CONFIG`
— that is **SUI, APT and DOGE**, three of the twelve configured pairs — falls
through to `{'tp_atr': 1.5, 'sl_atr': 3.0}`. A 0.5:1 reward-to-risk, needing a
67% win rate to break even before fees.

**Long wins ties.** `side = 'LONG' if long_sweep else 'SHORT'` — a bar that
sweeps both extremes is always taken long.

**Repainting on failover.** The OKX path filters `confirm != '0'`; the Binance
failover in `data_client.py` does not, so when OKX fails the bot fires signals
on a partially formed bar.

**Real leverage ≈ 9×.** `MAX_MARGIN_PER_POSITION_PCT=15` × `MAX_OPEN_POSITIONS=6`
× `LEVERAGE=10` allows $9,000 of notional on $1,000, across twelve assets that
are all the same directional bet. There is no correlation cap.

**State loss.** `DB_PATH=/data/paper_bot.db` needs a mounted Railway volume.
Without one the container either fails to open the database or loses every trade
on redeploy — which is the likely reason the reports restart at n=15, n=32, n=17.

**Efficiency.** 1,000 candles are re-downloaded for every pair every 60 seconds
(~120 HTTP calls per cycle) to compute one new bar.

**No ML.** The repo is named `ML_ANN_Paper_Bot`. There is no model.
`probability: 0.85` is a hardcoded constant, `meta.prob_th: 0.70` is never
applied, and `scikit-learn` and `joblib` are imported by `requirements.txt` and
used nowhere.

### The documentation describes a different bot

| Claimed in `Quantitive_Audit.md` | In the code |
|---|---|
| ADX regime-adaptive sizing (ADX<20 → 1.2/2.5 ATR) | `adx()` is defined and never called; TP/SL are hardcoded 2.0/1.5 |
| "Layer 2: Market Structure Shift" | absent |
| "Layer 3: Heikin-Ashi 1m confirmation" | absent |
| "0.35× ATR retest buffers" | absent |
| SUI/APT are Category 2 → SMC | they hit the `else` branch → CCI_BB |

`Research_Report.md` reports a "99.3% win rate" for CCI+BB and notes in the same
paragraph that the win rate is achieved by **disabling stop-losses**. A strategy
with no stop has a win rate approaching 100% and an unbounded left tail. That is
not an edge, it is a hidden liability.

`Quantitive_Audit.md` states **"APPROVED FOR LIVE DEPLOYMENT"** and projects
profit factor 4.75–5.82. That projection is produced by taking 99 closed trades,
ranking assets by outcome, keeping the top six as "superstars" (83.67% win rate)
and discarding the rest as "drags". Ranking any random set of assets and keeping
the winning half produces the same effect. **Delete both documents.**

---

## 2. The current strategy, measured honestly

15-minute bars, BTC/ETH/SOL/LINK/XRP, 2022-01-01 → 2026-08-29, starting equity
1,000 USDT at the deployed settings (1% risk, 10× leverage, 6 positions).

| assumptions | trades | win rate | payoff | PF | expectancy | net | max DD |
|---|---|---|---|---|---|---|---|
| A — repo's own: fill at signal close, exits polled on closes only, 4bps, no funding | 4,097 | 42.1% | 1.21 | 0.88 | −0.088R | **−99%** | 99% |
| B — A plus wicks made visible | 2,967 | 40.6% | 1.19 | 0.82 | −0.152R | **−99%** | 99% |
| C — next-bar fills, 5bps taker, stop slippage, funding | 2,168 | 41.0% | 1.10 | **0.77** | **−0.221R** | **−99%** | 99% |

**t-statistic on per-trade R: −8.13.** With 2,168 trades this is not variance.

By regime (C): bear PF 0.77, bull 0.73, chop 0.77 — it loses in all three.
By symbol: BTC −144, ETH −269, SOL −134, XRP −183, LINK +7.
Fees consume 23% of gross profit.

**On the 70% win rate.** It is real as a measurement and empty as a result.
`Quantitive_Audit.md` Phase 3 records 70.15% win rate with profit factor 1.03
and +0.70 USDT over 67 trades, because the average loss was 3.75× the average
win. High win rates are bought with tight targets and wide stops; they relabel
expectancy, they don't create it. The current geometry (2.0 ATR target, 1.5 ATR
stop) mechanically produces ~41%, and does.

---

## 3. Six rebuilt strategies, walk-forward tested

Parameters are fitted on a 365-day training window and scored only on the
following unseen 110 days; 12 folds, rolling to August 2026. Objective is the
t-statistic of per-trade expectancy, penalised for drawdown — not total return,
which rewards luck and leverage.

Every strategy passes `test_lookahead.py`: recomputed on truncated history, the
signal at bar *i* is identical to the one computed on the full series.

| strategy | timeframe | OOS trades | win% | PF | Sharpe | max DD | t |
|---|---|---|---|---|---|---|---|
| sweep/reclaim (rebuilt) | 15m | 1,472 | 33.4 | 0.75 | neg | — | neg |
| trend pullback | 15m | ~380/fold | — | 0.81–0.92 | neg | — | neg |
| squeeze breakout | 1h | 2,022 | 40.6 | 1.06 | 0.23 | 57.4% | 1.26 |
| squeeze breakout | 2h | 1,178 | 36.2 | 0.91 | −0.44 | 55.2% | −1.07 |
| squeeze breakout | 4h | 549 | 38.3 | 1.01 | 0.02 | 26.5% | 0.21 |
| squeeze breakout | 8h | 160 | 45.0 | 0.97 | −0.06 | 14.7% | −0.12 |
| donchian breakout | 1h | 1,400 | 33.9 | 0.94 | 0.37 | 61.7% | 1.32 |
| donchian breakout | 2h | 1,594 | 40.7 | 1.04 | 0.19 | 48.8% | 0.96 |
| donchian breakout | 4h | 685 | 35.8 | 0.99 | −0.05 | 42.3% | 0.05 |
| donchian breakout | 8h | 271 | 44.3 | 1.23 | 0.42 | 18.6% | 1.18 |
| **donchian + BTC-align** | **8h** | **152** | **38.8** | **1.96** | **0.92** | **15.4%** | **2.35** |
| TS-momentum portfolio | 1d | — | — | — | 1.03 | 65.7% | 1.88 |

### The timeframe result is the finding

Round-trip cost is a fixed ~14bps. Against a 15-minute scalp targeting 0.5% that
is a quarter of the gross edge; against an 8-hour trend trade targeting 6% it is
2%. Measured share of gross profit consumed by fees:

- legacy 15m: **23%**
- squeeze 1h: 6%
- donchian 8h: **1%**

Every strategy improved as the timeframe rose. The current bot runs at the
timeframe where costs are most punishing.

### A false positive I caught

The 4h squeeze first returned PF 1.37, Sharpe 1.03, t = 2.41 across 7 folds
ending April 2025 — a publishable-looking result. Extending the folds through
August 2026 collapsed it to PF 1.11, t = 1.02. The edge existed only in the
window first tested. This is the same mechanism that produced `Quantitive_Audit.md`,
and it is why every number here is quoted with the fold count and end date.

### The daily momentum portfolio is crypto beta

Vol-targeted time-series momentum across 10 assets, daily rebalance, walk-forward:
+301% over 3.3 years against BTC buy-and-hold's +172%, Sharpe 1.03 vs 0.88 —
but max drawdown 66% vs 51%, t = 1.88 (not significant), and Sharpe +2.13 in
bull regimes against **−2.45 in bear**. It is a leveraged long-crypto timing
overlay, not a market-independent edge.

---

## 4. The one configuration that survives

**Donchian channel breakout, 8-hour bars, long-only, aligned with BTC's daily
trend, Chandelier ATR trail, no fixed target.**

Entry: close above the prior 48-bar high, with BTC's daily close above its
100-day EMA and up over 30 days. Stop 3.0 ATR. Exit: a trailing stop at 4.5 ATR
below the highest price reached since entry. No profit target — the trail is
the exit.

### Out of sample (12 walk-forward folds, 2023-01 → 2026-08)

```
trades        125    (0.7/wk)   avg hold 240h
win rate      36.8%     payoff 3.38
profit factor 1.97     95% CI [1.15, 3.13]
expectancy    +0.542 R  (+3.96 USDT/trade)
net           +495.58 USDT   +54.4%   CAGR +12.8%
max drawdown  11.3%     Calmar 1.13
Sharpe 0.94   Sortino 0.67   t-stat 2.20
fees 11.70    (1% of gross profit)
P(edge > 0)   0.99
```

### Why this is more than a lucky window

**Cost-insensitive.** Full sample, sweeping the fee tier:

| | PF | expectancy | net | max DD |
|---|---|---|---|---|
| maker 2bps | 2.25 | +0.602R | +837 | 12.2% |
| taker 5bps | 2.22 | +0.594R | +821 | 12.3% |
| taker 7.5bps | 2.20 | +0.588R | +807 | 12.5% |
| taker 10bps | 2.18 | +0.581R | +793 | 12.6% |

Doubling the fee costs 3% of the edge. On the legacy 15m strategy the same
change is decisive. This is structural, not a fit.

**A plateau, not a spike.** Full sample, one parameter at a time:

```
entry_n         20:PF1.84  30:PF1.88  48:PF2.22  72:PF2.32
sl_atr          2.0:PF2.18  3.0:PF2.22
trail_atr       2.0:PF1.45  3.0:PF1.73  4.5:PF2.22
trail_start_r   0.0:PF2.22  0.5:PF2.28
adx_min         0:PF2.22    18:PF2.40
ema_filter      0:PF2.22    200:PF2.21
```

Every setting of every parameter is profitable. Overfitted results are sharp
peaks surrounded by cliffs; this is a table-top.

**It beats random entries.** Holding the exit logic, the universe, the sizing
and the trade count fixed, and shuffling the entry bars at random 20 times:

```
real entries    PF 2.22
random entries  PF mean 1.11, p5 0.99, p95 1.23
-> real beats 100% of controls
```

Random entries with the same trailing exit produce PF ≈ 1.11 — that is the
drift and the exit structure. The entry timing adds the rest.

### What is wrong with it

- **125 out-of-sample trades in 3.6 years.** The confidence interval is wide
  (PF 1.15–3.13) because the sample is small. 0.7 trades per week.
- **Shorts lose.** Both-sides: LONG PF 2.76, SHORT PF 0.56. Long-only is not a
  tuned choice — shorts were negative in essentially every test — but it means
  this is a long-biased strategy in an asset class that has risen over the window.
- **It does not work in bear markets.** Both-sides version: bull PF 3.09,
  chop 1.74, **bear 0.28**. The BTC alignment filter mostly keeps it flat in
  bear regimes rather than making it profitable there.
- **The last 18 months are flat.** By year: 2023 +204, 2024 +387, 2025 −13,
  2026 (7 trades) −49.
- **It missed the significance bar I set before running it** (t > 2.5, PF CI
  lower bound > 1.10, positive in ≥ 2 regimes). It cleared two of three; t was
  2.20–2.35 against a required 2.50.

---

## 5. What to do

**Do not put money on this.** Nothing here has cleared a bar that would justify
it. What has been established is narrower and still useful: the current strategy
loses with statistical certainty and should be turned off, and the direction
that survives testing is fewer, longer, trend-aligned trades on higher
timeframes — the opposite of the current design in every respect.

**Paper trade the 8h long-only configuration as data collection**, with the
understanding that one or two weeks produces roughly one trade and settles
nothing. At 0.7 trades/week, 100 trades takes about three years of live paper
trading. The way to shorten that is not to trade more often — that reintroduces
the fee problem — but to widen the universe. `metrics.py` refuses to declare an
edge below 100 closed trades for this reason.

**If you want the ML the repo is named after**, the sound construction is
triple-barrier labelling (did price reach +2R before −1R within N bars?) with a
meta-labelling classifier deciding which breakout signals to take, trained under
purged and embargoed cross-validation so overlapping trades cannot leak between
folds. That is a filter on an edge that already exists — it is not a substitute
for one, and applying it to the 15m strategy above would only learn to reproduce
a negative expectancy more efficiently.

---

## 6. Reproducing this

```bash
cd research
python okx_data.py 4h 2022-01-01 BTC,ETH,SOL,LINK,XRP,DOGE,AVAX,NEAR,APT,SUI
python test_lookahead.py     # must print ALL CLEAN before anything else means anything
python run_legacy15.py       # section 2
python run_final.py          # section 3
python run_best_lo.py        # section 4
python run_tiers.py          # market-cap tier hypothesis
```

Raw outputs are in `results/`: per-trade CSVs, fold-by-fold parameter choices,
and equity curves for every run quoted above.

---

## 7. The market-cap tier hypothesis

Tested directly: does splitting the universe into large / mid / low-cap tiers,
and fitting a **separate strategy family and thresholds to each**, beat one
strategy fitted across everything? 18 assets, 8-hour bars, 12 walk-forward
folds, judged only out of sample. Each tier could choose freely among four
families (Donchian breakout, squeeze breakout, mean reversion, trend pullback),
its own stop/target geometry, direction, and whether to use the BTC filter.

- **large**: BTC, ETH, BNB, XRP, TRX, LTC, ADA
- **mid**: SOL, LINK, AVAX, NEAR, APT, SUI
- **low**: DOGE, PEPE, SHIB, WIF, HYPE

| design | OOS trades | win% | PF | PF CI lower | expectancy | max DD | t |
|---|---|---|---|---|---|---|---|
| pooled — one strategy | 238 | 38.7 | 0.92 | 0.63 | −0.040R | 19.3% | −0.44 |
| **per-tier — three strategies** | **397** | **39.5** | **1.26** | **0.90** | **+0.142R** | **13.6%** | **1.43** |

**Tiering beat pooling out of sample.** That is a real result and it supports
the hypothesis directionally. Four qualifications matter:

**The low-cap tier is the only losing one.**

| tier | n | win% | PF | expectancy | max DD | t |
|---|---|---|---|---|---|---|
| large | 133 | 39.1 | 1.27 | +0.147R | 19.0% | 0.90 |
| mid | 196 | 41.8 | 1.41 | +0.208R | 12.1% | 1.34 |
| **low** | 68 | 33.8 | **0.89** | **−0.060R** | 11.9% | −0.32 |

The premise was that low-cap volatility needs its own treatment. Given complete
freedom to pick its own strategy, the low-cap tier still could not find one.

**Neither design is significant.** Per-tier reaches t = 1.43 with a profit-factor
confidence interval of [0.90, 1.71] — the lower bound is below 1. It is not
distinguishable from zero edge.

**The family choices are unstable, not tier-specific.** Across 12 folds:

```
large  donchian 5   squeeze 4   pullback 3     filter: none 7 / align 5
mid    donchian 7   pullback 3  squeeze 2      filter: none 7 / align 5
low    pullback 5   squeeze 5   donchian 2     filter: align 6 / none 6
```

There is a lean — low caps prefer pullback and squeeze, mid caps prefer breakout
— but every tier changes its mind repeatedly between adjacent folds. A genuine
structural difference would persist; this reads as the optimiser chasing noise.

**Two symbols carry the result.** BNB contributes +192 on 15 trades (PF 8.55)
and SOL +208 on 34. Remove those two and the combined book is roughly flat. That
is concentration, not a tier effect.

### The deeper lesson from this test

The pooled run here (PF 0.92) is *worse* than the restricted single strategy in
section 4 (PF 1.97) — on more assets and more data. The difference is search
space: section 4 searched one family with seven parameters; this pooled run
searched four families with sixteen. **More freedom produced worse out-of-sample
results.** That is the central discipline in this kind of work, and it is the
argument against a three-tier, three-decision-tree design: each additional node
threshold is another way to fit the past.

### What to keep from the idea

The tiers differ in **volatility**, and that difference is real and worth
respecting — but ATR-based stops and risk-based sizing already adapt to it
automatically. A PEPE position sized to risk 0.75% of equity against a 3-ATR
stop is already much smaller in notional terms than a BTC position, without any
tier logic. What the evidence does *not* support is three different decision
trees with independently fitted thresholds.

The defensible middle path, in order of evidence:

1. Run **one** strategy with **volatility-scaled** parameters.
2. Allow **per-tier risk multipliers** (one number per tier, not a tree) if
   out-of-sample testing justifies them.
3. Only then consider a per-tier family split — and require it to clear the
   same significance bar as everything else.

---

## 8. The universe was the binding constraint

Section 4's result (PF 1.97, Sharpe 0.94, 125 trades) had a per-trade edge of
+0.54R — strong — and a trade count too low to compound. The diagnosis was
wrong the first time: the problem was never the signal, it was that eleven
correlated pairs generate too few independent bets.

OKX lists **443 USDT perpetuals**. Re-running the identical strategy, identical
walk-forward, identical costs on the **39 with at least a year of history**:

| universe / concurrency | OOS trades | /wk | win% | PF | PF CI | expectancy | Sharpe | max DD | Calmar | CAGR | t |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 11 pairs, 5 open, 0.75% risk | 117 | 0.6 | 42.7 | 1.97 | [1.14, 3.26] | +0.510R | 0.87 | 14.9% | 0.76 | +11.3% | 2.19 |
| **39 perps, 8 open, 0.50% risk** | **248** | **1.3** | **44.4** | **2.13** | **[1.41, 3.11]** | **+0.606R** | **1.26** | **16.0%** | **1.24** | **+19.8%** | **3.13** |
| 39 perps, 14 open, 0.35% risk | 315 | 1.7 | 39.4 | 1.30 | [0.94, 1.77] | +0.159R | 0.36 | 15.1% | 0.26 | +3.9% | 1.39 |
| 39 perps, 20 open, 0.25% risk | 387 | 2.1 | 38.0 | 1.40 | [1.01, 1.88] | +0.211R | 0.47 | 11.4% | 0.41 | +4.6% | 1.82 |

**This is the first configuration to clear the bar set in section 3** — t = 3.13
against 2.50 required, profit-factor CI lower bound 1.41 against 1.10, positive
expectancy in both regimes it trades (bull PF 2.21, chop 1.85; the BTC gate
keeps it flat in bear).

Note that concurrency is not monotone. Going from 8 to 14 simultaneous
positions *halves* the profit factor. Past eight, the bot is accepting its ninth
and tenth best signal, and those are worse than the eighth. Diversification
helps until it becomes dilution.

### Risk is a dial, not a result

Same signals, same walk-forward, only risk-per-trade changed:

| risk/trade | CAGR | max DD | Sharpe | Calmar |
|---|---|---|---|---|
| 0.25% | +10.1% | 8.6% | 1.25 | 1.17 |
| 0.50% | +19.8% | 16.0% | 1.26 | 1.24 |
| 0.75% | +29.2% | 22.4% | 1.27 | 1.30 |
| 1.00% | +38.4% | 28.2% | 1.28 | 1.36 |

Sharpe is flat because leverage cannot manufacture edge. Quoting "13% CAGR" as
if it were a property of the strategy was a presentation error on my part — it
is a property of the risk setting.

### The refit is part of the strategy

Freezing one parameter set for the whole sample gives **PF 1.48–1.57, Sharpe
0.70–0.77, CAGR +8.7% to +14.5%** at 0.5% risk. The walk-forward's PF 2.13 and
Sharpe 1.26 come from an *adaptive* system that re-chose parameters on the
trailing 365 days every 110 days and then left them alone.

So the deliverable is not a parameter set, it is a procedure. `bot/refit.py`
implements it: search the same grid on trailing data, score on the
drawdown-penalised t-statistic, write `params.json` only if the winner clears a
minimum bar, and report the change to Telegram. Run it quarterly. If a refit
comes back empty, that is information — the strategy has stopped working on
recent data, and the bot should be paused rather than re-tuned harder.

### What is still wrong with it

- **248 trades.** Better than 117, still not many.
- **2024 carries it.** +581 of +724 net. 2023 +96, 2025 +52, 2026 −5 on 8 trades.
- **Two folds produced zero trades** and two more were negative.
- **Concentration.** XLM contributes +217 on 7 trades, KAITO +41 on one.
- **Non-crypto diversification is unavailable historically.** OKX now lists
  gold, silver, NVDA, MSTR and MU perpetuals, which would genuinely break the
  correlation — but only XAU has even a year of data. Revisit in 2027.

### Calibration against the industry

Crypto hedge funds, 2025: **algorithmic/quant strategies returned +0.4% average,
+3.2% median**; only 37% of funds were positive; 409 funds are dead; median
Sharpe since inception 1.17; average worst 12-month drawdown −26%
([Crypto Fund Research](https://cryptofundresearch.com/crypto-hedge-fund-performance/)).

Sharpe 1.26 with a 16% drawdown is therefore around the professional median with
a better drawdown — achieved without their fee tiers, colocation or multi-venue
credit. It is not a 70%-win-rate machine and no such thing exists. Published
research consensus for a competent retail systematic trader is **10–25%
annualised at Sharpe 0.5–1.5**; anything above 50%/yr or Sharpe > 3 at this
scale implies leverage on an unmeasured tail.
