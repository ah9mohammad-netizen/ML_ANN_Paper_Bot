# Forensic Code Audit + Professional Framework Research
**Date:** 2026-08-22 · **Scope:** full repo read (all 9 Python modules, configs, docs) + external research sweep (GitHub, Discord/Reddit trading communities, quant literature)
**Reviewer lens:** professional day trader / quant desk — judging this the way we'd judge a junior PM's book before giving it more capital.

---

# PART 1 — WHAT YOUR BOT ACTUALLY DOES (ground truth from code, not docs)

## 1.1 System map

```
┌────────────────────────────────────────────────────────────────┐
│ main.py — single synchronous while-True loop, sleeps 60s        │
│                                                                 │
│  1. TelegramUI.poll_once()   (commands: /stats /open /reset…)   │
│  2. PaperEngine.cycle()                                          │
│     a. update_positions() — for each OPEN position:             │
│        fetch price (last CONFIRMED candle close) → TP/SL/TIME   │
│     b. process_pair() for each of 12 pairs:                     │
│        fetch_okx(pair, tf) → StrategyBrain.latest_signal()      │
│        → dedupe → size → risk gates → sqlite                    │
│  3. sleep(60)                                                    │
└────────────────────────────────────────────────────────────────┘

Data:  OKX /market/history-candles (REST pagination) + Binance failover
Store: SQLite (signals, positions, state) on Railway volume
Deploy: Railway worker, python main.py, Telegram for control
```

Strategy routing (`strategy.py`, `latest_signal`):

| Category (intended) | Pairs | Engine TF | Strategy | Filters | Exits |
|---|---|---|---|---|---|
| 1 Majors | BTC, ETH | 15m | SMC sweep-reclaim: low < 36-bar swing low, close back above (mirror for shorts) | Volume ≥ 1.25×SMA20 · EMA-200 trend | SL 1.5×ATR / TP 2.0×ATR (≈1.33R) · 24h time-stop |
| 2 Mid-Vol | SOL, AVAX, BNB, LINK, NEAR | 15m | same SMC | same | same |
| 3 High-Vol | HYPE, PEPE, WIF, FET | 5m | CCI+BB mean reversion: close beyond BB(20, ~2–2.5σ) + CCI beyond ±130–150 | EMA-100 trend | SL 1.5×ATR / TP 2.0×ATR · 12h time-stop |

Sizing (`paper_engine.calc_size`) is **textbook fixed-fractional**: risk = equity × 1% ÷ stop distance; notional capped at 2.5× equity; margin capped at 15% equity/position, 90% total; 10× leverage. That part is genuinely fine.

## 1.2 What the docs claim vs. what the code does

Your `Quantitive_Audit.md` and `Research_Report.md` describe a system that **does not exist in this codebase**. Forensic diff:

| Claimed in docs / implied by repo name | Reality in code (verified by grep) |
|---|---|
| "ML/ANN" paper bot, ANN model v3, model artifacts un-ignored in `.gitignore` | **Zero ML.** `sklearn`/`joblib` are in `requirements.txt` and imported nowhere. `probability: 0.85` / `0.82` are hardcoded constants. The ANN artifacts directory does not exist. `sigmoid`, `clamp`, `hma`, `mfi`, `cmf`, `efficiency`, `chop_proxy`, `cross_above/below` are dead code from an earlier iteration (~40% of `strategy.py` is dead weight). |
| "ADX Regime-Adaptive Sizing Engine" (Regime A: 1.2/2.5×ATR, Regime B: 1.5/2.0×ATR) | `adx()` is defined and **never called once**. Every trade gets static 1.5/2.0 ATR exits. This is the single biggest doc-vs-code lie. |
| "50–60% Order Block pullback entries" (`pullback_pct=60/30`) | `pullback_pct` is accepted as an argument and **never used inside the function**. Entries are at market close of the trigger candle. |
| "Market Structure Shift layer" (close above max high of last 3 bars) | Not implemented. |
| "Heikin Ashi 1m confirmation" for Category 3 | Not implemented. |
| "FVG (fair value gap) check" | Implemented but `fvg_required=False` at every call site — dead. |
| Telegram "REGIME / CATEGORY AUDIT COHORT" stats | Strategy never writes `regime` or `category` into meta — the Telegram cohort report silently defaults everything to Trending / Category 3. Dead analytics. |
| "Backtest: 83.67% WR, PF 4.75–5.82" | Post-hoc slicing (see §1.4 — red flag #2). The honest raw number in your own audit is **68.7% WR, PF 1.21, +6.08 USDT on 99 trades with 10.39 USDT fees**. |

## 1.3 Critical defects — ranked by damage

**🔴 #1 — Three of your 12 pairs run an inverted risk/reward fallback.**
`config.py` ships `PAIRS = BTC,ETH,SOL,LINK,NEAR,SUI,APT,HYPE,PEPE,WIF,FET,DOGE`. But `latest_signal` categorizes none of **SUI, APT, DOGE** → they fall to the `else` branch → CCI_BB scalper on **15m candles** (wrong timeframe), and since they're absent from `SCALPER_ASSET_CONFIG` they get the fallback `{'tp_atr': 1.5, 'sl_atr': 3.0}` — **risking 2 units to make 1, with no trend filter.** This is a live production bug: a quarter of your universe trades a strategy you never designed, at negative expectancy by construction.

**🔴 #2 — The PF 4.75–5.82 "certified" number is data-mined.**
The audit reaches it by deleting losing pairs (BNB, AVAX, PEPE, ETH) *after* seeing results, multiplying hypothetical regime win rates. Removing losers post-hoc is the textbook definition of **selection bias / backtest overfitting**. With ~16 parameters per asset-direction (stdev, CCI threshold, trend filter…) tuned on a sample of 99 trades, the probability that the tuned configuration was the best *by chance* is extremely high. López de Prado's Deflated Sharpe Ratio exists precisely to punish this. A live desk would never size up on that math.

**🔴 #3 — TP/SL is checked against stale confirmed candle closes, not live price.**
`fetch_okx` filters to `confirm != '0'` (good for signals — no lookahead), but the same function is used for `quote()`. On 15m pairs your exit price can be **up to 15 minutes stale**; intrabar stop-outs that would happen in reality are silently missed, and when a stop is detected you "fill" at the exact stop price. Net effect: **paper results are systematically flattered** versus live fills. A pro engine marks positions on tick/trade data or at minimum the ticker endpoint.

**🔴 #4 — 10× leverage with zero funding-cost modeling.**
Holds average 3.5h and up to 24h. Perp funding (±0.01%/8h baseline, frequently 3–5× that on meme majors like PEPE/WIF/HYPE) is entirely unmodeled — applies to notional, i.e. can exceed your taker fee over multi-day holds. Combined with **no liquidation modeling**, the paper engine certifies risk profiles that don't exist.

**🟠 #5 — Fee drag is eating the edge, and your own numbers prove it.**
Phase-5 synthesis: gross edge is +16.5 USDT, fees 10.39 USDT → **~63% of gross PnL paid away**. Quick expectancy math at 15m: 1.5×ATR stop on BTC ≈ 0.2–0.45% → your 0.12% round-trip cost = **0.27–0.6R per trade** just to play. Gross EV at 68% WR and 1.33R payoff ≈ +0.58R; minus costs → **+0.0 to +0.28R**. That is a hair-thin edge, one regime shift from negative.

**🟠 #6 — The loop architecture delays risk checks.**
One synchronous thread: scan 12 pairs (up to ~10 paginated HTTP calls each, 8s timeouts × 4 endpoints each) *before* the next cycle's position update. Worst case, exits are evaluated minutes late. Positions and quotes should never wait behind signal scanning.

**🟡 #7 — Housekeeping:** `/reset` doesn't clear `positions` (contaminates stats across phases — which is exactly how the "99-trade synthesis" got pooled); `analyze_trades.py` hardcodes starting balance 100 vs. config 1000 (drawdown % off by 10×); `test_strategy.py` validates nothing (forces a 20σ synthetic spike); `same_pair_lock` still lets both directions exist via different setups; dedupe key uses `signal_time` = candle close time, so identical re-signals each new candle re-fire.

**What is genuinely good (keep it):** confirmed-candles-only signal generation (no candle-stream lookahead — many hobby bots fail here), fixed-fractional sizing with margin/notional caps, time-barrier exits, multi-endpoint data failover, Telegram ops loop, DB persistence on a mounted volume. The skeleton is a competent hobbyist paper bot. It's the *trading claims and the silent fallbacks* that are dangerous.

---

# PART 2 — WHAT PROFESSIONAL-GRADE BOTS ACTUALLY LOOK LIKE (research sweep)

Sources: GitHub (freqtrade ⭐~48k, hummingbot ⭐~18k, nautilus_trader ⭐~24k, jesse, LEAN), framework docs, r/algotrading & r/algotradingcrypto consensus threads, DeepWiki architecture analyses, López de Prado validation literature.

## 2.1 The framework landscape — who the serious players are

| Framework | What it IS | Core architecture lesson for you |
|---|---|---|
| **Freqtrade** (+ FreqAI) | The de-facto open-source crypto bot. SQLite persistence, Telegram control, dry-run mode — i.e., **your bot's feature set, but with 400 contributors and 6k forks** | Ships the tooling your repo lacks: `backtesting`, `hyperopt` (parameter search with anti-overfit loss functions), **`lookahead-analysis`** (systematically hunts the bias class found in your engine), `recursive-analysis`, edge positioning, FreqAI (self-retraining ML with backtest-emulated retraining). Your instinct ("ML + categorized strategies + Telegram on a cron") is *literally Freqtrade's design*. |
| **Hummingbot V2** | Execution-first framework: **Controllers** (strategy logic) orchestrate **Executors** (Position/DCA/Grid/TWAP/XEMM arbitrage) across 140+ CEX/DEX | Pros separate *decision* from *execution*. Contains the archetypes of where crypto edge actually lives: market making, cross-exchange market making, **spot-perp funding arbitrage**, stat-arb (v2.6 added a stat-arb controller with live Z-scores, hedge ratios, funding data). |
| **NautilusTrader** | Institutional grade. Rust core, event-driven, nanosecond resolution | The gold-standard principle: **backtest/live parity — one identical codepath**, order-book-aware fills, latency modeling, deterministic event ordering. The exact opposite of a paper engine that can't reproduce live fills. |
| **Jesse / LEAN / vectorbt / hftbacktest** | Clean research workflow / multi-asset ecosystem / ultra-fast parameter sweeps / order-book HFT simulation | Research speed and simulation fidelity are separate problems — pros use different tools for each. |

## 2.2 The 7 structural differences between your bot and a professional stack

1. **Research-first, trade-second loop.** Pros run `strategy idea → backtest on years of multi-regime data → walk-forward validation → hyperopt with overfit-penalized loss (e.g., Sortino, max-drawdown-weighted) → lookahead-analysis → dry-run 4–8 weeks → tiny live size → scale`. Your process inverted it: ran live-paper first, then wrote the audit around whatever happened.
2. **Validation discipline.** Walk-forward windows, purged cross-validation + embargo (López de Prado), Combinatorial Purged CV, **Deflated Sharpe Ratio** to punish multi-trial selection, pre-registered hypotheses. Never delete losing pairs after the fact *and then claim the survivors' stats*.
3. **Execution realism.** Tick/order-book fill modeling, maker-vs-taker fee tiers, spread + depth-based slippage, **funding accrual**, liquidation modeling, partial fills. If paper can't reproduce live fills, paper PnL is fiction.
4. **Risk engine separated from signal.** Portfolio-level kill switches: daily loss halt, max correlated exposure (BTC-direction cluster), per-strategy capital sleeves, volatility-scaled sizing (ATR/Kelly-fraction), circuit breakers on API failures.
5. **Regime awareness is real code, not a doc chapter.** Pros classify regime (ADX/vol/efficiency-ratio — you already have these helpers dead in your file) and **enable/disable strategy sleeves and re-scale size** accordingly — Hummingbot controllers literally do this dynamically.
6. **Portfolio of uncorrelated edges.** Nobody at a desk runs one mean-reversion scalper on 12 correlated alts. The consensus from every professional/community source on where crypto bot edge actually survives in 2025–26: **(a) funding-rate / spot-perp carry (delta-neutral), (b) market making / spread capture, (c) cross-exchange & structural arbitrage** — all with real, structural PnL drivers — followed by (d) momentum/stat-arb with strict validation. Pure indicator-soup scalping on taker fees is widely documented (incl. in your own audit phases) as the strategy class where **fees consume the edge**.
7. **Ops as a first-class citizen.** WebSocket streams (not REST polling), async risk loop independent of signal loop, monitored heartbeats, restart-safe state, database migrations, alerting on drawdown breach — Railway + SQLite + 60s sleep loop is fine for paper, it is not an ops stack.

## 2.3 Recommended migration path for THIS repo (concrete, ordered)

**Phase 0 — Honesty pass (this week, cheap):**
1. Fix the SUI/APT/DOGE fallback bug (route them into a designed category or drop them).
2. Kill or implement every dead feature; delete half of `strategy.py`. Make docs match code — a repo whose docs describe a different bot is self-deception at machine speed.
3. Quote exits from the ticker endpoint / unconfirmed last price; model intrabar high/low for SL/TP in analysis.
4. Add funding accrual + maker/taker fee realism to the paper engine; re-run the historical 99 trades → watch PF 1.21 tell the truth.
5. Fix `/reset` to actually clear positions; fix the `analyze_trades.py` hardcoded 100-USDT base.

**Phase 1 — Stand on a real framework (2–4 weeks):**
Port your two setups (SMC sweep-reclaim 15m, CCI+BB 5m) into **Freqtrade** strategies. You keep: Telegram, SQLite, Railway, dry-run. You gain for free: years of backtest data tooling, `hyperopt`, `lookahead-analysis`, proper stoploss-on-exchange semantics, FreqAI when you actually want the "ML/ANN" the repo is named after. Do not rewrite your own Nautilus — you're not at the stage where its complexity pays.
Alternative if you go HFT/market-making later: Hummingbot controllers.

**Phase 2 — Research gate before any live money:**
- ≥ 2 years multi-regime backtest (2022 bear, 2023 chop, 2024–25 trend) on the *exact* live strategy code.
- Walk-forward: re-tune params on window N, validate on N+1, roll. Report the distribution of OOS results, not the best one.
- Report honestly: expectancy in R after costs, PF, max DD, **funding-adjusted**, and a Deflated-Sharpe sanity check on your tuning trial count.
- Only then: minimum live size, 1× → scale.

**Phase 3 — Where to hunt for a *structural* edge (the part that resembles my book):**
Your mean-reversion engine is a *filter*, not an edge. The accounts that compound year after year are running sleeves of: funding carry on majors when |funding| annualized > ~10–20%, grid/market-making on high-vol pairs with maker rebates, and momentum only in confirmed high-ADX regimes — each sleeve small, uncorrelated, regime-gated. Your bot's true asset is its clean categorization + risk plumbing; point it at edges that don't depend on beating 0.12%/trade friction with a 1.33R coin flip.

---

## Bottom line

Your skeleton (loop, storage, sizing caps, Telegram ops) is solid junior-quant work. But the repo currently describes a bot that doesn't exist, runs three pairs on an accidental negative-expectancy fallback, validates exits on 15-minute-stale prices, ignores funding at 10× leverage, and certifies a PF of ~4.75 via post-hoc pair deletion — while the honest measured edge (PF 1.21 before funding) is one bad week from zero. None of this is fatal; all of it is fixable with Phase 0 + a Freqtrade port. The pros don't have secret indicators — they have **honest simulation, validation discipline, separated risk engines, and strategies whose PnL doesn't vanish after fees**. Build those four things and you're playing the actual game.
