# Phase 2 — The Research Gate (plan, pre-registered)

**Status:** planned · **Prerequisite:** Phase 0 done ✅ · Phase 1 (Freqtrade port) in progress next
**Purpose:** prove — or kill — each engine's edge with simulation that cannot lie, *before* any real money. Everything below is pre-registered: the acceptance thresholds are written down NOW, before we see results. Changing them after the fact is the exact bias Phase 0 removed.

---

## A. What Phase 2 is NOT

- Not "run a backtest and screenshot the green number." Anyone can produce a green backtest; the job of Phase 2 is to make producing a *fake* green backtest structurally impossible.
- Not a guarantee of profits. A clean pass means "edge survived honest simulation"; the live ramp (Gate G4) is still mandatory.

## B. How professional frameworks behave (the standard Phase 2 inherits)

| Behavior | Retail pattern (what we had) | Professional pattern (what Phase 1/2 adopt) |
|---|---|---|
| **Event model** | sleep(60) polling loop; risk checked when the loop gets around to it | Event-driven: orders/fills/candles/funding are events; risk gates fire synchronously with events |
| **Backtest↔live gap** | Different logic per environment; results don't transfer | **One identical codepath** (Freqtrade strategy / Nautilus actor); only the venue adapter changes |
| **Order lifecycle** | A row in SQLite; bot crash = naked position until restart | State machines with recovery + exchange-side protection (stop orders resting *on the venue*) |
| **Fills** | Fill at signal price; stops fill at exact stop price | Intrabar high/low simulation, pessimistic tie-breaks, spread/slippage models, funding accrual |
| **Cost accounting** | Flat fee knob added late | Fees tier-accurate per side; funding from exchange schedules; costs booked per trade, always |
| **Risk ownership** | Signals decide; risk is an afterthought | Independent risk engine between signal and order: caps, cluster exposure, kill switches, cooldowns |
| **Research rigor** | Tune on the whole sample; delete losers; report best | Walk-forward OOS, parameter *plateaus* not peaks, lookahead audits, Deflated Sharpe on trial count |
| **Ops** | Restarts wipe context | Crash-safe persistence, reconciliation vs venue on boot, heartbeat alerts, drawdown auto-pause |

Phase 2's rules simply enforce the right-hand column on our two engines.

---

## C. Workstreams (executed in order)

### W1 — Data foundation (before any backtest is believed)
- Download **2+ years** of 5m and 15m OHLCV for all 12 pairs (Freqtrade `download-data`), covering at minimum: a bear regime, a chop regime, a bull trend regime. A single-regime sample proves nothing.
- Log data quality: gaps, zero-volume candles, outliers. Note per-pair funding-rate history separately (OKX public funding API) so W2 can book *actual* historical funding instead of the flat 0.01% Phase 0 assumption.
- Freeze the dataset (immutable copy). Every number reported in Phase 2 references this frozen snapshot. Retuning on a mutated dataset voids prior results.

### W2 — Truth run (baseline backtest, no tuning)
- Run the **exact** Phase 0/1 strategy code, factory defaults, all 12 pairs, full window, with taker fees + historical funding + slippage buffer.
- Report per engine, per pair, per side: trades, WR, PF, avg R, expectancy (R/trade after costs), max DD, fees+funding share of gross PnL.
- **No parameter changes at this step.** This run's only question: *is there anything here at all?*

### W3 — Walk-forward validation (the engine of truth)
- Rolling protocol: optimize on 12 months → validate on next 3 months → step forward 3 months → repeat to end of window. (Expanding-window variant also run as a cross-check.)
- Concatenate ONLY the out-of-sample legs into one equity curve. That curve is the strategy's résumé; the in-sample curves are marketing material and we don't read them.
- Purged boundaries + embargo at each fold edge (labels/parameters must not peek across the fold — López de Prado discipline).
- Report the full **distribution** of OOS leg results (median, worst decile), not just the mean.

### W4 — Robustness, not optimization
- Hyperopt with an overfit-penalized objective (Sortino × drawdown-penalty), per category — hunting for a **plateau**: chosen parameters must be profitable with every neighbor perturbed ±10–25%. A needle-peak parameter set is rejected even if its Sharpe is higher.
- Count every trial. Final check: **Deflated Sharpe Ratio** against the number of trials run — if DSR shows the result is consistent with luck-given-that-many-attempts, the strategy is rejected regardless of its headline number.

### W5 — Bias audits (lookahead-analysis and friends)
- Freqtrade `lookahead-analysis` and `recursive-analysis` on both engines, full window.
- Manual code review gate: signals computed only on closed candles; no use of current-bar close before it closes; dedupe can't skip-and-refire differently in backtest vs live; **the 1m intrabar exit semantics and funding accrual in the paper engine must match the backtester 1:1** (this is the parity check for everything Phase 0 built).
- Any bias found → fix → re-run W2–W4 from scratch.

### W6 — Regime & correlation slicing (feeds Phase 3)
- Tag every OOS trade with: ADX bucket, BTC macro trend state (above/below 200d EMA), realized-vol regime (efficiency-ratio quartile), day-of-week, funding sign.
- Output: conditional performance table per engine → **the evidence base for regime gating** (which sleeves run in which regime — this becomes a Phase 3 config, not vibes).
- Correlation analysis: the 12 pairs are one beta cluster. Compute max simultaneous same-direction exposure historically, then set the concrete rule (e.g., max 2–3 same-direction alt positions, or a portfolio-level "cluster VaR" cap). Numbers from data, not from me.

### W7 — Paper-vs-backtest reconciliation
- Compare the clean **Phase 0 paper cohort** (post-reset, honest costs) against the backtest re-run over the same calendar weeks.
- Tolerance: PF within ±20%, WR within ±5pp. Inside tolerance → simulator validated, W4 numbers can be trusted. Outside → find the modeling gap (fills? funding? latency?) **before** believing anything else.

---

## D. The Go/No-Go Gates (pre-registered, pass/fail)

| Gate | Threshold | Failure consequence |
|---|---|---|
| **G1 Live-sample (current)** | Phase 0 paper cohort: PF ≥ 1.3 after costs, ≥ 60 trades, DD ≤ 15% | Engine goes back to W1 with that pair/engine sliced out |
| **G2 Baseline (W2)** | PF ≥ 1.3 after costs on the 2y frozen set, ≥ 200 trades per engine | Engine is killed or redesigned — no tuning allowed to "rescue" a G2 fail |
| **G3 Walk-forward (W3)** | Median OOS leg PF ≥ 1.25 AND ≥ 70% of OOS legs profitable | Strategy rejected; a robust edge must survive time-shuffling |
| **G4 Robustness (W4)** | Plateau: ≥ 80% of ±10–25% perturbed neighbors profitable; DSR passes at trial count | Parameters rejected; mean-revert to factory defaults or kill |
| **G5 Bias audits (W5)** | lookahead/recursive analysis clean; paper↔backtest parity confirmed | Fix + full re-run |
| **G6 Reconciliation (W7)** | \|paper − backtest\| within tolerance | Modeling gap fixed first; no live without G6 |

**Only after G1–G6:** micro-live ramp — $300–500, 1× leverage, 4 weeks, three-way reconciliation (live vs paper vs backtest). Scale gradually (2×, then size) only while live tracking stays inside tolerance. First breach of tolerance = automatic de-risk + post-mortem.

## E. Deliverables produced by Phase 2

- `research/frozen_dataset/` + checksum manifest
- `research/W2_baseline_report.md`, `W3_walkforward_report.md`, `W4_robustness_report.md`
- `research/regime_slicing_table.csv` (the Phase 3 gating evidence)
- Gate verdict sheet: one line per engine — PASS/KILL, signed against the pre-registered thresholds above

## F. Honest expectations

The audit's prior (fees ate ~63% of gross PnL; PF 1.21 raw) says the **CCI_BB scalper on taker fees at 5m is the most likely G2 casualty** — its friction budget is the worst in the book. The SMC engine on 15m has a better fee-to-signal geometry and is the more probable survivor. If both fail cleanly, that is a *successful* Phase 2: it cost us weeks instead of a live account, and Phase 3's structural edges (funding carry, maker-side mean reversion) become the main line. The gates protect capital; they don't protect feelings.
