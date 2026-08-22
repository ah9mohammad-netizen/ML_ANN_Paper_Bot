# Crypto Paper Trading Bot (OKX/Binance data · Railway worker · Telegram ops)

A single-process **paper trading** bot for crypto perpetual-style strategies, backed by
SQLite, controlled over Telegram. No exchange keys required — it never places real orders.

📐 **Research harness:** `research/` contains the Phase 2 gate (frozen multi-regime
data, event-realistic backtester with funding costs, lookahead + recursive bias audits,
regime slicing). Run `bash run_research_gate.sh` anywhere with exchange access;
see `research/README.md`. Gate thresholds are pre-registered in `PHASE2_RESEARCH_GATE.md`.

> **Honesty note (Phase 0, 2026-08-22):** This repo was previously named "ML_ANN" and its
> reports referenced probability models, an ADX regime-sizing engine, order-block pullback
> entries and Heikin-Ashi confirmations. **None of that exists in code.** This build is a
> deliberately truthful one: what is documented below is exactly what runs. There is no ML
> model in the current system. See `FORENSIC_AUDIT_AND_PRO_FRAMEWORK_RESEARCH.md` for the
> full audit and the roadmap to a professional-grade stack.

## Architecture

```
main.py  ── 60s loop ──►  PaperEngine.cycle()
                              │
                              ├─ update_positions()      (risk first, always before entries)
                              │     ├─ 1m candles since entry → intrabar SL/TP check
                              │     │   (pessimistic tie-break: stop wins if both spanned)
                              │     ├─ live ticker mark (fetch_ticker)
                              │     └─ TIME exit after max_hold_hours
                              │
                              └─ process_pair() × 12 pairs
                                    ├─ fetch_okx(tf candles, Binance failover)
                                    ├─ StrategyBrain.latest_signal()  (category routing)
                                    ├─ dedupe (pair|side|setup|candle)
                                    ├─ fixed-fractional sizing + caps
                                    └─ SQLite signals/positions
```

## Strategies (the complete truth)

| Category | Pairs | TF | Setup | Filters | Exits |
|---|---|---|---|---|---|
| SMC | BTC, ETH, SOL, AVAX, BNB, LINK, NEAR, SUI, APT | 15m | Liquidity sweep of the 36-bar swing + reclaim close | Volume ≥ 1.25×SMA20, EMA-200 trend | SL 1.5×ATR / TP 2.0×ATR (≈1.33R), 24h time-stop |
| Scalper | HYPE, PEPE, WIF, FET, DOGE | 5m | Close beyond BB(20, 2–2.5σ) with CCI(14) beyond ±130–150 | EMA-100 trend (per-pair tuned) | SL 1.5×ATR / TP 2.0×ATR, 12h time-stop |

**Uncategorized pairs are rejected, never traded.** Categories/timeframes live in one place
(`strategy.py: CATEGORY_SMC / CATEGORY_SCALPER / timeframe_for_pair`) and are imported by
`paper_engine.py` — they cannot drift apart.

## Risk & cost model

- **Sizing:** fixed-fractional — risk `RISK_PER_TRADE_PCT`% of equity ÷ stop distance; notional
  capped at `MAX_NOTIONAL_PCT`% of equity; margin capped at `MAX_MARGIN_PER_POSITION_PCT`%
  per position and `MAX_TOTAL_MARGIN_PCT`% total. Oversized positions are de-levered by caps.
- **Costs (all subtracted from paper PnL):** taker fee per side (`TAKER_FEE_PCT`, default
  0.04%) + slippage per side (`SLIPPAGE_PCT`, 0.02%) + **perp funding**
  (`FUNDING_RATE_PCT_8H`, default 0.01% of notional per 8h boundary crossed — funding was
  entirely unmodeled before Phase 0).
- **Exits:** checked intrabar on 1m candle high/low since entry, then against the live ticker;
  if one candle spans both SL and TP the **stop is assumed to fill first** (pessimistic).
- **Locks:** one position per pair, max `MAX_OPEN_POSITIONS`,
  `/pause` from Telegram halts new entries.

## Operating it

```bash
pip install -r requirements.txt
python test_strategy.py   # 78-assertion suite: routing, RR sanity, forced setups,
                          # funding math, sizing caps, intrabar exit engine
python main.py            # starts the loop (Telegram optional)
```

Telegram commands: `/stats` `/open` `/recent` `/backup` `/pause` `/resume`
`/reset <amount>` — **wipes all history** and starts a clean stats cohort (previously only
zeroed the balance while old trades kept polluting the win rate).

Railway: `worker: python main.py` (see `railway.json`, `Procfile`). Set `DB_PATH=/data/paper_bot.db`
on a mounted volume. All knobs are env vars — see `config.py` and `.env.example`.

## Known limitations (documented, not hidden)

- Paper fills at exact SL/TP price; real stop-market fills slip, especially on 5m meme pairs.
- Funding is a flat configurable assumption, not the live per-pair rate (OKX funding API
  integration is a Phase 1 candidate).
- No liquidation modeling (10× leverage on a 6-position book would matter live).
- Single-threaded polling: position updates happen once per ~60s cycle.
- Indicator strategies on taker fees are a thin edge: measure expectancy in R **after** all
  modeled costs over a multi-regime sample before believing any positive number.
