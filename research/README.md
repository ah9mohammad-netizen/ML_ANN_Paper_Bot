# research/ — the Phase 2 gate harness

A professional research pipeline built around one rule: **simulation cannot lie
and cannot be self-deceived into profitability.** The strategy code itself is
single-sourced — this harness imports `StrategyBrain` from `strategy.py`, so
what is backtested is by construction what the paper engine trades.

## Layout

| file | role |
|---|---|
| `fetch_data.py` | OKX multi-regime candle + **historical funding-rate** puller → frozen parquet snapshot + sha256 manifest. Runs where exchange APIs are reachable (Railway / laptop / VPS). |
| `backtester.py` | Event-realistic portfolio engine: closed-candle signals, intrabar high/low exits with **pessimistic SL-first tie-break**, taker+slippage fees, funding accrual per 8h interval, time-stops, same risk/sizing/cap math as `paper_engine.py`. Fast vectorized path (`signals_fast.py`) + strict per-bar path. |
| `signals_fast.py` | Vectorized signal layer + `parity_report()` strict-vs-fast agreement checker (measured: 100% on sampled bars). |
| `bias_audits.py` | `lookahead_audit` (entry-shift asymmetry signatures), `recursive_audit` (warmup-invariance), and a **mutant self-test** proving the detector catches a planted perfect-foresight cheater (PF 27.2 → 0.02 collapse → flagged). |
| `regimes.py` | ADX / EMA-200 / efficiency-ratio regime tagging + performance slicing (W6 evidence for Phase 3 gating). |
| `report.py` | Markdown + CSV report writer. |
| `run_research_gate.py` | Orchestrator: `--synthetic` (machinery verification, no network) or `--frozen` (the real gate). |

## Run it

```bash
# anywhere with internet access to OKX (Railway shell, laptop, VPS):
bash run_research_gate.sh                 # 2y all pairs, ~30-60 min total
DAYS=400 bash run_research_gate.sh        # shorter window

# pipeline self-check without network:
bash run_research_gate.sh --synthetic-only
```

Outputs → `research/output/`:
- `REAL_W2_truth_run.md` — baseline, factory params, all costs incl. *actual*
  historical funding; per-pair/side/regime tables; pre-registered G2 verdict
- `REAL_W5_bias_audits.md` — lookahead + recursive audit verdicts
- `trades.csv` — every trade with entry/exit/reason/fees/funding/R-multiple

## The honest control result

On regime-labeled **synthetic noise** (bull/chop/bear generator), the strategy
loses (PF ≈ 0.6, ≈ −0.3R/trade). That's a feature: a simulator that can't
print phantom profits on random data is one you can trust when real data says
something. The audits come out CLEAN on honest code and BIASED on a planted
cheater — the two calibration points that matter.

## Known, accepted semantics gaps vs the live engine

- Entry at signal-candle close (live engine fills within ~60s of it).
- Intrabar exits resolved on the strategy TF candle (live uses 1m candles).
  Pessimistic tie-break in both — bias runs conservative in both.
- `pair in open_pos` same-bar re-entry: live engine can re-enter ~60s later;
  the sim waits one additional candle. Conservative.
