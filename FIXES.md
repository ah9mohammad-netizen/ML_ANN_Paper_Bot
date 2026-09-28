# Paper execution, accounting and risk fixes

**Prepared: 2026-09-28 · Base commit: `df2c357cb37d6bb10c3ad3f8206c32f9a9950d48`**
**Branch: `fix/paper-execution-accounting-risk`**
**Status: locally implemented and offline-tested; NOT deployed, NOT pushed to GitHub.**

The workspace could read the public repository, but `git push --dry-run` failed because no GitHub authentication was available. No repository permissions, access tokens, exchange credentials, remote branches, or production databases were changed.

## Bottom line

Keep this in **paper mode**. The patch fixes reproducible execution/accounting defects; it does not establish a profitable strategy. The old 19.8%/38.4% CAGR figures and other saved research results predate these fixes and must be rerun. They remain in the repository as historical artifacts with prominent warnings, not as validation of this version.

The numerical strategy parameters, configured risk/leverage defaults, pair list and deployment configuration were not tuned. Correcting execution semantics necessarily changes which trades are accepted and their exits.

## What changed

### 1. Exit timestamps and missing history

- One-minute candles are filtered against the position's actual opening timestamp, even on the very first management pass.
- Candles that begin before entry, including the partially overlapping entry minute, cannot trigger exits or move a trailing stop.
- Candles must be closed; repeated management/restarts do not replay already processed candles.
- A missing segment of exit history is flagged in the position and event log and creates an `execution_halt_reason` that blocks new entries. Existing positions continue to be managed against available data; those records are not classified as clean validation trades.
- Time exits are evaluated in chronological order, before later candles can stop out a position that should already have expired.

**Resolution limitation:** if entry occurs at 12:00:30, the candle beginning 12:00 is excluded. Its high/low cannot tell us what happened before versus after 12:00:30. Up to one minute is therefore unmodelled. Tick/sub-minute data or an explicit next-boundary fill model is required to eliminate this limitation; do not mistake this for tick-accurate exchange execution.

### 2. Stops versus liquidations

`bot/execution.py` supplies common, deterministic OHLC exit rules for paper and research:

1. A bar opening beyond the approximate liquidation level is a `LIQ_GAP` at its opening price.
2. Other gaps through stop/target levels fill at the opening price, with configured stop slippage.
3. On an intrabar adverse move, a nearer stop is checked before a farther liquidation threshold.
4. If both stop and target are touched and the opening price does not resolve the order, the default remains pessimistic: stop first.

New positions are rejected with `stop_beyond_liquidation` when their intended stop, including configured stop slippage, lies beyond the modelled liquidation level. The code does **not** silently change the user's leverage or widen/tighten strategy stops.

**Model limitation:** the liquidation formula is an approximation. It is not OKX's mark-price, tiered maintenance-margin, liquidation-fee or cross-margin engine. Intrabar ordering is an explicit assumption, not observed tick data.

### 3. Risk halts

- Refreshing fresh market data no longer clears daily-loss, drawdown, consecutive-loss or manual halts.
- Data-feed halts have a separate state key and can recover automatically without overriding financial risk limits.
- `RiskManager.can_open()` independently checks risk and data-quality halts, so calling the scanner directly cannot bypass them.
- Equity is refreshed between prospective entries in a scan.
- Peak equity is maintained from mark-to-market account equity, rather than treating realized cash as the entire account.
- Missing BTC market-filter data blocks new signals instead of silently disabling the filter.
- `/resume` rechecks the limits and reports when entries remain blocked. It does not erase a missing-execution-history flag.

These are **entry gates**, not guaranteed loss ceilings or automatic portfolio liquidation. Open positions can still lose money after a halt.

### 4. Accounting and persistence

For each closed trade:

```text
net trade P&L = gross P&L - entry fee - exit fee - funding cost
cash change at close = gross P&L - exit fee - funding cost
```

The entry fee is already paid when the position opens; charging it again at close would double-count it.

- Automatic exits and Telegram `/flat` use the same accounting path.
- Opening a position and charging its entry fee are one SQLite transaction.
- Closing a position and updating cash/derived state are one transaction.
- A repeated close is idempotent: it cannot credit/debit cash again.
- Accrued funding costs are included in marked equity before they are settled at close.
- Positive funding is explicitly labelled **paid**, negative funding **received**.
- `/backup` uses SQLite's online backup API so committed WAL contents are included, rather than copying an incomplete main database file.

### 5. Old records and statistics

No historical fills or cash balances are rewritten on startup. No trades are deleted, and no automatic reset is added.

When closed records are read for reporting, `gross - fees - funding` is calculated in memory. Raw stored P&L/R fields remain in the database and are exposed as `raw_pnl`/`raw_r_multiple` in returned records. This repairs the reporting arithmetic where components are available; it **cannot repair an invented historical exit**.

- `/stats` displays the whole sample and flags legacy/missing-history trades as unvalidated.
- `/stats new` excludes those trades without deleting them. This is a post-fix sample, **not certification that the strategy is validated**.
- Drawdown remains explicitly labelled as the **whole-account** equity curve even when trade statistics are filtered.
- The hard-coded 1.3-trades/week waiting-time forecast is removed. An observed rate is displayed only when a measurement interval is available.
- `P(edge>0)` is replaced with **positive bootstrap means**, explaining that it is not a posterior probability of a real edge.
- Expectancy and PF intervals use a deterministic trade bootstrap. All-winning resamples are retained as infinite PF rather than discarded.
- Trade resampling still assumes independence. Correlated crypto positions, regime changes and multiple strategy selection require additional statistical analysis.

### 6. Trailing stops and the research engines

New paper positions use the shared Chandelier formula: running favorable extreme and current strategy-timeframe ATR. Levels update after a closed strategy bar; existing levels can trigger on any subsequent one-minute candle.

Positions already open when upgrading retain their legacy close/entry-ATR trailing mode, so an upgrade does not silently replace that part of an existing position's management plan. Safety checks and corrected accounting still apply. Such positions remain flagged as legacy.

Additional research defects fixed:

- Entry-bar wicks now test a new position's stop. The old loop opened positions only after processing that bar's exits.
- Current-bar closing marks are no longer used before entering at that same bar's open.
- Final forced-close fees are reflected in the final equity point used for CAGR.
- Zero-valued `trail_start_r` stays zero instead of becoming one through `or 1.0`.
- Break-even activation cannot loosen a stop that trailing already tightened.
- The refit simulator delegates to the research engine instead of maintaining another divergent implementation; this also removes its double charging of entry fees.
- Configured research correlation/per-side limits and the refit consecutive-loss gate are enforced.
- Missing supplied historical funding is prominently flagged, including in walk-forward reports and refit metadata.

**Not full execution parity:** paper uses one-minute exits while the historical engine uses its supplied bar timeframe. Fill timing, partial-minute handling, portfolio marks, data windows, funding and operational interruptions still differ. Common formulas do not prove identical performance.

## Funding: what is and is not fixed

This patch fixes the **sign labels, net-P&L treatment and marked-equity treatment** of funding already recorded by the bot.

It does **not** replace the existing paper funding collector with a fully replayable settlement ledger. That collector still polls the current OKX rate around assumed eight-hour settlement times, can miss/reorder costs around exits/restarts, and does not model instruments with different settlement intervals. A follow-up should consume actual settled-rate history, apply it at the correct position/settlement timestamps, persist per-event deduplication and audit coverage. Until then, funding is an estimate, not a verified exchange ledger.

Historical scripts without funding data still run for research, but now warn that costs are omitted. An enabled funding flag is not proof that history was supplied. Full time coverage must also be checked when only partial history is available.

## Offline test results

Executed locally:

```text
python -m pytest -q -W error
63 passed

python -m compileall -q bot research
passed

git diff --check
passed
```

Environment: Python 3.13.14, pandas 2.2.3, NumPy 2.3.5, pytest 9.0.3. Tests use synthetic candles, in-memory/temporary SQLite stores and fake feeds. HTTP requests are blocked by the test fixture; no production service or account was contacted.

Coverage includes first-pass/pre-entry filtering, long/short exit ordering, gaps, stop/target ambiguity, sizing caps, fee/funding reconciliation, transaction rollback, duplicate-close protection, missing history, restart cursor persistence, market/risk gates, retained legacy records, safe WAL backup, bootstrap labels, entry-bar backtest stops, terminal equity, refit accounting and causal Donchian signal checks.

A GitHub Actions workflow is included for Python **3.11 and 3.13**. It has not run on GitHub because these changes have not been pushed. The deployment's Python 3.11 runtime was not available for a local test.

**Not performed:** a full historical rerun, exchange integration tests, real funding replay, an extended forward-paper soak, production database inspection or a comprehensive security audit. Passing these tests does not certify profitability or live-trading readiness.

## Apply the patch

Use a local clone of the original repository, with a clean working tree:

```bash
git fetch origin
git switch -c fix/paper-execution-accounting-risk \
  origin/claude/github-repo-connection-wt3063

git apply --check /path/to/paper-bot-fixes.patch
git apply /path/to/paper-bot-fixes.patch

python -m pip install -r requirements-dev.txt
python -m pytest -q
python -m compileall -q bot research

git add .
git commit -m "Fix paper execution, accounting and risk gates; add regression tests"
git push -u origin fix/paper-execution-accounting-risk
```

Then open a pull request targeting `claude/github-repo-connection-wt3063`. If the target branch has advanced, `git apply --check` may fail; resolve the differences and rerun tests rather than forcing an overwrite.

The ZIP contains a complete source snapshot as an alternative. It excludes `.git`, databases, credentials and local test caches. The included strategy parameters are the original defaults, not a replacement for your deployment's environment variables.

## Paper deployment checklist — only after review

1. Do **not** merge into an auto-deployed branch before reviewing the changes and arranging a maintenance window.
2. Preserve a verified SQLite backup, including WAL content, and the existing environment/configuration. The **old** `/backup` command copied only the main DB; stop the old process and use SQLite's backup API/checkpoint safely if backing up before this patch is deployed.
3. Review any open positions and legacy trailing behavior. Do not automatically flatten them or reset the database as part of deploying this patch.
4. Test first against a **copy** of the database in an isolated paper environment. Use separate/disabled Telegram credentials to avoid two instances consuming the same commands.
5. Keep `MODE=paper`; live mode remains deliberately rejected by configuration validation.
6. Confirm `/config`, `/risk`, `/stats` and `/stats new`. Old results should be marked unvalidated; corrected net P&L may be slightly lower due to entry fees.
7. Watch for `stop_beyond_liquidation` rejections at 10x leverage and for `execution_history_halt`. Do not simply disable these protections to restore trade frequency.
8. A missing-history halt requires investigation/replay before an operator explicitly clears its state. `/resume` intentionally does not clear it. Existing exit management continues.
9. Rerun historical validation with matching settings and real funding coverage, then gather a fresh forward-paper sample. Do not reuse the old CAGR as an expected return.

No remote push, PR, deployment, database migration or live trade was performed while preparing this patch.
