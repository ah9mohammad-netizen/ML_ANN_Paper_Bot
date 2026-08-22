#!/usr/bin/env bash
# ============================================================================
# ONE-COMMAND RESEARCH GATE — run anywhere exchange APIs are reachable
# (Railway shell, your laptop, a VPS — NOT inside network-restricted sandboxes)
#
#   bash run_research_gate.sh            # 2 years, all 12 pairs (~30-60 min)
#   DAYS=400 bash run_research_gate.sh   # shorter pull
#   bash run_research_gate.sh --synthetic-only   # verify pipeline w/o network
#
# It will: (1) install research deps, (2) freeze multi-regime OKX data +
# historical funding rates, (3) run the truth backtest with pessimistic fills,
# (4) run lookahead + recursive bias audits, (5) write reports to
# research/output/. Gate verdicts use the pre-registered thresholds from
# PHASE2_RESEARCH_GATE.md.
# ============================================================================
set -euo pipefail
cd "$(dirname "$0")"

DAYS="${DAYS:-730}"
PY="${PYTHON:-python3}"

if [ "${1:-}" == "--synthetic-only" ]; then
  $PY -m pip install -q -r requirements-research.txt
  $PY -m research.run_research_gate --synthetic
  exit 0
fi

echo "==> [1/4] deps"
$PY -m pip install -q -r requirements-research.txt

echo "==> [2/4] freezing $DAYS days of OKX candles + funding (this is the slow part)"
$PY -m research.fetch_data --days "$DAYS"

echo "==> [3/4] truth-run backtest + bias audits on frozen data"
$PY -m research.run_research_gate --frozen

echo "==> [4/4] done — read research/output/REAL_W2_truth_run.md and REAL_W5_bias_audits.md"
