"""Research-gate orchestrator (W2 truth run + W5 bias audits + W6 regime slicing).

Modes:
  python -m research.run_research_gate --synthetic     # pipeline verification (sandbox/CI)
  python -m research.run_research_gate --frozen        # REAL run on research/data/frozen/*

The synthetic mode proves the machinery (incl. the lookahead detector's mutant
self-test) but says NOTHING about edge. Only --frozen output feeds Go/No-Go.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from config import BotConfig
from strategy import CATEGORY_SMC, CATEGORY_SCALPER, timeframe_for_pair
from .backtester import PortfolioBacktester
from .bias_audits import lookahead_audit, recursive_audit, mutant_self_test
from .regimes import tag_trades
from .report import summary_block, per_pair_table, slice_markdown, write_report
from .synthetic_data import generate_pair_history

OUT = Path(__file__).parent / 'output'
FROZEN = Path(__file__).parent / 'data' / 'frozen'

# Pre-registered gate thresholds (from PHASE2_RESEARCH_GATE.md)
G2_MIN_PF, G2_MIN_TRADES = 1.3, 200

SYNTH_BLOCKS = [('bear', 12000), ('chop', 12000), ('bull', 12000)]  # pipeline demo


def load_frozen():
    data, funding = {}, {}
    for f in sorted(FROZEN.glob('*.parquet')):
        if 'funding' in f.name:
            pair = f.name.split('-')[0]
            fdf = pd.read_parquet(f).set_index('date')
            funding[pair] = fdf
        else:
            pair = f.name.split('-')[0]
            df = pd.read_parquet(f).set_index('date')
            data[pair] = df
    return data, funding


def synthetic_set():
    reps = [('BTC', '15m'), ('SOL', '15m'), ('PEPE', '5m'), ('DOGE', '5m')]
    return {p: generate_pair_history(p, tf, SYNTH_BLOCKS, seed=11) for p, tf in reps}


def regime_report_sections(res, data, label):
    tagged = tag_trades(res.trades, data)
    s = res.summary
    secs = [
        f'# Phase 2 Research Run — {label}',
        summary_block(f'Portfolio (12-pair routing, all engines)', s),
        '## Per pair / side\n' + per_pair_table(tagged),
        '## Regime slicing (W6)\n' + slice_markdown(tagged, 'adx_bucket', 'ADX regime'),
        slice_markdown(tagged, 'trend', 'EMA-200 trend state'),
        slice_markdown(tagged, 'vol_bucket', 'Efficiency-ratio quartile'),
        slice_markdown(tagged, 'pair', 'Per-pair'),
        slice_markdown(tagged, 'reason', 'Exit reason'),
    ]
    gate_pass = (s['profit_factor'] is not None and s['profit_factor'] >= G2_MIN_PF
                 and s['trades'] >= G2_MIN_TRADES)
    secs.append(
        f"## Gate G2 verdict (pre-registered: PF ≥ {G2_MIN_PF} with ≥ {G2_MIN_TRADES} trades)\n\n"
        + ('✅ PASS' if gate_pass else '❌ FAIL — strategy rejected at pre-registered threshold, no tuning rescue allowed')
    )
    return secs, tagged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--synthetic', action='store_true')
    ap.add_argument('--frozen', action='store_true')
    args = ap.parse_args()
    cfg = BotConfig()
    bt = PortfolioBacktester(cfg)

    if args.synthetic:
        OUT.mkdir(parents=True, exist_ok=True)
        data = synthetic_set()
        print('🧪 SYNTHETIC pipeline verification (NOT edge evidence)')
        # fast-path vs strict-path parity (research/signals_fast.py contract)
        from .signals_fast import parity_report
        for p, d in data.items():
            rep = parity_report(p, d, n_samples=120)
            print(f'   parity {p}: {rep["agreement"]*100:.1f}% agreement over {rep["samples"]} sampled bars')
        res = bt.run(data)
        secs, tagged = regime_report_sections(res, data, 'SYNTHETIC verification')
        write_report(OUT / 'synthetic_truth_run.md', 'synthetic', secs, tagged, OUT)

        print('🔍 lookahead audit (real strategy, synthetic data)...')
        la = lookahead_audit(bt, data)
        print('🔁 recursive audit...')
        ra = recursive_audit(bt, data)
        print('🧬 detector mutant self-test...')
        mt = mutant_self_test(cfg, data)
        audit_md = [
            '# Bias audits — SYNTHETIC data (machinery validation)',
            f"## Lookahead audit (strategy.py): **{la['verdict']}**",
            f"- flags: {la['flags']}",
            f"- PF early/honest/late: {la['pf_early']:.2f} / {la['pf_honest']:.2f} / {la['pf_late']:.2f}",
            f"- PnL early/honest/late: {la['pnl_early']:+.2f} / {la['pnl_honest']:+.2f} / {la['pnl_late']:+.2f}",
            f"## Recursive audit: **{ra['verdict']}**",
            f"- PF by warmup offset: { {k: round(v, 2) for k, v in ra['pf_by_warmup_offset'].items()} }",
            f"## Detector mutant self-test: **{'WORKS' if mt['detector_works'] else 'BROKEN'}**",
            f"- mutant verdict: {mt['mutant_verdict']} flags={mt['mutant_flags']}",
            f"- mutant PF late/honest: {mt['mutant_pf_late']:.2f} / {mt['mutant_pf_honest']:.2f} "
            f"(foresight edge must collapse on late entry)",
        ]
        write_report(OUT / 'synthetic_bias_audits.md', 'audits', audit_md)

    elif args.frozen:
        if not FROZEN.exists() or not any(FROZEN.glob('*.parquet')):
            raise SystemExit('no frozen data — run: python -m research.fetch_data --days 730')
        data, funding = load_frozen()
        print(f'📦 frozen set: {sorted(data.keys())}')
        res = bt.run(data, funding)
        secs, tagged = regime_report_sections(res, data, 'REAL frozen dataset')
        hdr = ['# Phase 2 — REAL RUN (frozen OKX data + historical funding)',
               f"pairs: {sorted(data.keys())}"]
        write_report(OUT / 'REAL_W2_truth_run.md', 'real', hdr + secs, tagged, OUT)

        la = lookahead_audit(bt, data, funding)
        ra = recursive_audit(bt, data, funding)
        audit_md = [
            '# Bias audits — REAL frozen data',
            f"## Lookahead audit: **{la['verdict']}**", f"- flags: {la['flags']}",
            f"- PF early/honest/late: {la['pf_early']:.2f} / {la['pf_honest']:.2f} / {la['pf_late']:.2f}",
            f"- PnL early/honest/late: {la['pnl_early']:+.2f} / {la['pnl_honest']:+.2f} / {la['pnl_late']:+.2f}",
            f"## Recursive audit: **{ra['verdict']}**",
            f"- PF by warmup offset: { {k: in_round(v) for k, v in ra['pf_by_warmup_offset'].items()} }",
        ]
        write_report(OUT / 'REAL_W5_bias_audits.md', 'audits', audit_md)
    else:
        ap.error('choose --synthetic or --frozen')


def in_round(v):
    return round(v, 2)


if __name__ == '__main__':
    main()
