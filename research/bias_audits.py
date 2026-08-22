"""Bias audits — the hostile-witness layer.

Two audits, mirroring Freqtrade's lookahead-analysis / recursive-analysis
methodology, adapted to this codebase's entry conventions:

1. lookahead_audit: runs the backtest with the ENTRY FILL shifted one candle
   earlier (-1 conceptually) and one candle later (+1) while signals are kept
   bitwise identical. For honest closed-candle strategies the three runs agree
   within noise. The classic leak signature is an edge that is large in the
   honest run but COLLAPSES when entry is moved one candle later (the "next-bar
   move" was the leaked information) — plus inflated performance when fills are
   moved earlier under a leak conditioned on the signal bar's own range.

2. recursive_audit: re-runs with increasing amounts of leading data dropped
   (startup candle counts). Correct indicators converge: results must be
   statistically invariant. Divergence = recursive/warmup formula problem.

3. mutant_self_test: sanity proof that the lookahead audit would actually CATCH
   a leak: we run a planted perfect-foresight strategy (signals conditioned on
   the NEXT bar's return) and assert the audit flags it. If the detector can't
   catch a blatant cheater, its verdict on the real strategy is worthless.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .backtester import PortfolioBacktester


def _pf(trades: pd.DataFrame) -> float:
    if trades.empty:
        return 0.0
    w = trades.loc[trades.pnl > 0, 'pnl'].sum()
    l = abs(trades.loc[trades.pnl <= 0, 'pnl'].sum())
    return float(w / l) if l > 0 else (float('inf') if w > 0 else 0.0)


# ---------------------------------------------------------------------------
def lookahead_audit(bt: PortfolioBacktester, data, funding=None) -> dict:
    r_early = bt.run(data, funding, lookahead_shift=1)
    r_honest = bt.run(data, funding, lookahead_shift=0)
    r_late = bt.run(data, funding, lookahead_shift=-1)

    pf_e, pf_h, pf_l = _pf(r_early.trades), _pf(r_honest.trades), _pf(r_late.trades)
    pnl_e, pnl_h, pnl_l = r_early.trades.pnl.sum(), r_honest.trades.pnl.sum(), r_late.trades.pnl.sum()

    flags = []
    # Signature A: edge collapses when entry moved later (leaked next-bar move)
    if pf_h > 1.2 and pf_l < max(pf_h * 0.7, pf_h - 0.35):
        flags.append('edge_collapse_on_late_entry')
    # Signature B: unnaturally large uplift when fills move earlier
    # (positive territory only — on losing books inequality noise is meaningless)
    if (pf_h > 1.0 and pnl_h > 0 and np.isfinite(pf_e)
            and pf_e > pf_h * 1.5 and pnl_e > pnl_h * 1.4):
        flags.append('suspicious_uplift_on_early_entry')
    # Generation must be shift-invariant (with the fast path this holds by
    # construction; strict path double-checks it end-to-end)
    if len({r_early.n_signals, r_honest.n_signals, r_late.n_signals}) != 1:
        flags.append('signal_count_changed_under_shift')

    verdict = 'CLEAN' if not flags else 'BIASED'
    return {
        'verdict': verdict, 'flags': flags,
        'n_signals': r_honest.n_signals,
        'pf_early': pf_e, 'pf_honest': pf_h, 'pf_late': pf_l,
        'pnl_early': float(pnl_e), 'pnl_honest': float(pnl_h), 'pnl_late': float(pnl_l),
        'trades': len(r_honest.trades),
        'runs': {'early': r_early, 'honest': r_honest, 'late': r_late},
    }


# ---------------------------------------------------------------------------
def recursive_audit(bt, data, funding=None, offsets=(0, 400, 1200)) -> dict:
    variants = {off: bt.run(data, funding, lookahead_shift=0, start_offset=off)
                for off in offsets}
    pfs = {off: _pf(r.trades) for off, r in variants.items()}
    base = pfs[offsets[0]]
    max_dev = max((abs(pf - base) for pf in pfs.values()), default=0.0)
    verdict = 'CLEAN' if max_dev <= 0.25 or all(v == 0.0 for v in pfs.values()) else 'DIVERGENT'
    return {'verdict': verdict, 'pf_by_warmup_offset': pfs,
            'max_pf_deviation': max_dev,
            'trades_by_offset': {off: len(r.trades) for off, r in variants.items()}}


# ---------------------------------------------------------------------------
class PerfectForesightBrain:
    """Planted mutant for the detector self-test: LONG signals conditioned on
    the NEXT bar's return exceeding a threshold. Blatant lookahead."""

    def __init__(self, full_data: dict[str, pd.DataFrame]):
        self.signals = {}
        for pair, df in full_data.items():
            fwd_ret = df.close.shift(-1) / df.close - 1
            th = fwd_ret.rolling(200).std() * 3.0  # only monster up-bars
            mask = fwd_ret > th
            # canonical leak: AT bar i, decide using the i→i+1 move
            sig_idx = np.where(mask.fillna(False).values)[0]
            self.signals[pair] = set(sig_idx[sig_idx > 0])
        self._data = full_data

    def latest_signal(self, pair, window):
        # called by the backtester with window ending at bar i; our dict knows
        # whether a foresight signal exists for the bar BEFORE the last one
        df = self._data[pair]
        i = len(window) - 1 if len(df) == len(window) else df.index.get_loc(window.index[-1])
        if i not in self.signals[pair]:
            return None
        entry = float(window.close.iloc[-1])
        # TP sits comfortably inside the foreseen monster bar; SL deep enough
        # that ordinary wick noise can't stop out first. Foresight profit must
        # be big in the honest run and COLLAPSE when the fill moves one bar late.
        return {
            'pair': pair, 'side': 'LONG', 'setup': 'MUTANT_FORESIGHT',
            'probability': 0.5, 'predicted_R': 0.0, 'entry': entry,
            'sl': entry * (1 - 0.035), 'tp': entry * (1 + 0.010),
            'risk_pct': 0.035, 'signal_time': str(window.index[-1]),
            'risk_mult': 1.0, 'max_hold_hours': 12, 'meta': {},
        }


def mutant_self_test(bt_cfg, data, funding=None, max_bars=9000) -> dict:
    """The detector MUST flag a strategy that peeks one bar ahead. Runs the
    mutant in STRICT mode (real per-bar brain calls) on a trimmed slice —
    strict mode is O(bars × window) so we keep the forensic sample bounded."""
    bt_normal = PortfolioBacktester(bt_cfg)
    honest = lookahead_audit(bt_normal, data, funding)

    lean = {p: df.tail(max_bars) for p, df in data.items()}
    bt_mutant = PortfolioBacktester(bt_cfg, use_fast=False)
    bt_mutant.brain = PerfectForesightBrain(lean)
    leaked = lookahead_audit(bt_mutant, lean, funding)

    return {
        'real_strategy_verdict': honest['verdict'],
        'real_strategy_flags': honest['flags'],
        'mutant_verdict': leaked['verdict'],
        'mutant_flags': leaked['flags'],
        'mutant_pf_honest': leaked['pf_honest'],
        'mutant_pf_late': leaked['pf_late'],
        'detector_works': leaked['verdict'] == 'BIASED',
        'detail': honest,
    }
