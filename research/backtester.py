"""Event-realistic portfolio backtester.

PARITY CONTRACT: signal generation calls the SAME StrategyBrain code the live
paper engine runs (strategy.py is single-sourced). Execution semantics mirror
paper_engine.py Phase 0 as closely as candle data allows:

  - signals evaluated only on CLOSED candles (no intrabar lookahead)
  - entry fills at the signal candle CLOSE (paper engine parity)
  - exits checked on every subsequent candle's HIGH/LOW (intrabar), with the
    pessimistic tie-break: if one candle spans both SL and TP, the SL fills first
  - costs: taker fee + slippage each side, plus perp funding accrued at 8h
    boundaries (real per-interval rates when a funding series is supplied)
  - time-stop exits at candle close after max_hold_hours
  - portfolio rules from the live engine: one position per pair, max concurrent
    positions, fixed-fractional sizing with margin/notional caps

Conscious coarseness vs the live engine: live exits check 1m candles; the
backtester works on the strategy timeframe candle itself (5m/15m). Both use
pessimistic tie-breaks, so bias runs conservative in both. Documented, accepted.
"""
from __future__ import annotations

import heapq
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from config import BotConfig
from strategy import StrategyBrain, timeframe_for_pair

WARMUP = 420  # bars of indicator warm-up prefacing each signal evaluation


@dataclass
class BacktestResult:
    trades: pd.DataFrame
    equity_curve: pd.DataFrame
    n_signals: int
    meta: dict = field(default_factory=dict)

    @property
    def summary(self) -> dict:
        t = self.trades
        out = {
            'trades': len(t), 'n_signals': self.n_signals,
            'start_balance': self.meta['start_balance'], 'end_balance': self.meta['end_balance'],
            'net_pnl': 0.0, 'win_rate': 0.0, 'profit_factor': None,
            'avg_R': 0.0, 'expectancy_R': 0.0,
            'fees_total': 0.0, 'funding_total': 0.0, 'cost_share_of_gross': None,
            'max_drawdown_pct': 0.0,
        }
        if t.empty:
            return out
        wins = t[t.pnl > 0]; losses = t[t.pnl <= 0]
        gross_win = wins.pnl.sum(); gross_loss = abs(losses.pnl.sum())
        out.update({
            'net_pnl': float(t.pnl.sum()),
            'win_rate': float(100 * len(wins) / len(t)),
            'profit_factor': float(gross_win / gross_loss) if gross_loss > 0 else None,
            'avg_R': float(t.R_multiple.mean()),
            'expectancy_R': float(t.R_multiple.mean()),
            'fees_total': float(t.fees.sum() - t.funding_cost.sum()),
            'funding_total': float(t.funding_cost.sum()),
            'cost_share_of_gross': float((t.fees.sum()) / gross_win) if gross_win > 0 else None,
        })
        eq = self.equity_curve
        if not eq.empty:
            dd = (eq.balance.cummax() - eq.balance) / eq.balance.cummax() * 100
            out['max_drawdown_pct'] = float(dd.max())
        return out


def _funding_boundaries(start_ts, end_ts, funding_series, flat_rate_pct):
    """Yield (boundary_ts, rate_pct) pairs between start and end.

    funding_series: optional DataFrame indexed by funding time with column 'rate_pct'.
    Fallback: flat rate every 8h from data start.
    """
    if start_ts >= end_ts:
        return []
    if funding_series is None or funding_series.empty:
        n = int(end_ts.timestamp()) // 28800 - int(start_ts.timestamp()) // 28800
        return [(None, flat_rate_pct)] * max(0, n)
    mask = (funding_series.index > start_ts) & (funding_series.index <= end_ts)
    return list(funding_series.loc[mask, 'rate_pct'].items())


class PortfolioBacktester:
    def __init__(self, cfg: BotConfig | None = None, warmup=WARMUP, use_fast=True):
        self.cfg = cfg or BotConfig()
        self.warmup = warmup
        self.use_fast = use_fast
        self.brain = StrategyBrain(self.cfg)
        self._sig_frames: dict[str, pd.DataFrame] = {}

    # ------------------------------------------------------------------
    def run(self, data: dict[str, pd.DataFrame], funding: dict[str, pd.DataFrame] | None = None,
            lookahead_shift: int = 0, start_offset: int = 0) -> BacktestResult:
        """lookahead_shift=0 is the honest run; 1/-1 shift the ENTRY one candle
        earlier/later while signals stay identical — the primitive behind the
        lookahead-bias audit (research/bias_audits.py)."""
        funding = funding or {}
        pairs = sorted(data.keys())
        frames = {p: data[p].sort_index() for p in pairs}
        if self.use_fast:
            from .signals_fast import compute_signals
            if set(self._sig_frames) != set(pairs) or any(
                    self._sig_frames[p].index is not frames[p].index for p in pairs):
                self._sig_frames = {p: compute_signals(p, f) for p, f in frames.items()}
            # seen-signals are an INFORMATION-SET property: count them from the
            # matrix so the metric is invariant to portfolio state and shifts
            n_signals = sum(
                int(sf.signal.iloc[self.warmup + start_offset:].sum())  # noqa: E203
                for sf in self._sig_frames.values()
            )
        else:
            n_signals = 0

        # Merged chronological candle cursor across all pairs
        streams = [(frames[p].index, p) for p in pairs]
        merged = heapq.merge(*[((ts, i, p) for i, ts in enumerate(ix)) for ix, p in streams],
                             key=lambda x: x[0])

        equity = float(self.cfg.starting_balance)
        start_balance = equity
        open_pos: dict[str, dict] = {}
        trades = []
        equity_curve = []

        for ts, i, pair in merged:
            df = frames[pair]
            just_exited = False
            # ---- exits first (risk before entries), never on the entry candle
            pos = open_pos.get(pair)
            if pos is not None and ts > pos['opened_at']:
                high = float(df.high.iat[i]); low = float(df.low.iat[i])
                close = float(df.close.iat[i])
                reason = None; exit_px = None
                if pos['side'] == 'LONG':
                    if low <= pos['sl']:
                        reason, exit_px = 'SL', pos['sl']
                    elif high >= pos['tp']:
                        reason, exit_px = 'TP', pos['tp']
                else:
                    if high >= pos['sl']:
                        reason, exit_px = 'SL', pos['sl']
                    elif low <= pos['tp']:
                        reason, exit_px = 'TP', pos['tp']
                if reason is None and self.cfg.close_on_time_exit:
                    if ts >= pos['opened_at'] + pd.Timedelta(hours=pos['max_hold_hours']):
                        reason, exit_px = 'TIME', close
                if reason:
                    equity += self._close(pos, exit_px, ts, reason, funding.get(pair), trades)
                    del open_pos[pair]
                    equity_curve.append((ts, equity))
                    just_exited = True  # no re-entry on the exit candle, but the
                                        # bar's signal is still COUNTED below so
                                        # seen-signal metrics stay shift-invariant

            # ---- entries: guards first, then signal evaluation, then gates,
            # so signal counting stays a pure information-set property
            if i < self.warmup + start_offset:
                continue
            entry_i = i - lookahead_shift if lookahead_shift > 0 else (i + 1 if lookahead_shift < 0 else i)
            if not (0 <= entry_i < len(df)):
                continue
            if self.use_fast:
                sf = self._sig_frames[pair]
                if not sf.signal.iat[i]:
                    continue
                sig = {
                    'pair': pair, 'side': sf.side.iat[i],
                    'setup': sf.setup.iat[i] or 'FAST',
                    'entry': float(df.close.iat[i]),
                    'sl': float(sf.sl.iat[i]), 'tp': float(sf.tp.iat[i]),
                    'risk_pct': float(sf.risk_pct.iat[i]),
                    'max_hold_hours': int(sf.max_hold_hours.iat[i]),
                    'risk_mult': 1.0,
                }
            else:
                window = df.iloc[i - self.warmup:i + 1]  # noqa: E203
                sig = self.brain.latest_signal(pair, window)
                if not sig:
                    continue
                n_signals += 1
            if just_exited:
                continue  # counted the signal above; no same-candle re-entry
            if pair in open_pos:
                continue
            if len(open_pos) >= self.cfg.max_open_positions:
                continue
            # signals are ALWAYS computed against data through bar i only;
            # lookahead_shift moves the FILL candle, never the information set
            entry_px = float(df.close.iat[entry_i])
            notional, margin, qty = self._size(sig, equity, entry_px)
            open_pos[pair] = {
                'pair': pair, 'side': sig['side'], 'setup': sig['setup'],
                'entry': entry_px, 'sl': sig['sl'], 'tp': sig['tp'],
                'notional': notional, 'margin': margin,
                'opened_at': df.index[entry_i], 'signal_at': ts,
                'max_hold_hours': sig['max_hold_hours'],
                'regime': None,  # attached by regimes.py post-hoc
            }
        # force-close anything still open at data end
        for pair, pos in list(open_pos.items()):
            df = frames[pair]
            equity += self._close(pos, float(df.close.iat[-1]), df.index[-1], 'EOD', funding.get(pair), trades)
            equity_curve.append((df.index[-1], equity))

        tdf = pd.DataFrame(trades)
        eq_df = pd.DataFrame(equity_curve, columns=['ts', 'balance'])
        if not eq_df.empty:
            eq_df = eq_df.set_index('ts')
        return BacktestResult(
            trades=tdf, equity_curve=eq_df, n_signals=n_signals,
            meta={'start_balance': start_balance, 'end_balance': equity,
                  'warmup': self.warmup, 'lookahead_shift': lookahead_shift,
                  'start_offset': start_offset, 'pairs': pairs},
        )

    # ------------------------------------------------------------------
    def _size(self, sig, equity, entry_px):
        """Mirrors PaperEngine.calc_size exactly."""
        risk_usdt = equity * (self.cfg.risk_per_trade_pct / 100) * float(sig.get('risk_mult', 1.0))
        stop_pct = max(float(sig['risk_pct']), 1e-6)
        notional = risk_usdt / stop_pct
        notional = min(notional, equity * (self.cfg.max_notional_pct / 100))
        margin = notional / self.cfg.leverage
        if margin > equity * (self.cfg.max_margin_per_position_pct / 100):
            margin = equity * (self.cfg.max_margin_per_position_pct / 100)
            notional = margin * self.cfg.leverage
        return notional, margin, notional / entry_px

    def _close(self, pos, exit_px, ts, reason, funding_series, trades):
        notional = pos['notional']; entry = pos['entry']
        gross = (exit_px / entry - 1) if pos['side'] == 'LONG' else (entry / exit_px - 1)
        drag = 2 * ((self.cfg.taker_fee_pct + self.cfg.slippage_pct) / 100)
        # funding: longs PAY positive rates, shorts RECEIVE them (and pay negative)
        events = _funding_boundaries(pos['opened_at'], ts, funding_series, self.cfg.funding_rate_pct_8h)
        sign = 1.0 if pos['side'] == 'LONG' else -1.0
        funding_cost = sum(notional * (rate / 100) * sign for _, rate in events)
        pnl = notional * (gross - drag) - funding_cost
        fees = notional * 2 * (self.cfg.taker_fee_pct / 100) + funding_cost
        risk_usdt = abs(entry - pos['sl']) * (notional / entry)
        trades.append({
            'pair': pos['pair'], 'side': pos['side'], 'setup': pos['setup'],
            'entry_time': pos['opened_at'], 'exit_time': ts,
            'entry': entry, 'exit': exit_px, 'reason': reason,
            'notional': notional, 'pnl': pnl,
            'pnl_pct': 100 * pnl / notional,
            'R_multiple': pnl / risk_usdt if risk_usdt > 0 else 0.0,
            'fees': fees, 'funding_cost': funding_cost,
            'funding_events': len(events),
            'hold_hours': (ts - pos['opened_at']).total_seconds() / 3600,
        })
        return pnl
