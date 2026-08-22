"""Report generation: markdown + CSV artifacts from backtest results."""
from __future__ import annotations

from pathlib import Path

import pandas as pd

from .regimes import slice_table


def _fmt_pf(pf):
    return '∞' if pf in (None, float('inf')) else f'{pf:.2f}'


def summary_block(name: str, s: dict) -> str:
    cs = f"{100*s['cost_share_of_gross']:.0f}%" if s['cost_share_of_gross'] is not None else 'n/a'
    return (
        f"### {name}\n"
        f"| metric | value |\n|---|---|\n"
        f"| trades (signals seen) | {s['trades']} ({s['n_signals']}) |\n"
        f"| win rate | {s['win_rate']:.1f}% |\n"
        f"| profit factor | {_fmt_pf(s['profit_factor'])} |\n"
        f"| expectancy | {s['expectancy_R']:+.3f} R / trade |\n"
        f"| net PnL | {s['net_pnl']:+.2f} USDT on {s['start_balance']:.0f} |\n"
        f"| max drawdown | {s['max_drawdown_pct']:.1f}% |\n"
        f"| fees paid | {s['fees_total']:.2f} USDT |\n"
        f"| funding paid | {s['funding_total']:.2f} USDT |\n"
        f"| cost share of gross wins | {cs} |\n"
    )


def per_pair_table(trades: pd.DataFrame) -> str:
    if trades.empty:
        return '_no trades_\n'
    rows = ['| pair | side | trades | win% | net PnL | PF | avg R | funding |', '|---|---|---|---|---|---|---|---|']
    for (pair, side), g in trades.groupby(['pair', 'side']):
        w = g.loc[g.pnl > 0, 'pnl'].sum(); l = abs(g.loc[g.pnl <= 0, 'pnl'].sum())
        rows.append(
            f"| {pair} | {side} | {len(g)} | {100*(g.pnl > 0).mean():.0f}% "
            f"| {g.pnl.sum():+.2f} | {_fmt_pf(round(w / l, 2) if l > 0 else None)} "
            f"| {g.R_multiple.mean():+.3f} | {g.funding_cost.sum():.2f} |"
        )
    return '\n'.join(rows) + '\n'


def slice_markdown(trades: pd.DataFrame, col: str, title: str) -> str:
    t = slice_table(trades, col)
    if t.empty:
        return f'#### {title}\n_no data_\n'
    lines = [f'#### {title}', f'| {col} | trades | win% | net PnL | PF | avg R |', '|---|---|---|---|---|---|']
    for _, r in t.iterrows():
        lines.append(f"| {r[col]} | {r.trades} | {r.win_rate} | {r.net_pnl} | {_fmt_pf(r.PF)} | {r.avg_R} |")
    return '\n'.join(lines) + '\n'


def write_report(path: Path, title: str, sections: list[str],
                 trades: pd.DataFrame | None = None, artifact_dir: Path | None = None):
    if trades is not None and artifact_dir is not None:
        artifact_dir.mkdir(parents=True, exist_ok=True)
        trades.to_csv(artifact_dir / 'trades.csv', index=False)
    Path(path).write_text('\n\n'.join(sections))
    print(f'📝 wrote {path}')
