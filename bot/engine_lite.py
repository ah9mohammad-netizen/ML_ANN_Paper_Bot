"""Refit adapter to the research engine (no second execution implementation).

No funding history is downloaded by refit.py. The report says so explicitly;
refit statistics must not be represented as a fully costed live forecast.
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from research.engine import Backtester, RiskConfig, ExecConfig


def run_portfolio(data, sigs, cfg, bar_minutes):
    data = {s: d for s, d in data.items() if s in sigs and len(d)}
    if not data:
        return {"n_trades": 0}
    risk = RiskConfig(
        starting_equity=cfg.starting_equity, risk_per_trade=cfg.risk_per_trade,
        leverage=cfg.leverage, max_open=cfg.max_open,
        max_gross_notional=cfg.max_gross_notional,
        max_notional_per_trade=cfg.max_notional_per_trade,
        same_symbol_lock=cfg.same_symbol_lock,
        max_daily_loss=cfg.max_daily_loss, max_drawdown_halt=cfg.max_drawdown_halt,
        correlation_group_cap=cfg.max_correlated,
        max_consecutive_losses=cfg.max_consecutive_losses)
    ex = ExecConfig(taker_fee=cfg.taker_fee, slip_entry_bps=cfg.slip_entry_bps,
                    slip_stop_bps=cfg.slip_stop_bps, use_funding=cfg.apply_funding,
                    chandelier=True)
    return Backtester(data, sigs, risk=risk, ex=ex, bar_minutes=bar_minutes).run()
