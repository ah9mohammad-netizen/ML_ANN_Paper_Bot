"""Event-driven, bar-close portfolio backtester for crypto perpetuals.

Design rules (the ones the old bot broke):
  1. A signal is generated from bar `i` only after bar `i` has CLOSED.
  2. The fill happens at the OPEN of bar `i+1`, plus slippage. Never at the
     close you just looked at.
  3. Exits are checked against the full OHLC of each subsequent bar, so wicks
     that touch the stop are seen. If a bar touches BOTH stop and target,
     the STOP is assumed to fill first (pessimistic, un-resolvable at bar res).
  4. Gaps through a stop fill at the bar's open, not at the stop price.
  5. Fees are taker on both legs; funding is charged at every 8h settlement.
  6. Liquidation is checked against the bar low/high at the configured leverage.

The same Strategy objects are used by the live bot, so backtest and production
see identical signal code.
"""
from __future__ import annotations
import math
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from bot.execution import bar_exit, liquidation_price, trailing_stop, trade_pnl
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd


# ────────────────────────────────────────────────────────────── config

@dataclass
class ExecConfig:
    taker_fee: float = 0.0005        # 5 bps per side (OKX/Binance perp taker, no VIP)
    maker_fee: float = 0.0002
    slip_entry_bps: float = 2.0      # market entry impact
    slip_stop_bps: float = 5.0       # stops fill worse: they trigger into momentum
    use_funding: bool = True
    maint_margin: float = 0.005      # 0.5% maintenance margin -> liq buffer
    pessimistic_ambiguous_bar: bool = True
    # --- switches used only to reproduce the old bot's optimistic assumptions
    fill_at_signal_close: bool = False   # True = fill at the close you just saw
    exits_on_close_only: bool = False    # True = ignore wicks, poll closes only
    model_liquidation: bool = True
    chandelier: bool = False


@dataclass
class RiskConfig:
    starting_equity: float = 1000.0
    risk_per_trade: float = 0.0075   # 0.75% of equity risked to stop
    leverage: float = 5.0
    max_open: int = 4
    max_open_per_side: int = 99
    max_gross_notional: float = 3.0  # x equity
    max_notional_per_trade: float = 1.0
    same_symbol_lock: bool = True
    max_daily_loss: float = 0.06     # halt new entries for the day past this
    max_drawdown_halt: float = 0.25  # hard stop the whole run
    correlation_group_cap: int = 99
    max_consecutive_losses: int = 0  # 0 disables this optional paper-style gate


# ────────────────────────────────────────────────────────────── objects

@dataclass
class Position:
    symbol: str
    side: int                # +1 long, -1 short
    entry_ts: pd.Timestamp
    entry: float
    qty: float
    notional: float
    sl: float
    tp: float
    max_hold_bars: int
    tag: str
    bars_held: int = 0
    entry_fee: float = 0.0
    funding_paid: float = 0.0
    r_unit: float = 0.0      # $ risked at entry
    be_moved: bool = False
    trail_active: bool = False
    peak: float = 0.0
    meta: dict = field(default_factory=dict)


@dataclass
class Trade:
    symbol: str; side: int; tag: str
    entry_ts: pd.Timestamp; exit_ts: pd.Timestamp
    entry: float; exit: float
    qty: float; notional: float
    sl: float; tp: float
    reason: str
    gross: float; fees: float; funding: float; pnl: float
    r_unit: float; r_multiple: float
    bars_held: int
    equity_after: float
    meta: dict = field(default_factory=dict)


# ────────────────────────────────────────────────────────────── engine

class Backtester:
    def __init__(self, data: Dict[str, pd.DataFrame],
                 signals: Dict[str, pd.DataFrame],
                 risk: RiskConfig = None, ex: ExecConfig = None,
                 funding: Dict[str, pd.DataFrame] = None,
                 bar_minutes: int = 15):
        """
        data    : symbol -> OHLCV frame (tz-aware UTC index, uniform bars)
        signals : symbol -> frame aligned to data index with columns
                  side (+1/-1/0), sl, tp, max_hold_bars, tag, risk_mult,
                  and optional trail_atr / be_at_r
        """
        self.data = data
        self.sig = signals
        self.risk = risk or RiskConfig()
        self.ex = ex or ExecConfig()
        self.funding = funding or {}
        self.bar_minutes = bar_minutes

        idx = None
        for d in data.values():
            idx = d.index if idx is None else idx.union(d.index)
        self.index = idx.sort_values()

        # numpy views: label lookups in the bar loop are the whole cost centre
        self.arr = {s: {c: d[c].to_numpy(dtype=float)
                        for c in ("open", "high", "low", "close")}
                    for s, d in data.items()}
        self.kpos = {s: {t: k for k, t in enumerate(d.index)}
                     for s, d in data.items()}
        self.sarr = {}
        for s, g in signals.items():
            self.sarr[s] = {
                "side": g["side"].to_numpy(dtype=float),
                "sl": g["sl"].to_numpy(dtype=float),
                "tp": g["tp"].to_numpy(dtype=float),
                "ref_price": g["ref_price"].to_numpy(dtype=float),
                "max_hold_bars": g["max_hold_bars"].to_numpy(dtype=float),
                "atr": g["atr"].to_numpy(dtype=float),
                "risk_mult": g.get("risk_mult", pd.Series(1.0, index=g.index)).to_numpy(dtype=float),
                "be_at_r": g["be_at_r"].to_numpy(dtype=float),
                "trail_atr": g["trail_atr"].to_numpy(dtype=float),
                "trail_start_r": g["trail_start_r"].to_numpy(dtype=float),
                "tag": g["tag"].to_numpy(),
                "regime": g["regime"].to_numpy(),
            }
        self.fund_arr = {}
        for s, f in (funding or {}).items():
            if len(f):
                # Pandas 3 can default to microsecond indices. Timestamp.value
                # is nanoseconds, so normalize both search arrays explicitly.
                self.fund_arr[s] = (f.index.as_unit("ns").asi8,
                                    f["rate"].to_numpy(dtype=float))

        self.equity = self.risk.starting_equity
        self.peak_equity = self.equity
        self.positions: Dict[str, Position] = {}
        self.trades: List[Trade] = []
        self.curve: List[tuple] = []
        self.blocked = 0
        self.block_reasons: Dict[str, int] = {}
        self.halted = False
        self.consecutive_losses = 0
        self._day = None
        self._day_start_equity = self.equity

    # ---- helpers ------------------------------------------------------

    def _mark_equity(self, ts, kmap):
        unreal = 0.0
        for p in self.positions.values():
            k = kmap.get(p.symbol)
            if k is not None:
                unreal += p.side * (self.arr[p.symbol]["close"][k] - p.entry) * p.qty
        eq = self.equity + unreal - sum(p.funding_paid for p in self.positions.values())
        self.peak_equity = max(self.peak_equity, eq)
        self.curve.append((ts, eq, self.equity, len(self.positions)))
        return eq

    def _funding_due(self, symbol, t_prev, t_now):
        f = self.fund_arr.get(symbol)
        if not self.ex.use_funding or f is None:
            return 0.0
        ts_i8, rates = f
        lo = np.searchsorted(ts_i8, t_prev, side="right")
        hi = np.searchsorted(ts_i8, t_now, side="right")
        return float(rates[lo:hi].sum()) if hi > lo else 0.0

    def _block(self, reason):
        self.blocked += 1
        self.block_reasons[reason] = self.block_reasons.get(reason, 0) + 1

    # ---- exits --------------------------------------------------------

    def _close(self, p: Position, ts, price, reason):
        ex = self.ex
        gross = p.side * (price - p.entry) * p.qty
        exit_fee = abs(price * p.qty) * ex.taker_fee
        fees = p.entry_fee + exit_fee
        pnl, cash_delta = trade_pnl(gross, p.entry_fee, exit_fee, p.funding_paid)
        self.equity += cash_delta
        r = pnl / p.r_unit if p.r_unit > 0 else 0.0
        self.trades.append(Trade(
            symbol=p.symbol, side=p.side, tag=p.tag,
            entry_ts=p.entry_ts, exit_ts=ts, entry=p.entry, exit=price,
            qty=p.qty, notional=p.notional, sl=p.sl, tp=p.tp, reason=reason,
            gross=gross, fees=fees, funding=p.funding_paid, pnl=pnl,
            r_unit=p.r_unit, r_multiple=r, bars_held=p.bars_held,
            equity_after=self.equity, meta=dict(p.meta)))
        self.consecutive_losses = self.consecutive_losses + 1 if pnl <= 0 else 0
        if (self.risk.max_consecutive_losses and
                self.consecutive_losses >= self.risk.max_consecutive_losses):
            self.halted = True
        del self.positions[p.symbol]

    def _process_exits(self, ts, prev_ts, kmap, t_prev_i8, t_now_i8):
        for sym in list(self.positions.keys()):
            p = self.positions[sym]
            k = kmap.get(sym)
            if k is None:
                continue
            a = self.arr[sym]
            o, h, l, c = a["open"][k], a["high"][k], a["low"][k], a["close"][k]
            p.bars_held += 1

            # funding
            rate = self._funding_due(sym, max(t_prev_i8, p.entry_ts.value), t_now_i8)
            if rate:
                cost = rate * p.notional * p.side      # long pays positive rate
                p.funding_paid += cost

            ex = self.ex
            sslip = ex.slip_stop_bps / 1e4
            liq = (p.entry * (1 - 1 / self.risk.leverage + ex.maint_margin)
                   if p.side > 0 else
                   p.entry * (1 + 1 / self.risk.leverage - ex.maint_margin))

            if ex.exits_on_close_only:
                # what paper_engine.py actually does: poll the last close and
                # fill the stop/target at their exact price. Wicks are invisible.
                if ex.model_liquidation and ((p.side > 0 and c <= liq) or
                                             (p.side < 0 and c >= liq)):
                    self._close(p, ts, liq, "LIQ"); continue
                if (p.side > 0 and c <= p.sl) or (p.side < 0 and c >= p.sl):
                    self._close(p, ts, p.sl, "SL"); continue
                if (p.side > 0 and c >= p.tp) or (p.side < 0 and c <= p.tp):
                    self._close(p, ts, p.tp, "TP"); continue
                if p.max_hold_bars and p.bars_held >= p.max_hold_bars:
                    self._close(p, ts, c, "TIME")
                continue

            px, reason = bar_exit(
                p.side, o, h, l, p.sl, p.tp,
                liq if ex.model_liquidation else None, sslip,
                pessimistic=ex.pessimistic_ambiguous_bar)
            if reason:
                self._close(p, ts, px, reason)
                continue

            # 4. time barrier
            if p.max_hold_bars and p.bars_held >= p.max_hold_bars:
                self._close(p, ts, c * (1 - p.side * ex.slip_entry_bps / 1e4),
                            "TIME"); continue

            # Same trailing formula as paper; the supplied bars determine
            # temporal resolution. Current-bar ATR is used only at its close.
            atr = p.meta.get("atr_at_entry", 0)
            if ex.chandelier:
                av = self.sarr[sym]["atr"][k]
                if np.isfinite(av) and av > 0:
                    atr = float(av)
            p.sl, p.peak, p.be_moved = trailing_stop(
                side=p.side, entry=p.entry, sl0=p.meta["sl0"], sl=p.sl,
                close=c, high=h, low=l, atr=atr, peak=p.peak,
                trail_atr=p.meta.get("trail_atr", 0),
                trail_start_r=p.meta.get("trail_start_r", 1),
                be_at_r=p.meta.get("be_at_r", 0), be_done=p.be_moved,
                fee_rate=ex.taker_fee, chandelier=ex.chandelier)

    # ---- entries ------------------------------------------------------

    def _size(self, entry, sl, risk_mult):
        stop_d = abs(entry - sl)
        if stop_d <= 0:
            return 0, 0, 0
        risk_usd = self.equity * self.risk.risk_per_trade * risk_mult
        qty = risk_usd / stop_d
        notional = qty * entry
        cap = self.equity * self.risk.max_notional_per_trade
        if notional > cap:
            notional = cap; qty = notional / entry
        gross = sum(p.notional for p in self.positions.values())
        room = self.equity * self.risk.max_gross_notional - gross
        if notional > room:
            notional = max(0.0, room); qty = notional / entry
        return qty, notional, qty * stop_d

    def _try_open(self, ts, sym, row, bar_open, fill_px=None):
        R = self.risk
        if self.halted:
            return self._block("halted")
        if R.same_symbol_lock and sym in self.positions:
            return self._block("symbol_locked")
        if len(self.positions) >= R.max_open:
            return self._block("max_open")
        if (self.equity - self._day_start_equity) / max(self._day_start_equity, 1e-9) \
                <= -R.max_daily_loss:
            return self._block("daily_loss_halt")

        side = int(row["side"])
        if sum(p.side == side for p in self.positions.values()) >= R.max_open_per_side:
            return self._block("max_open_per_side")
        if sym not in {"BTC", "ETH"} and sum(
                p.side == side and p.symbol not in {"BTC", "ETH"}
                for p in self.positions.values()) >= R.correlation_group_cap:
            return self._block("correlation_cap")
        slip = self.ex.slip_entry_bps / 1e4
        base = float(bar_open) if fill_px is None else float(fill_px)
        entry = base * (1 + side * slip)
        sl, tp = float(row["sl"]), float(row["tp"])
        # signal levels were computed off the previous close; re-anchor the
        # distances to the actual fill so R is what we think it is
        ref = float(row["ref_price"])
        sl = entry - (ref - sl)
        tp = entry + (tp - ref)
        if side > 0 and not (sl < entry < tp):
            return self._block("bad_levels")
        if side < 0 and not (tp < entry < sl):
            return self._block("bad_levels")

        liq = liquidation_price(entry, side, R.leverage, self.ex.maint_margin)
        stop_fill = sl * (1 - side * self.ex.slip_stop_bps / 1e4)
        if self.ex.model_liquidation and ((side > 0 and stop_fill <= liq) or
                                           (side < 0 and stop_fill >= liq)):
            return self._block("stop_beyond_liquidation")

        rm = row.get("risk_mult", 1.0)
        qty, notional, r_unit = self._size(entry, sl, 1.0 if not np.isfinite(rm) else float(rm))
        if qty <= 0 or notional < 10:
            return self._block("no_room")
        margin = notional / R.leverage
        if margin > self.equity * 0.9:
            return self._block("margin")

        atr_v = row.get("atr", 0.0)
        fee = notional * self.ex.taker_fee
        self.positions[sym] = Position(
            symbol=sym, side=side, entry_ts=ts, entry=entry, qty=qty,
            notional=notional, sl=sl, tp=tp,
            max_hold_bars=int(row.get("max_hold_bars", 0) or 0),
            tag=str(row.get("tag", "")), entry_fee=fee, r_unit=r_unit,
            peak=entry,
            meta={"sl0": sl,
                  "atr_at_entry": float(atr_v) if np.isfinite(atr_v) else 0.0,
                  "be_at_r": float(row.get("be_at_r", 0.0) or 0.0),
                  "trail_atr": float(row.get("trail_atr", 0.0) or 0.0),
                  "trail_start_r": float(row.get("trail_start_r", 1.0)),
                  "regime": row.get("regime", "")})
        self.equity -= fee

    # ---- main loop ----------------------------------------------------

    def run(self):
        idx = self.index
        i8 = idx.as_unit("ns").asi8
        syms = list(self.data)
        kpos = self.kpos
        prev_i8 = i8[0]
        days = idx.normalize()
        for j, ts in enumerate(idx):
            kmap = {}
            for s in syms:
                k = kpos[s].get(ts)
                if k is not None:
                    kmap[s] = k

            day = days[j]
            if day != self._day:
                self._day = day
                self._day_start_equity = self.equity

            # entries: a signal printed on the PREVIOUS bar fills at THIS open
            if not self.halted:
                for sym, k in kmap.items():
                    if k == 0:
                        continue
                    sa = self.sarr.get(sym)
                    if sa is None:
                        continue
                    side = sa["side"][k - 1]
                    if not side:
                        continue
                    row = {c: sa[c][k - 1] for c in
                           ("side", "sl", "tp", "ref_price", "max_hold_bars",
                            "atr", "risk_mult", "be_at_r", "trail_atr",
                            "trail_start_r", "tag", "regime")}
                    fill = row["ref_price"] if self.ex.fill_at_signal_close else None
                    self._try_open(ts, sym, row, self.arr[sym]["open"][k],
                                   fill_px=fill)

            # Entry at this open must be exposed to THIS bar's wick. Previous
            # code processed exits first and skipped every entry-bar stop.
            self._process_exits(ts, None, kmap, prev_i8, i8[j])
            eq = self._mark_equity(ts, kmap)
            if self.peak_equity > 0 and (self.peak_equity - eq) / self.peak_equity \
                    >= self.risk.max_drawdown_halt:
                self.halted = True
            prev_i8 = i8[j]

        # force-close anything still open at the last mark
        for sym in list(self.positions.keys()):
            p = self.positions[sym]
            self._close(p, idx[-1], float(self.data[sym].close.iloc[-1]), "EOD")
        # EOD exit costs must be reflected in the curve used for CAGR.
        self.curve[-1] = (idx[-1], self.equity, self.equity, 0)
        result = self.report()
        result["funding_history_supplied"] = bool(self.fund_arr)
        if self.ex.use_funding and not self.fund_arr:
            result["funding_warning"] = "Funding enabled but no funding history supplied; costs omitted"
        return result

    # ---- reporting ----------------------------------------------------

    def trades_df(self):
        if not self.trades:
            return pd.DataFrame()
        df = pd.DataFrame([t.__dict__ for t in self.trades])
        df["regime"] = df["meta"].apply(lambda m: m.get("regime", ""))
        return df.drop(columns=["meta"])

    def equity_df(self):
        c = pd.DataFrame(self.curve, columns=["ts", "equity", "realized", "n_open"])
        return c.set_index("ts")

    def report(self):
        t = self.trades_df()
        eq = self.equity_df()
        return build_report(t, eq, self.risk.starting_equity, self.bar_minutes,
                            blocked=self.block_reasons)


# ────────────────────────────────────────────────────────────── metrics

def build_report(t: pd.DataFrame, eq: pd.DataFrame, start_equity: float,
                 bar_minutes: int = 15, blocked=None):
    out = {"n_trades": 0, "blocked": blocked or {}}
    if t is None or t.empty:
        return out
    wins = t[t.pnl > 0]; losses = t[t.pnl <= 0]
    gw = wins.pnl.sum(); gl = abs(losses.pnl.sum())
    e = eq.equity
    dd = (e.cummax() - e) / e.cummax()

    bars_per_day = 1440 / bar_minutes
    ret = e.pct_change().fillna(0)
    ann = math.sqrt(365 * bars_per_day)
    sharpe = (ret.mean() / ret.std() * ann) if ret.std() > 0 else 0.0
    downside = ret[ret < 0].std()
    sortino = (ret.mean() / downside * ann) if downside and downside > 0 else 0.0
    days = max((e.index[-1] - e.index[0]).days, 1)
    total_ret = e.iloc[-1] / start_equity - 1
    cagr = (e.iloc[-1] / start_equity) ** (365 / days) - 1 if e.iloc[-1] > 0 else -1

    R = t.r_multiple
    out.update({
        "n_trades": len(t),
        "win_rate": 100 * len(wins) / len(t),
        "profit_factor": (gw / gl) if gl > 0 else float("inf"),
        "net_pnl": t.pnl.sum(),
        "total_return_pct": 100 * total_ret,
        "cagr_pct": 100 * cagr,
        "expectancy_R": R.mean(),
        "expectancy_usd": t.pnl.mean(),
        "std_R": R.std(),
        "avg_win": wins.pnl.mean() if len(wins) else 0.0,
        "avg_loss": losses.pnl.mean() if len(losses) else 0.0,
        "payoff": (wins.pnl.mean() / abs(losses.pnl.mean())) if len(losses) and losses.pnl.mean() != 0 else float("inf"),
        "max_dd_pct": 100 * dd.max(),
        "calmar": (cagr / dd.max()) if dd.max() > 0 else float("inf"),
        "sharpe": sharpe,
        "sortino": sortino,
        "total_fees": t.fees.sum(),
        "total_funding": t.funding.sum(),
        "fees_pct_of_gross_profit": 100 * t.fees.sum() / gw if gw > 0 else float("inf"),
        "avg_bars_held": t.bars_held.mean(),
        "avg_hold_hours": t.bars_held.mean() * bar_minutes / 60,
        "trades_per_week": len(t) / (days / 7),
        "exit_mix": t.reason.value_counts().to_dict(),
        "n_liquidations": int(t.reason.str.startswith("LIQ").sum()),
        "blocked": blocked or {},
        "final_equity": e.iloc[-1],
        "start": str(e.index[0])[:10], "end": str(e.index[-1])[:10],
    })
    # t-stat on per-trade R and a bootstrap CI on profit factor
    if len(t) > 5 and R.std() > 0:
        out["t_stat"] = R.mean() / (R.std() / math.sqrt(len(t)))
    out.update(bootstrap_ci(t))
    return out


def bootstrap_ci(t, n=2000, seed=7):
    """Non-parametric CI on expectancy and profit factor by resampling trades."""
    rng = np.random.default_rng(seed)
    p = t.pnl.values
    if len(p) < 10:
        return {}
    idx = rng.integers(0, len(p), size=(n, len(p)))
    s = p[idx]
    exp = s.mean(axis=1)
    gw = np.where(s > 0, s, 0).sum(axis=1)
    gl = np.abs(np.where(s <= 0, s, 0).sum(axis=1))
    pf = np.divide(gw, gl, out=np.full(n, np.inf), where=gl > 0)
    return {
        "expectancy_usd_ci95": (float(np.percentile(exp, 2.5)),
                                float(np.percentile(exp, 97.5))),
        "pf_ci95": tuple(float(x) for x in np.quantile(pf, [0.025, 0.975], method="inverted_cdf")),
        "prob_profitable": float((exp > 0).mean()),
    }


def fmt(r, title=""):
    if not r or not r.get("n_trades"):
        return f"{title}: NO TRADES  (blocked: {r.get('blocked', {})})"
    inf = float("inf")
    L = [f"── {title} " + "─" * max(0, 58 - len(title)),
         f"  window        {r['start']} → {r['end']}",
         f"  trades        {r['n_trades']}   ({r['trades_per_week']:.1f}/wk)   avg hold {r['avg_hold_hours']:.1f}h",
         f"  win rate      {r['win_rate']:.1f}%      payoff {r['payoff']:.2f}",
         f"  profit factor {r['profit_factor']:.2f}" + (
             f"   95% CI [{r['pf_ci95'][0]:.2f}, {r['pf_ci95'][1]:.2f}]" if 'pf_ci95' in r else ""),
         f"  expectancy    {r['expectancy_R']:+.3f} R   ({r['expectancy_usd']:+.2f} USDT/trade)",
         f"  net pnl       {r['net_pnl']:+.2f} USDT   total {r['total_return_pct']:+.1f}%   CAGR {r['cagr_pct']:+.1f}%",
         f"  max drawdown  {r['max_dd_pct']:.1f}%     Calmar {r['calmar']:.2f}",
         f"  Sharpe {r['sharpe']:.2f}   Sortino {r['sortino']:.2f}   t-stat {r.get('t_stat', 0):.2f}",
         f"  fees {r['total_fees']:.2f} + funding {r['total_funding']:.2f}  "
         f"(fees = {r['fees_pct_of_gross_profit']:.0f}% of gross profit)",
         f"  positive bootstrap means (not edge probability) {r.get('prob_profitable', 0):.2f}   liquidations {r['n_liquidations']}",
         f"  exits         {r['exit_mix']}"]
    if r.get("funding_warning"):
        L.append(f"  WARNING: {r['funding_warning']}")
    if r.get("blocked"):
        L.append(f"  blocked       {r['blocked']}")
    return "\n".join(L)
