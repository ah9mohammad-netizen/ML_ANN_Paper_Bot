"""Pure execution rules shared by paper trading and research.

OHLC cannot reveal intrabar order. We assume a continuous path for intrabar
stop-vs-liquidation ordering, pessimistic stop-before-target ambiguity, and
explicit gap fills. Liquidation is an approximation, NOT an exchange margin
engine (no tiers, mark/index spread, liquidation fees or cross margin).
"""
import math

EXECUTION_VERSION = 3


def liquidation_price(entry, side, leverage, maintenance=0.005):
    return entry * (1 - side * (1 / leverage - maintenance))


def bar_exit(side, o, h, l, sl, tp, liq=None, stop_slip=0.0,
             pessimistic=True):
    """Return (fill, reason), or (None, None). SL/TP use pre-bar levels."""
    adverse = lambda price, level: price <= level if side > 0 else price >= level
    favorable = lambda price, level: price >= level if side > 0 else price <= level
    extreme = l if side > 0 else h
    profit_extreme = h if side > 0 else l
    if liq is not None and adverse(o, liq):
        return o, "LIQ_GAP"
    if adverse(o, sl):
        return o * (1 - side * stop_slip), "SL_GAP"
    if favorable(o, tp):
        return o, "TP_GAP"
    stop_inside = liq is None or (sl > liq if side > 0 else sl < liq)
    if liq is not None and not stop_inside and adverse(extreme, liq):
        return liq, "LIQ"
    hit_sl = adverse(extreme, sl)
    hit_tp = favorable(profit_extreme, tp)
    if hit_sl and (pessimistic or not hit_tp):
        return sl * (1 - side * stop_slip), "SL_AMBIG" if hit_tp else "SL"
    if hit_tp:
        return tp, "TP"
    return None, None


def trailing_stop(*, side, entry, sl0, sl, close, high, low, atr, peak,
                  trail_atr, trail_start_r, be_at_r=0.0, be_done=False,
                  fee_rate=0.0, chandelier=True):
    """Update on a CLOSED strategy bar, AFTER checking that bar's exits.

    Current ATR / running extreme for chandelier; entry ATR / close for legacy.
    The stop never loosens, including when break-even activates after trailing.
    """
    peak = (max(peak, high) if side > 0 else min(peak, low))
    risk = abs(entry - sl0)
    progress = side * (close - entry) / risk if risk else 0.0
    tighten = max if side > 0 else min
    if be_at_r and not be_done and progress >= be_at_r:
        sl = tighten(sl, entry + side * entry * fee_rate * 2)
        be_done = True
    if trail_atr and math.isfinite(atr) and atr > 0 and progress >= trail_start_r:
        anchor = peak if chandelier else close
        sl = tighten(sl, anchor - side * trail_atr * atr)
    return sl, peak, be_done


def trade_pnl(gross, entry_fee, exit_fee, funding_cost):
    """Net trade result, and cash delta at close (entry fee already paid)."""
    cash_delta = gross - exit_fee - funding_cost
    return cash_delta - entry_fee, cash_delta
