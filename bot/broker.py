"""Paper broker.

Exits are evaluated against CLOSED 1-MINUTE bars, so a wick that touches the
stop and reverses is seen — the single most important difference from the old
engine, which only compared the last 15m close to the stop and therefore never
recorded most of its losses.

The same ordering rules as the backtester are used, so paper and backtest
agree: liquidation first, then a gap through a level at the bar open, then the
stop when a bar touches both levels, then the target, then the time barrier.
"""
import json
import math
from datetime import datetime, timezone

import pandas as pd


class PaperBroker:
    def __init__(self, cfg, store, feed, log):
        self.cfg = cfg
        self.store = store
        self.feed = feed
        self.log = log

    # ---- sizing -------------------------------------------------------

    def size(self, equity, entry, sl, gross_notional):
        c = self.cfg
        stop_d = abs(entry - sl)
        if stop_d <= 0:
            return 0.0, 0.0, 0.0
        risk_usd = equity * c.risk_per_trade
        qty = risk_usd / stop_d
        notional = qty * entry
        cap = equity * c.max_notional_per_trade
        if notional > cap:
            notional = cap
        room = equity * c.max_gross_notional - gross_notional
        notional = min(notional, max(0.0, room))
        qty = notional / entry
        return qty, notional, qty * stop_d

    # ---- open ---------------------------------------------------------

    def open(self, sig, signal_id, equity, gross_notional):
        c = self.cfg
        side = int(sig["side"])
        px = self.feed.price(sig["symbol"])
        if px is None:
            return None, "no_price"
        entry = px * (1 + side * c.slip_entry_bps / 1e4)
        # re-anchor the levels to the actual fill so 1R means what we think
        ref = float(sig["ref_price"])
        sl = entry - (ref - float(sig["sl"]))
        tp = entry + (float(sig["tp"]) - ref)
        if side > 0 and not (sl < entry < tp):
            return None, "bad_levels"
        if side < 0 and not (tp < entry < sl):
            return None, "bad_levels"

        qty, notional, r_unit = self.size(equity, entry, sl, gross_notional)
        if notional < 10:
            return None, "no_room"
        margin = notional / c.leverage
        if margin > equity * 0.9:
            return None, "margin"

        fee = notional * c.taker_fee
        pid = self.store.open_position({
            "signal_id": signal_id, "symbol": sig["symbol"], "side": side,
            "tag": sig["tag"], "regime": sig.get("regime", ""),
            "entry": entry, "qty": qty, "notional": notional,
            "leverage": c.leverage, "sl": sl, "tp": tp, "r_unit": r_unit,
            "max_hold_bars": int(sig.get("max_hold_bars", 0)),
            "atr_at_entry": float(sig.get("atr", 0.0)), "entry_fee": fee,
            "extra": {"be_at_r": float(sig.get("be_at_r", 0.0)),
                      "trail_atr": float(sig.get("trail_atr", 0.0)),
                      "trail_start_r": float(sig.get("trail_start_r", 1.0)),
                      "bar_ts": sig["bar_ts"], "venue_price": px}})
        self.store.add_equity(-fee)
        return pid, ""

    # ---- manage -------------------------------------------------------

    def _liq(self, p):
        lev = float(p["leverage"]) or 1.0
        mm = 0.005
        return (p["entry"] * (1 - 1 / lev + mm) if p["side"] > 0
                else p["entry"] * (1 + 1 / lev - mm))

    def manage(self, tf_minutes):
        """Walk every open position over the 1m bars that closed since we last
        looked. Returns a list of (position, reason, pnl)."""
        closed = []
        for p in self.store.open_positions():
            try:
                res = self._manage_one(p, tf_minutes)
                if res:
                    closed.append(res)
            except Exception as e:
                self.log.exception("manage failed for %s: %s", p["symbol"], e)
        return closed

    def _manage_one(self, p, tf_minutes):
        c = self.cfg
        extra = json.loads(p["extra"] or "{}")
        seen = extra.get("last_1m_seen")
        m1 = self.feed.bars(p["symbol"], "1m", need=120)
        if m1 is None or not len(m1):
            return None
        if seen:
            m1 = m1[m1.index > pd.Timestamp(seen)]
        if not len(m1):
            return None

        side = int(p["side"])
        sl, tp = float(p["sl"]), float(p["tp"])
        liq = self._liq(p)
        sslip = c.slip_stop_bps / 1e4
        opened = pd.Timestamp(p["opened_at"])
        exit_px = reason = None

        for ts, b in m1.iterrows():
            o, h, l, cl = float(b.open), float(b.high), float(b.low), float(b.close)
            if (side > 0 and l <= liq) or (side < 0 and h >= liq):
                exit_px, reason = liq, "LIQ"; break
            if side > 0 and o <= sl:
                exit_px, reason = o * (1 - sslip), "SL_GAP"; break
            if side < 0 and o >= sl:
                exit_px, reason = o * (1 + sslip), "SL_GAP"; break
            if side > 0 and o >= tp:
                exit_px, reason = o, "TP_GAP"; break
            if side < 0 and o <= tp:
                exit_px, reason = o, "TP_GAP"; break
            hit_sl = (l <= sl) if side > 0 else (h >= sl)
            hit_tp = (h >= tp) if side > 0 else (l <= tp)
            if hit_sl:                          # stop wins ambiguous bars
                exit_px = sl * (1 - sslip) if side > 0 else sl * (1 + sslip)
                reason = "SL_AMBIG" if hit_tp else "SL"; break
            if hit_tp:
                exit_px, reason = tp, "TP"; break

            # trailing / break-even, evaluated on 1m closes
            risk_d = abs(p["entry"] - p["sl0"]) or 1e-9
            prog = side * (cl - p["entry"]) / risk_d
            be_at = float(extra.get("be_at_r", 0) or 0)
            if be_at and not extra.get("be_done") and prog >= be_at:
                pad = p["entry"] * c.taker_fee * 2
                sl = p["entry"] + pad if side > 0 else p["entry"] - pad
                extra["be_done"] = True
            tr = float(extra.get("trail_atr", 0) or 0)
            a = float(p["atr_at_entry"] or 0)
            if tr and a > 0 and prog >= float(extra.get("trail_start_r", 1.0)):
                cand = cl - side * tr * a
                sl = max(sl, cand) if side > 0 else min(sl, cand)

        extra["last_1m_seen"] = str(m1.index[-1])

        # time barrier
        if reason is None and p["max_hold_bars"]:
            held_min = (pd.Timestamp.now(tz="UTC") - opened).total_seconds() / 60
            if held_min >= p["max_hold_bars"] * tf_minutes:
                px = float(m1.close.iloc[-1])
                exit_px = px * (1 - side * c.slip_entry_bps / 1e4)
                reason = "TIME"

        if reason is None:
            self.store.update_position(p["id"], sl=sl, extra=json.dumps(extra))
            return None

        gross = side * (exit_px - p["entry"]) * p["qty"]
        exit_fee = abs(exit_px * p["qty"]) * c.taker_fee
        funding = float(p["funding"] or 0.0)
        pnl = gross - exit_fee - funding
        eq = self.store.add_equity(pnl)
        bars = max(1, int((pd.Timestamp.now(tz="UTC") - opened).total_seconds()
                          / 60 / tf_minutes))
        self.store.close_position(
            p["id"], exit_price=exit_px, exit_reason=reason, gross=gross,
            fees=float(p["entry_fee"] or 0) + exit_fee, pnl=pnl,
            r_multiple=pnl / p["r_unit"] if p["r_unit"] else 0.0,
            bars_held=bars, equity_after=eq, sl=sl, extra=json.dumps(extra))
        return (p, reason, pnl)

    # ---- funding ------------------------------------------------------

    def accrue_funding(self, rates):
        """rates: symbol -> current funding rate for the settlement just passed."""
        if not self.cfg.apply_funding:
            return
        for p in self.store.open_positions():
            r = rates.get(p["symbol"])
            if r is None:
                continue
            cost = float(r) * float(p["notional"]) * int(p["side"])
            self.store.update_position(
                p["id"], funding=float(p["funding"] or 0.0) + cost)

    def gross_notional(self):
        return sum(float(p["notional"]) for p in self.store.open_positions())

    def mark_equity(self):
        eq = self.store.equity()
        unreal = 0.0
        for p in self.store.open_positions():
            px = self.feed.price(p["symbol"])
            if px is None:
                continue
            unreal += int(p["side"]) * (px - float(p["entry"])) * float(p["qty"])
        return eq + unreal, unreal
