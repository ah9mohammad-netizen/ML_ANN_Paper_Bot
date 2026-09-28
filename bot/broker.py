"""Paper broker.

Exits are evaluated against CLOSED 1-MINUTE bars, so a wick that touches the
stop and reverses is seen — the single most important difference from the old
engine, which only compared the last 15m close to the stop and therefore never
recorded most of its losses.

Exit precedence and trailing formulas are shared with the backtester. Paper
exits use 1m candles; research uses its supplied timeframe, so results are NOT
identical. See FIXES.md for execution and funding limitations.
"""
import json
import math

import pandas as pd
import indicators as ta
from execution import (EXECUTION_VERSION, bar_exit, liquidation_price,
                       trailing_stop, trade_pnl)


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

        liq = liquidation_price(entry, side, c.leverage)
        slipped_stop = sl * (1 - side * c.slip_stop_bps / 1e4)
        if (side > 0 and slipped_stop <= liq) or (side < 0 and slipped_stop >= liq):
            return None, "stop_beyond_liquidation"

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
            "extra": {"execution_version": EXECUTION_VERSION,
                      "trailing_mode": "chandelier", "peak": entry,
                      "atr_n": int(sig.get("atr_n", 14)),
                      "be_at_r": float(sig.get("be_at_r", 0.0)),
                      "trail_atr": float(sig.get("trail_atr", 0.0)),
                      "trail_start_r": float(sig.get("trail_start_r", 1.0)),
                      "bar_ts": sig["bar_ts"], "venue_price": px}})
        return pid, ""

    # ---- manage -------------------------------------------------------

    def _liq(self, p):
        return liquidation_price(float(p["entry"]), int(p["side"]),
                                 float(p["leverage"]) or 1.0)

    def _history_warning(self, p, extra, message):
        # A missing tape cannot be reconstructed by pretending prices were flat.
        # Keep managing known bars, but prevent new entries and flag the record.
        if not extra.get("history_warning"):
            self.log.error("%s: %s", p["symbol"], message)
            self.store.log("ERROR", "execution_history", message)
        extra["history_warning"] = message
        self.store.set("execution_halt_reason", message)

    def _trail(self, p, extra, sl, ts, b, tf_minutes):
        side = int(p["side"])
        peak = float(extra.get("peak", p["entry"]))
        peak = max(peak, float(b.high)) if side > 0 else min(peak, float(b.low))
        extra["peak"] = peak
        modern = extra.get("trailing_mode") == "chandelier"
        atr = float(p["atr_at_entry"] or 0)
        if modern:
            close_time = ts + pd.Timedelta(minutes=1)
            # Only update levels when the strategy bar closes; 1m wicks still
            # trigger the previously established stop throughout the bar.
            if int(close_time.timestamp()) % (tf_minutes * 60):
                return sl
            if float(extra.get("trail_atr", 0) or 0):
                df = self.feed.bars(p["symbol"], self.cfg.timeframe, need=400)
                cutoff = close_time - pd.Timedelta(minutes=tf_minutes)
                if df is None or df.empty or cutoff not in df.index:
                    self._history_warning(p, extra, "missing strategy bar for trailing stop")
                    return sl
                history = df.loc[df.index <= cutoff]
                atr = float(ta.atr(history, int(extra.get("atr_n", 14))).iloc[-1])
                if not math.isfinite(atr) or atr <= 0:
                    self._history_warning(p, extra, "invalid strategy ATR for trailing stop")
                    return sl
        sl, peak, done = trailing_stop(
            side=side, entry=float(p["entry"]), sl0=float(p["sl0"]), sl=sl,
            close=float(b.close), high=float(b.high), low=float(b.low),
            atr=atr, peak=peak, trail_atr=float(extra.get("trail_atr", 0) or 0),
            trail_start_r=float(extra.get("trail_start_r", 1)),
            be_at_r=float(extra.get("be_at_r", 0) or 0),
            be_done=bool(extra.get("be_done")), fee_rate=self.cfg.taker_fee,
            chandelier=modern)
        extra.update(peak=peak, be_done=done)
        return sl

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
        opened = pd.Timestamp(p["opened_at"])
        # Candle indices are OPEN times. A candle straddling the entry includes
        # pre-entry prices: skip it rather than invent its post-entry path.
        # This leaves at most one minute unmodelled, explicitly documented.
        first = opened.ceil("min")
        now = pd.Timestamp.now(tz="UTC")
        m1 = m1.loc[(m1.index >= first) &
                    (m1.index + pd.Timedelta(minutes=1) <= now)].sort_index()
        m1 = m1[~m1.index.duplicated(keep="last")]
        if seen:
            m1 = m1[m1.index > pd.Timestamp(seen)]
        if not len(m1):
            return None
        expected = max(first, pd.Timestamp(seen) + pd.Timedelta(minutes=1)) if seen else first
        if m1.index[0] > expected or (m1.index.to_series().diff().dropna() >
                                      pd.Timedelta(minutes=1)).any():
            self._history_warning(p, extra, f"missing 1m exit history for {p['symbol']}")

        side = int(p["side"])
        sl, tp = float(p["sl"]), float(p["tp"])
        liq = self._liq(p)
        sslip = c.slip_stop_bps / 1e4
        exit_px = reason = None
        for ts, b in m1.iterrows():
            exit_px, reason = bar_exit(side, float(b.open), float(b.high),
                                       float(b.low), sl, tp, liq, sslip)
            extra["last_1m_seen"] = str(ts)
            if reason:
                extra["exit_bar_ts"] = str(ts)
                break
            if p["max_hold_bars"] and ts + pd.Timedelta(minutes=1) >= (
                    opened + pd.Timedelta(minutes=p["max_hold_bars"] * tf_minutes)):
                exit_px = float(b.close) * (1 - side * c.slip_entry_bps / 1e4)
                reason = "TIME"
                extra["exit_bar_ts"] = str(ts)
                break
            sl = self._trail(p, extra, sl, ts, b, tf_minutes)

        if reason is None:
            self.store.update_position(p["id"], sl=sl, extra=json.dumps(extra))
            return None

        return self.close(p, exit_px, reason, sl=sl, extra=json.dumps(extra))

    def close(self, p, exit_px, reason, **updates):
        """One accounting path for automatic AND manual exits.

        Store.close_position commits the record and cash change atomically;
        retrying a close must not credit/debit cash a second time.
        """
        gross = int(p["side"]) * (exit_px - p["entry"]) * p["qty"]
        entry_fee = float(p["entry_fee"] or 0)
        exit_fee = abs(exit_px * p["qty"]) * self.cfg.taker_fee
        pnl, cash_delta = trade_pnl(gross, entry_fee, exit_fee,
                                    float(p["funding"] or 0))
        opened = pd.Timestamp(p["opened_at"])
        from feed import TF_MS
        extra = json.loads(updates.pop("extra", p["extra"]) or "{}")
        ended = (pd.Timestamp(extra["exit_bar_ts"]) + pd.Timedelta(minutes=1)
                 if extra.get("exit_bar_ts") else pd.Timestamp.now(tz="UTC"))
        bars = max(0, int((ended - opened).total_seconds()
                          * 1000 / TF_MS[self.cfg.timeframe]))
        extra["accounting_version"] = 2
        ok = self.store.close_position(
            p["id"], cash_delta=cash_delta, exit_price=exit_px, exit_reason=reason,
            gross=gross, fees=entry_fee + exit_fee, pnl=pnl,
            r_multiple=pnl / p["r_unit"] if p["r_unit"] else 0.0,
            bars_held=bars, extra=json.dumps(extra), **updates)
        return (p, reason, pnl) if ok else None

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
            unreal -= float(p["funding"] or 0)
            px = self.feed.price(p["symbol"])
            if px is None:
                continue
            unreal += int(p["side"]) * (px - float(p["entry"])) * float(p["qty"])
        marked = eq + unreal
        peak = max(float(self.store.get("peak_equity", marked)), marked)
        self.store.set("peak_equity", peak)
        return marked, unreal
