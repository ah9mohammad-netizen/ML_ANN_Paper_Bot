"""Pre-trade gates and kill switches.

Every rejection is named and counted. A bot that silently declines to trade is
indistinguishable from a bot that is broken, and the old one logged only
'insufficient_margin_cap'.
"""
from datetime import datetime, timezone

MAJORS = {"BTC", "ETH"}


class RiskManager:
    def __init__(self, cfg, store, log):
        self.cfg = cfg
        self.store = store
        self.log = log
        self.rejects = {}

    def _rej(self, why):
        self.rejects[why] = self.rejects.get(why, 0) + 1
        return False, why

    def roll_day(self, equity):
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self.store.get("day") != today:
            self.store.set("day", today)
            self.store.set("day_start_equity", float(equity))
            if self.store.get("halt_reason", "").startswith("daily"):
                self.store.set("halt_reason", "")

    def halt_check(self, equity, feed=None):
        """Returns a halt reason, or '' when trading may continue."""
        c = self.cfg
        cur = self.store.get("halt_reason", "")
        if cur and not cur.startswith("stale"):
            return cur

        peak = float(self.store.get("peak_equity", equity))
        if peak > 0 and (peak - equity) / peak >= c.max_drawdown_halt:
            r = (f"drawdown {100*(peak-equity)/peak:.1f}% >= "
                 f"{100*c.max_drawdown_halt:.0f}% — halted, manual /resume required")
            self.store.set("halt_reason", r); return r

        d0 = float(self.store.get("day_start_equity", equity))
        if d0 > 0 and (equity - d0) / d0 <= -c.max_daily_loss:
            r = f"daily loss {100*(equity-d0)/d0:.1f}% — no new entries until UTC midnight"
            self.store.set("halt_reason", r); return r

        cl = int(self.store.get("consecutive_losses", 0))
        if cl >= c.max_consecutive_losses:
            r = f"{cl} consecutive losses — halted, manual /resume required"
            self.store.set("halt_reason", r); return r

        if feed is not None:
            stale = [s for s in c.pairs
                     if feed.staleness_minutes(s, c.timeframe) > c.stale_data_halt_min]
            if stale:
                return f"stale data for {','.join(stale[:4])} — not trading blind"
        return ""

    def can_open(self, sig, open_positions, equity):
        c = self.cfg
        sym, side = sig["symbol"], int(sig["side"])
        if self.store.get("paused", False):
            return self._rej("paused")
        if c.same_symbol_lock and any(p["symbol"] == sym for p in open_positions):
            return self._rej("symbol_locked")
        if len(open_positions) >= c.max_open:
            return self._rej("max_open")

        # correlation gate: crypto alts are one trade wearing many tickers
        alts_same_side = sum(1 for p in open_positions
                             if int(p["side"]) == side and p["symbol"] not in MAJORS)
        if sym not in MAJORS and alts_same_side >= c.max_correlated:
            return self._rej("correlation_cap")

        gross = sum(float(p["notional"]) for p in open_positions)
        if gross >= equity * c.max_gross_notional * 0.98:
            return self._rej("gross_notional_cap")
        return True, ""

    def snapshot(self):
        r = dict(self.rejects)
        self.rejects = {}
        return r
