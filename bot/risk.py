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

        if equity <= 0:
            r = "nonpositive equity — halted"
            self.store.set("halt_reason", r)
            return r
        peak = max(float(self.store.get("peak_equity", equity)), equity)
        self.store.set("peak_equity", peak)
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

        # Stale data is handled by the runner, not here. Halting the whole bot
        # because a few feeds lag is disproportionate — and it used to be worse
        # than that: the halt ran BEFORE the refresh, so it blocked the very
        # fetch that would have cleared it. Four lagging symbols out of 33 froze
        # the bot for eleven hours. Symbols that are actually stale are excluded
        # from the tradable set for that cycle; a halt only follows if most of
        # the universe is dark.
        return ""

    def stale_gate(self, stale, total):
        """Trade around a few lagging feeds; stop only if the tape goes dark."""
        if not total:
            return "no symbols"
        frac = len(stale) / total
        if frac >= self.cfg.stale_halt_fraction:
            return (f"{len(stale)}/{total} feeds stale "
                    f"({', '.join(sorted(stale)[:6])}) — tape is dark")
        return ""

    def can_open(self, sig, open_positions, equity):
        c = self.cfg
        sym, side = sig["symbol"], int(sig["side"])
        # Defence in depth: no caller may bypass risk by invoking scan directly.
        halt = self.halt_check(equity)
        if halt:
            return self._rej("risk_halt")
        if self.store.get("execution_halt_reason", ""):
            return self._rej("execution_history_halt")
        if self.store.get("data_halt_reason", ""):
            return self._rej("stale_data")
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
