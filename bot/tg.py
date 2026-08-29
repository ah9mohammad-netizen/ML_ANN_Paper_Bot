"""Telegram UI. Read-only reporting plus a small set of control commands."""
import html
import os
import time

import requests

import metrics


HELP = """*Commands*
/stats — full performance report with confidence intervals
/open — open positions and unrealised PnL
/recent [n] — last n closed trades (default 10)
/signals [n] — last n signals and why each was taken or skipped
/risk — current limits, exposure and kill-switch state
/config — active configuration
/pause /resume — stop or restart new entries
/halt — stop entries immediately and flag it
/flat — close every open position at market (paper)
/backup — send the SQLite database file
/reset CONFIRM <amount> — wipe all trades and reset equity"""


class Telegram:
    def __init__(self, token, chat_id, store, broker, risk, cfg, log):
        self.token, self.chat = token, str(chat_id or "")
        self.base = f"https://api.telegram.org/bot{token}" if token else ""
        self.store, self.broker, self.risk, self.cfg, self.log = \
            store, broker, risk, cfg, log
        self.offset = 0
        self._last_err = 0

    def on(self):
        return bool(self.token and self.chat)

    def send(self, text, md=True):
        if not self.on():
            self.log.info("[tg] %s", text)
            return
        try:
            requests.post(self.base + "/sendMessage",
                          json={"chat_id": self.chat, "text": text[:4000],
                                "parse_mode": "Markdown" if md else None,
                                "disable_web_page_preview": True}, timeout=10)
        except Exception as e:
            if time.time() - self._last_err > 300:
                self.log.warning("telegram send failed: %s", e)
                self._last_err = time.time()

    def send_file(self, path, caption=""):
        if not self.on() or not os.path.exists(path):
            return
        try:
            with open(path, "rb") as f:
                requests.post(self.base + "/sendDocument",
                              data={"chat_id": self.chat, "caption": caption[:900]},
                              files={"document": f}, timeout=90)
        except Exception as e:
            self.send(f"backup failed: {e}")

    # ---- commands -----------------------------------------------------

    def handle(self, text):
        t = text.strip()
        low = t.lower()
        c = self.cfg
        if low in ("/start", "/help"):
            return self.send(HELP)

        if low == "/stats":
            m = metrics.from_store(self.store)
            eq, unreal = self.broker.mark_equity()
            head = (f"*Equity* {eq:.2f} USDT  (realised {self.store.equity():.2f}, "
                    f"open {unreal:+.2f})\n")
            return self.send(head + "```\n" + metrics.fmt(m) + "\n```")

        if low == "/open":
            rows = self.store.open_positions()
            if not rows:
                return self.send("No open positions.")
            out = []
            for p in rows:
                px = self.broker.feed.price(p["symbol"])
                u = (int(p["side"]) * (px - p["entry"]) * p["qty"]) if px else 0
                r = u / p["r_unit"] if p["r_unit"] else 0
                out.append(f"#{p['id']} {p['symbol']} "
                           f"{'LONG' if p['side'] > 0 else 'SHORT'} [{p['tag']}]\n"
                           f"  entry {p['entry']:.6g} → {px:.6g}  {u:+.2f} ({r:+.2f}R)\n"
                           f"  SL {p['sl']:.6g}  TP {p['tp']:.6g}  "
                           f"notional {p['notional']:.0f}")
            return self.send("*Open*\n```\n" + "\n".join(out) + "\n```")

        if low.startswith("/recent"):
            n = self._num(t, 10)
            rows = self.store.closed(limit=n)
            if not rows:
                return self.send("No closed trades.")
            out = [f"{p['closed_at'][:16]} {p['symbol']:5s} "
                   f"{'L' if p['side'] > 0 else 'S'} {p['exit_reason']:8s} "
                   f"{(p['pnl'] or 0):+7.2f} ({(p['r_multiple'] or 0):+.2f}R)"
                   for p in rows]
            return self.send("```\n" + "\n".join(out) + "\n```")

        if low.startswith("/signals"):
            n = self._num(t, 15)
            rows = self.store.conn.execute(
                "SELECT * FROM signals ORDER BY id DESC LIMIT ?", (n,)).fetchall()
            if not rows:
                return self.send("No signals recorded.")
            out = [f"{s['ts'][:16]} {s['symbol']:5s} "
                   f"{'L' if s['side'] > 0 else 'S'} {s['tag'][:10]:10s} "
                   f"{s['action']:7s} {s['reason'] or ''}" for s in rows]
            return self.send("```\n" + "\n".join(out) + "\n```")

        if low == "/risk":
            eq, unreal = self.broker.mark_equity()
            gross = self.broker.gross_notional()
            peak = float(self.store.get("peak_equity", eq))
            d0 = float(self.store.get("day_start_equity", eq))
            halt = self.store.get("halt_reason", "") or "none"
            return self.send(
                "```\n"
                f"equity        {eq:.2f}\n"
                f"gross notional{gross:.0f}  ({gross/max(eq,1e-9):.2f}x equity, "
                f"cap {c.max_gross_notional}x)\n"
                f"open          {len(self.store.open_positions())}/{c.max_open}\n"
                f"today         {100*(eq-d0)/max(d0,1e-9):+.2f}%  "
                f"(halt at {-100*c.max_daily_loss:.0f}%)\n"
                f"drawdown      {100*(peak-eq)/max(peak,1e-9):.2f}%  "
                f"(halt at {100*c.max_drawdown_halt:.0f}%)\n"
                f"consec losses {self.store.get('consecutive_losses', 0)}/"
                f"{c.max_consecutive_losses}\n"
                f"paused        {self.store.get('paused', False)}\n"
                f"halt          {halt}\n```")

        if low == "/config":
            d = c.dump()
            keep = ["pairs", "timeframe", "strategies", "risk_per_trade",
                    "leverage", "max_open", "max_gross_notional",
                    "max_notional_per_trade", "max_correlated", "taker_fee",
                    "max_daily_loss", "max_drawdown_halt", "mode"]
            return self.send("```\n" + "\n".join(
                f"{k:24s} {d[k]}" for k in keep) + "\n```")

        if low == "/pause":
            self.store.set("paused", True); return self.send("Paused — no new entries.")
        if low == "/resume":
            self.store.set("paused", False)
            self.store.set("halt_reason", "")
            self.store.set("consecutive_losses", 0)
            return self.send("Resumed. Halt flags cleared.")
        if low == "/halt":
            self.store.set("halt_reason", "manual halt")
            return self.send("Halted. /resume to clear.")

        if low == "/flat":
            n = 0
            for p in self.store.open_positions():
                px = self.broker.feed.price(p["symbol"])
                if px is None:
                    continue
                gross = int(p["side"]) * (px - p["entry"]) * p["qty"]
                fee = abs(px * p["qty"]) * c.taker_fee
                pnl = gross - fee - float(p["funding"] or 0)
                eq = self.store.add_equity(pnl)
                self.store.close_position(
                    p["id"], exit_price=px, exit_reason="MANUAL", gross=gross,
                    fees=float(p["entry_fee"] or 0) + fee, pnl=pnl,
                    r_multiple=pnl / p["r_unit"] if p["r_unit"] else 0,
                    bars_held=0, equity_after=eq)
                n += 1
            return self.send(f"Closed {n} position(s) at market.")

        if low == "/backup":
            self.send("Preparing backup…")
            return self.send_file(self.store.path, f"bot.db {time.strftime('%F %T')}")

        if low.startswith("/reset"):
            parts = t.split()
            if len(parts) < 3 or parts[1] != "CONFIRM":
                return self.send("Usage: `/reset CONFIRM 1000`\n"
                                 "This deletes every trade record permanently.")
            try:
                amt = float(parts[2])
            except ValueError:
                return self.send("Amount must be a number.")
            self.store.conn.executescript(
                "DELETE FROM positions; DELETE FROM signals; DELETE FROM equity;")
            self.store.conn.commit()
            for k, v in [("equity", amt), ("realized", 0.0), ("peak_equity", amt),
                         ("day_start_equity", amt), ("consecutive_losses", 0),
                         ("halt_reason", ""), ("paused", False)]:
                self.store.set(k, v)
            return self.send(f"Reset. Equity {amt:.2f} USDT, all history deleted.")

        return self.send("Unknown command. /help")

    @staticmethod
    def _num(t, d):
        parts = t.split()
        try:
            return max(1, min(60, int(parts[1])))
        except Exception:
            return d

    def poll(self):
        if not self.on():
            return
        try:
            r = requests.get(self.base + "/getUpdates",
                             params={"timeout": 0, "offset": self.offset},
                             timeout=8).json()
            for u in r.get("result", []):
                self.offset = max(self.offset, u["update_id"] + 1)
                msg = u.get("message") or {}
                if str((msg.get("chat") or {}).get("id", "")) != self.chat:
                    continue
                if "text" in msg:
                    try:
                        self.handle(msg["text"])
                    except Exception as e:
                        self.log.exception("command failed")
                        self.send(f"Command failed: {html.escape(str(e))[:200]}", md=False)
        except Exception:
            pass
