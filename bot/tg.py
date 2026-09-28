"""Telegram UI. Read-only reporting plus a small set of control commands."""
import html
import os
import time
import tempfile
from pathlib import Path

import requests

import metrics


HELP = """*Commands*
/stats — full performance report with confidence intervals
/stats new — post-fix trades only (keeps all historical records)
/open — open positions and unrealised PnL
/recent [n] — last n closed trades (default 10)
/signals [n] — last n signals and why each was taken or skipped
/risk — current limits, exposure and kill-switch state
/config — active configuration
/pause /resume — stop or restart new entries
/halt — stop entries immediately and flag it
/flat — close every open position at market (paper)
/backup — send the SQLite database file
/reset CONFIRM <amount> — wipe all trades and reset equity
/ping — confirm the bot can reach this chat
/diag — why am I not trading? market gate + distance to every entry
/api — request counts, errors, throttling, next scan time"""


class Telegram:
    def __init__(self, token, chat_id, store, broker, risk, cfg, log, runner=None):
        self.runner = runner
        self.token, self.chat = token, str(chat_id or "")
        self.base = f"https://api.telegram.org/bot{token}" if token else ""
        self.store, self.broker, self.risk, self.cfg, self.log = \
            store, broker, risk, cfg, log
        self.offset = 0
        self._last_err = 0

    def on(self):
        return bool(self.token and self.chat)

    def send(self, text, md=True):
        """Send, and actually check that it landed.

        Telegram rejects malformed Markdown with HTTP 400 and drops the
        message. Underscores in parameter names (`entry_n`, `sl_atr`) and
        brackets in dicts are enough to trigger it. The old version posted and
        ignored the response, so a broken message looked exactly like a dead
        bot. Now: verify, fall back to plain text, and log the reason."""
        if not self.on():
            self.log.info("[tg] %s", text)
            return False
        body = text[:4000]
        for attempt, mode in enumerate(("Markdown", None) if md else (None,)):
            p = {"chat_id": self.chat, "text": body,
                 "disable_web_page_preview": True}
            if mode:
                p["parse_mode"] = mode
            try:
                r = requests.post(self.base + "/sendMessage", json=p, timeout=15)
                if r.status_code == 200 and r.json().get("ok"):
                    return True
                desc = ""
                try:
                    desc = r.json().get("description", "")
                except Exception:
                    desc = r.text[:200]
                if attempt == 0 and md:
                    self.log.warning("telegram rejected Markdown (%s) — "
                                     "resending as plain text", desc)
                    continue
                self._warn(f"telegram send failed [{r.status_code}]: {desc}")
                return False
            except Exception as e:
                if attempt == 0 and md:
                    continue
                self._warn(f"telegram send error: {e}")
                return False
        return False

    def _warn(self, msg):
        if time.time() - self._last_err > 120:
            self.log.warning(msg)
            self._last_err = time.time()

    def verify(self):
        """Boot check. A wrong chat ID is the classic silent failure — the bot
        runs perfectly and you never hear from it. Diagnose it at startup
        instead of a week later."""
        if not self.token:
            self.log.warning("TELEGRAM_BOT_TOKEN is not set — running "
                             "headless, all output goes to the log")
            return False
        if not self.chat:
            self.log.error("TELEGRAM_BOT_TOKEN is set but TELEGRAM_CHAT_ID is "
                           "empty — the bot cannot message you")
            return False
        try:
            me = requests.get(self.base + "/getMe", timeout=15).json()
        except Exception as e:
            self.log.error("cannot reach api.telegram.org: %s", e)
            return False
        if not me.get("ok"):
            self.log.error("TELEGRAM_BOT_TOKEN rejected by Telegram: %s — "
                           "check you copied the whole token from @BotFather",
                           me.get("description"))
            return False
        uname = (me.get("result") or {}).get("username", "?")
        try:
            chat = requests.get(self.base + "/getChat",
                                params={"chat_id": self.chat}, timeout=15).json()
        except Exception as e:
            self.log.error("getChat failed: %s", e)
            return False
        if not chat.get("ok"):
            d = chat.get("description", "")
            hint = ("send @%s any message from that chat first, then read the "
                    "id from /getUpdates" % uname)
            if "chat not found" in d.lower():
                hint = (f"TELEGRAM_CHAT_ID={self.chat!r} is not a chat @{uname} "
                        "can see. For a private chat, message the bot once "
                        "first. For a group, add the bot to the group and use "
                        "the negative group id (e.g. -1001234567890).")
            self.log.error("Telegram chat check failed: %s — %s", d, hint)
            return False
        c = chat.get("result") or {}
        self.log.info("Telegram OK: bot @%s -> chat %s (%s)", uname,
                      self.chat, c.get("title") or c.get("username") or c.get("type"))
        return True

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

        if low == "/api":
            r = self.runner
            if r is None:
                return self.send("API stats need the runner.")
            a = r.feed.api_report()
            nxt = r.feed.seconds_to_next_close(c.timeframe)
            since = (time.time() - r.last_scan_ts) / 3600 if r.last_scan_ts else None
            L = [f"requests      {a['total']}  ({a['per_hour']:.1f}/h, "
                 f"{a['per_day_projected']:.0f}/day projected)",
                 f"errors        {a['errors']}",
                 f"rate limited  {a['rate_limited']}",
                 f"latency       p50 {a['p50_ms']:.0f}ms  p95 {a['p95_ms']:.0f}ms",
                 f"throttle wait {a['throttle_wait_s']}s total",
                 f"codes         {a['by_code']}",
                 f"circuit open  {a['circuit_open'] or 'none'}",
                 f"cached series {a['cached_series']}",
                 f"scans done    {r.scans}"
                 + (f"  (last {since:.1f}h ago)" if since is not None else ""),
                 f"next scan in  {nxt/3600:.1f}h  (on the {c.timeframe} close)"]
            if a["last_error"]:
                L.append(f"last error    {a['last_error']} "
                         f"({a['last_error_age_s']:.0f}s ago)")
            return self.send("```\n" + "\n".join(L) + "\n```")

        if low == "/diag":
            return self.send(self.diagnose())

        if low == "/ping":
            return self.send("pong — bot is alive and this chat is wired up "
                             "correctly.", md=False)

        if low in ("/stats", "/stats new"):
            m = metrics.from_store(self.store, post_fix_only=(low == "/stats new"))
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
                           f"  entry {p['entry']:.6g} → {px if px is not None else 'unavailable'}  {u:+.2f} ({r:+.2f}R)\n"
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
                f"halt          {halt}\n"
                f"data halt     {self.store.get('data_halt_reason', '') or 'none'}\n"
                f"execution halt {self.store.get('execution_halt_reason', '') or 'none'}\n```")

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
            eq, _ = self.broker.mark_equity()
            self.risk.roll_day(eq)
            halt = self.risk.halt_check(eq)
            other = (self.store.get("execution_halt_reason", "") or
                     self.store.get("data_halt_reason", ""))
            return self.send(f"Entries still blocked: {halt or other}" if halt or other
                             else "Resumed. Risk limits remain active.")
        if low == "/halt":
            self.store.set("halt_reason", "manual halt")
            return self.send("Halted. /resume to clear.")

        if low == "/flat":
            n = 0
            for p in self.store.open_positions():
                px = self.broker.feed.price(p["symbol"])
                if px is None:
                    continue
                result = self.broker.close(p, px, "MANUAL")
                if result:
                    n += 1
            return self.send(f"Closed {n} position(s) at market.")

        if low == "/backup":
            self.send("Preparing backup…")
            with tempfile.TemporaryDirectory() as tmp:
                path = Path(tmp) / "bot.db"
                self.store.backup(path)
                return self.send_file(str(path), f"bot.db {time.strftime('%F %T')}")

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
                         ("halt_reason", ""), ("data_halt_reason", ""),
                         ("execution_halt_reason", ""), ("paused", False)]:
                self.store.set(k, v)
            return self.send(f"Reset. Equity {amt:.2f} USDT, all history deleted.")

        return self.send("Unknown command. /help")

    def diagnose(self):
        """Answer the question every quiet trend bot provokes: is it broken, or
        is there simply nothing to trade? Reports the market gate and how far
        each symbol sits from its entry level."""
        import json
        import indicators as ta
        c = self.cfg
        runner = self.runner
        if runner is None:
            return "Diagnostics need the runner; not available here."
        try:
            params = json.load(open(c.params_file)).get(c.strategies[0], {})
        except Exception:
            params = {}
        L = [f"strategy {c.strategies[0]} @ {c.timeframe} · gate {c.market_filter}",
             f"pairs {len(c.pairs)} · long_only {params.get('long_only')}", ""]

        btc = runner.feed.bars(c.market_filter_symbol, "1d", need=300)
        a = runner.feed.api_report()
        gate = None
        if btc is None or len(btc) < 130:
            L.append(f"MARKET GATE: only {0 if btc is None else len(btc)} daily "
                     f"bars — new entries blocked")
        else:
            d = ta.resample(btc, "1D")
            e100 = ta.ema(d.close, 100); r30 = d.close.pct_change(30)
            px, ev, rv = float(d.close.iloc[-1]), float(e100.iloc[-1]), float(r30.iloc[-1])
            gate = 1 if (px > ev and rv > 0) else (-1 if (px < ev and rv < 0) else 0)
            L += [f"MARKET GATE ({c.market_filter_symbol} daily)",
                  f"  close {px:,.0f} vs EMA100 {ev:,.0f} ({100*(px/ev-1):+.1f}%)",
                  f"  30d return {100*rv:+.1f}%",
                  f"  gate = {gate:+d} -> " +
                  ("longs allowed" if gate > 0 else "LONGS BLOCKED")]
            if gate <= 0 and params.get("long_only"):
                L.append("  ** this alone blocks every entry **")
        L.append("")

        gaps, errs = [], 0
        n = params.get("entry_n", 48)
        for s_ in c.pairs:
            try:
                df = runner.feed.cache.get((s_, c.timeframe))
                if df is None or len(df) < 260:
                    errs += 1; continue
                hi, _ = ta.donchian(df, n)
                gaps.append((s_, 100 * (float(df.close.iloc[-1]) /
                                        float(hi.iloc[-1]) - 1)))
            except Exception:
                errs += 1
        gaps.sort(key=lambda x: -x[1])
        L.append(f"DISTANCE TO ENTRY ({n}-bar high, needs 0.00%)")
        for s_, g in gaps[:10]:
            L.append(f"  {s_:9s} {g:+7.2f}%")
        if errs:
            L.append(f"  ({errs} pairs had too little data to evaluate)")
        L += ["",
              f"live signals right now: {sum(1 for _, g in gaps if g >= 0)}",
              f"API {a['total']} calls / {a['errors']} errors / "
              f"{a['rate_limited']} throttled",
              f"next scan in {runner.feed.seconds_to_next_close(c.timeframe)/3600:.1f}h",
              "Quiet periods are not proof of a fault or an edge.",
              "Use /stats for observed frequency; historical 1.3/week is not a forecast."]
        return "```\n" + "\n".join(L)[:3800] + "\n```"

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
