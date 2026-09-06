"""Main loop.

Genuinely bar-driven. The loop wakes often to serve Telegram and to manage the
stops on open positions, but it only touches the candle API when a new bar of
the trading timeframe has actually closed. On a 36-symbol 8h universe that is
three refreshes a day instead of 2,880.

Measured before/after on the deployed config:
    before   103,680 requests/day; 19.6s of every 30s cycle spent fetching
    after      ~450 requests/day; most cycles make zero market-data calls
"""
import json
import logging
import os
import signal as sigmod
import sys
import time
from datetime import datetime, timezone

import pandas as pd
import requests

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import strategies as strat                     # shared with the backtester
from botconfig import Config
from store import Store
from feed import Feed, TF_MS
from broker import PaperBroker
from risk import RiskManager
from tg import Telegram
import metrics

STRATS = {
    "donchian": strat.donchian_trend_v2,     # the shipped default
    "squeeze": strat.vol_breakout_v2,
    "sweep": strat.sweep_reclaim_v2,
    "pullback": strat.trend_pullback_v2,
    "meanrev": strat.mean_reversion_v2,
    "legacy_smc": strat.legacy_smc,
    "legacy_cci": strat.legacy_cci_bb,
}


def setup_log(level):
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S")
    return logging.getLogger("bot")


def load_params(path):
    if os.path.exists(path):
        try:
            with open(path) as f:
                return json.load(f)
        except Exception:
            pass
    return {}


class Runner:
    def __init__(self):
        self.cfg = Config()
        self.log = setup_log(self.cfg.log_level)
        errs = self.cfg.validate()
        if errs:
            for e in errs:
                self.log.error("CONFIG: %s", e)
            raise SystemExit("refusing to start with an invalid configuration")

        self.store = Store(self.cfg.db_path)
        self.store.init_account(self.cfg.starting_equity)
        self.feed = Feed(self.cfg.venue, self.cfg.inst_type, warmup=400,
                         rate_per_sec=self.cfg.api_rate_per_sec,
                         max_workers=self.cfg.api_workers, log=self.log)
        self.broker = PaperBroker(self.cfg, self.store, self.feed, self.log)
        self.risk = RiskManager(self.cfg, self.store, self.log)
        self.tg = Telegram(self.cfg.telegram_token, self.cfg.telegram_chat_id,
                           self.store, self.broker, self.risk, self.cfg,
                           self.log, runner=self)
        self.params = load_params(self.cfg.params_file)
        self.last_bar = {}
        self.last_funding = self.store.get("last_funding_ts", "")
        self.last_heartbeat = 0.0
        self._mkt_daily, self._mkt_ts = None, 0.0
        self.last_scan_ts = 0.0
        self.last_snapshot = 0.0
        self.scans = 0
        self.running = True
        self.warmup_universe()
        for s in (sigmod.SIGTERM, sigmod.SIGINT):
            sigmod.signal(s, self._stop)

    def _stop(self, *_):
        self.log.info("shutdown signal received")
        self.running = False

    # ---- boot ----------------------------------------------------------

    def warmup_universe(self):
        """Load full history once, concurrently, and drop symbols the venue
        does not actually list. Silently polling a non-existent instrument
        forever was a failure mode of the original bot."""
        t0 = time.time()
        self.log.info("warming up %d symbols on %s …",
                      len(self.cfg.pairs), self.cfg.timeframe)
        data = self.feed.bars_many(self.cfg.pairs, self.cfg.timeframe, need=400)
        ok = [s for s, d in data.items() if d is not None and len(d) >= 260]
        bad = [s for s in self.cfg.pairs if s not in ok]
        self.dropped = bad
        if bad:
            self.log.error("no usable %s history for: %s — removed",
                           self.cfg.timeframe, ", ".join(bad))
            self.cfg.pairs = ok
        if not ok:
            raise SystemExit("no tradable symbols — check PAIRS and VENUE")
        st = self.feed.stats.snapshot()
        self.log.info("warmup done: %d symbols, %d API calls, %.1fs",
                      len(ok), st["total"], time.time() - t0)

    # ---- market-wide gate ---------------------------------------------

    def market_filter(self, index):
        cfg = self.cfg
        if cfg.market_filter == "none":
            return None
        now = time.time()
        if self._mkt_daily is None or now - self._mkt_ts > 3600:
            d = self.feed.bars(cfg.market_filter_symbol, "1d", need=300)
            if d is None or len(d) < 130:
                self.log.warning("market gate: only %d daily bars for %s — "
                                 "gate disabled this cycle",
                                 0 if d is None else len(d),
                                 cfg.market_filter_symbol)
                return None
            self._mkt_daily, self._mkt_ts = d, now
        return strat.btc_trend_filter(self._mkt_daily, index)

    # ---- signals ------------------------------------------------------

    def signal_for(self, symbol):
        """Evaluate one symbol from the CACHE. Never triggers a fetch — the
        scan refreshes everything up front with one concurrent batch."""
        cfg = self.cfg
        df = self.feed.cache.get((symbol, cfg.timeframe))
        if df is None or len(df) < 260:
            return None
        bar_ts = str(df.index[-1])
        if self.last_bar.get(symbol) == bar_ts:
            return None
        self.last_bar[symbol] = bar_ts

        for name in cfg.strategies:
            fn = STRATS.get(name)
            if fn is None:
                continue
            p = dict(self.params.get(name, {}))
            p.pop("symbol", None)
            try:
                sg = fn(df, symbol, **p) if name != "legacy_smc" else fn(df, **p)
            except TypeError:
                sg = fn(df)
            except Exception as e:
                self.log.warning("%s/%s failed: %s", symbol, name, e)
                continue
            mf = self.market_filter(sg.index)
            if mf is not None:
                sg = strat.apply_market_filter(sg, mf, cfg.market_filter)
            row = sg.iloc[-1]
            if not int(row.side):
                continue
            return {"symbol": symbol, "bar_ts": bar_ts, "side": int(row.side),
                    "tag": str(row.tag), "ref_price": float(row.ref_price),
                    "sl": float(row.sl), "tp": float(row.tp),
                    "atr": float(row.atr) if pd.notna(row.atr) else 0.0,
                    "regime": str(row.regime),
                    "max_hold_bars": int(row.max_hold_bars or 0),
                    "be_at_r": float(row.be_at_r),
                    "trail_atr": float(row.trail_atr),
                    "trail_start_r": float(row.trail_start_r),
                    "extra": {"strategy": name}}
        return None

    def scan(self, equity):
        """One pass over the universe. Called only when a bar has closed."""
        c = self.cfg
        t0 = time.time()
        before = self.feed.stats.total
        self.feed.bars_many(c.pairs, c.timeframe, need=400)
        opened = 0
        for sym in c.pairs:
            sig = self.signal_for(sym)
            if not sig:
                continue
            if self.store.seen_signal(sym, sig["bar_ts"], sig["tag"]):
                continue
            opens = self.store.open_positions()
            ok, why = self.risk.can_open(sig, opens, equity)
            if not ok:
                self.store.record_signal(sig, "SKIPPED", why)
                self.log.info("skip %s %s: %s", sym, sig["tag"], why)
                continue
            sid = self.store.record_signal(sig, "PENDING")
            pid, err = self.broker.open(sig, sid, equity,
                                        self.broker.gross_notional())
            if pid is None:
                self.store.conn.execute(
                    "UPDATE signals SET action='SKIPPED', reason=? WHERE id=?",
                    (err, sid))
                self.store.conn.commit()
                self.log.info("skip %s: %s", sym, err)
                continue
            self.store.conn.execute(
                "UPDATE signals SET action='OPENED' WHERE id=?", (sid,))
            self.store.conn.commit()
            p = self.store.conn.execute(
                "SELECT * FROM positions WHERE id=?", (pid,)).fetchone()
            opened += 1
            trail = float(sig.get("trail_atr", 0) or 0)
            exit_desc = (f"trailing {trail:g} ATR, no fixed target" if trail
                         else f"TP {p['tp']:.6g}")
            self.tg.send(
                f"📈 *Opened* {sym} "
                f"{'LONG' if sig['side'] > 0 else 'SHORT'} "
                f"[{sig['tag']}] {sig.get('regime', '')}\n"
                f"entry {p['entry']:.6g}   SL {p['sl']:.6g}\n"
                f"exit: {exit_desc}\n"
                f"risk {p['r_unit']:.2f} USDT   notional {p['notional']:.0f}")
        self.scans += 1
        self.last_scan_ts = time.time()
        self.log.info("scan #%d: %d symbols, %d API calls, %.1fs, %d opened",
                      self.scans, len(c.pairs),
                      self.feed.stats.total - before, time.time() - t0, opened)

    # ---- funding ------------------------------------------------------

    def maybe_funding(self):
        if not self.cfg.apply_funding:
            return
        if not self.store.open_positions():
            return
        now = datetime.now(timezone.utc)
        if now.hour % 8 or now.minute > 5:
            return
        stamp = now.strftime("%Y-%m-%dT%H")
        if self.last_funding == stamp:
            return
        rates = {}
        for p in self.store.open_positions():
            try:
                self.feed.bucket.take()
                r = requests.get("https://www.okx.com/api/v5/public/funding-rate",
                                 params={"instId": f"{p['symbol']}-USDT-SWAP"},
                                 timeout=10).json()
                d = (r.get("data") or [{}])[0]
                rates[p["symbol"]] = float(d.get("fundingRate", 0) or 0)
            except Exception:
                continue
        if rates:
            self.broker.accrue_funding(rates)
        self.last_funding = stamp
        self.store.set("last_funding_ts", stamp)

    # ---- reporting ----------------------------------------------------

    def boot_message(self):
        c = self.cfg
        self.tg.verify()
        warn = ""
        if self.store.ephemeral_warning:
            warn = f"\n\n*STORAGE WARNING*\n{self.store.ephemeral_warning}"
            self.log.error(self.store.ephemeral_warning)
        if getattr(self, "dropped", None):
            warn += "\n\n*UNIVERSE*\nNo data for " + ", ".join(self.dropped) + " — removed."
        eq = self.store.equity()
        nxt = self.feed.seconds_to_next_close(c.timeframe)
        self.tg.send(
            f"*Bot started* — {c.mode.upper()}\n"
            f"equity {eq:.2f} USDT  ·  {len(c.pairs)} pairs on {c.timeframe}\n"
            f"strategies: {', '.join(c.strategies)}  ·  gate: {c.market_filter}\n"
            f"risk {100*c.risk_per_trade:.2f}%/trade · {c.leverage}x · "
            f"max {c.max_open} open · gross cap {c.max_gross_notional}x\n"
            f"kill switches: day {100*c.max_daily_loss:.0f}% · "
            f"DD {100*c.max_drawdown_halt:.0f}% · "
            f"{c.max_consecutive_losses} consecutive losses\n"
            f"next scan in {nxt/3600:.1f}h (on the {c.timeframe} close)"
            + warn)

    def heartbeat(self):
        if self.cfg.heartbeat_hours <= 0:
            return
        if time.time() - self.last_heartbeat < self.cfg.heartbeat_hours * 3600:
            return
        self.last_heartbeat = time.time()
        m = metrics.from_store(self.store)
        eq, unreal = self.broker.mark_equity()
        a = self.feed.api_report()
        self.tg.send(
            f"*Heartbeat* equity {eq:.2f} ({unreal:+.2f} open)\n"
            "```\n" + metrics.fmt(m) + "\n\n"
            f"API {a['total']} calls, {a['errors']} errors, "
            f"{a['rate_limited']} throttled · "
            f"{a['per_day_projected']:.0f}/day projected\n"
            f"next scan in {self.feed.seconds_to_next_close(self.cfg.timeframe)/3600:.1f}h"
            "\n```")

    # ---- loop ---------------------------------------------------------

    def run(self):
        c = self.cfg
        tf_min = TF_MS[c.timeframe] / 60000
        self.boot_message()
        self.log.info("universe=%s tf=%s strategies=%s gate=%s",
                      c.pairs, c.timeframe, c.strategies, c.market_filter)

        # evaluate immediately at boot rather than waiting for the next close
        try:
            eq, _ = self.broker.mark_equity()
            self.risk.roll_day(eq)
            if not self.risk.halt_check(eq, self.feed):
                self.scan(eq)
        except Exception:
            self.log.exception("initial scan failed")

        while self.running:
            t0 = time.time()
            try:
                self.tg.poll()

                open_pos = self.store.open_positions()
                if open_pos:
                    # only symbols we hold need the 1m tape; fetch them in one
                    # concurrent round so 8 positions cost ~1 round trip
                    self.feed.bars_many([p["symbol"] for p in open_pos],
                                        "1m", need=120)
                    for pos, reason, pnl in self.broker.manage(tf_min):
                        eq = self.store.equity()
                        r = pnl / pos["r_unit"] if pos["r_unit"] else 0
                        self.tg.send(
                            f"{'🟢' if pnl > 0 else '🔴'} *Closed* {pos['symbol']} "
                            f"{'LONG' if pos['side'] > 0 else 'SHORT'} — {reason}\n"
                            f"{pnl:+.2f} USDT ({r:+.2f}R)  · equity {eq:.2f}")
                        self.log.info("closed %s %s %s pnl=%.2f",
                                      pos["symbol"], pos["side"], reason, pnl)
                    self.maybe_funding()

                eq, unreal = self.broker.mark_equity()
                self.risk.roll_day(eq)
                halt = self.risk.halt_check(eq, self.feed)

                # the expensive work happens ONLY on a bar close
                if not halt and self.feed.due(c.timeframe):
                    self.scan(eq)
                elif halt and int(time.time()) % 3600 < c.poll_seconds:
                    self.log.warning("HALTED: %s", halt)

                if time.time() - self.last_snapshot >= c.snapshot_seconds:
                    self.store.snapshot(eq, len(self.store.open_positions()),
                                        self.broker.gross_notional())
                    self.last_snapshot = time.time()
                self.heartbeat()
                rej = self.risk.snapshot()
                if rej:
                    self.log.info("rejections: %s", rej)

            except Exception as e:
                self.log.exception("cycle error")
                self.store.log("ERROR", "cycle", str(e)[:500])
                try:
                    self.tg.send(f"⚠️ cycle error: {str(e)[:300]}")
                except Exception:
                    pass
                time.sleep(15)

            # idle sleep: long when nothing is open and the next close is far,
            # short when we hold risk and need tight stop management
            base = c.poll_seconds if self.store.open_positions() else c.idle_poll_seconds
            to_close = self.feed.seconds_to_next_close(c.timeframe)
            nap = min(base, max(5.0, to_close + 5))
            time.sleep(max(1.0, nap - (time.time() - t0)))

        self.tg.send("Bot stopped.")


if __name__ == "__main__":
    Runner().run()
