"""Main loop.

Bar-driven, not poll-driven: the loop wakes often, but a symbol is only
evaluated when a NEW CLOSED bar of its trading timeframe has appeared. Signals
come from the same strategies.py the backtester uses, so what is tested is what
runs.
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
        self.feed = Feed(self.cfg.venue, self.cfg.inst_type, warmup=400)
        self.broker = PaperBroker(self.cfg, self.store, self.feed, self.log)
        self.risk = RiskManager(self.cfg, self.store, self.log)
        self.tg = Telegram(self.cfg.telegram_token, self.cfg.telegram_chat_id,
                           self.store, self.broker, self.risk, self.cfg, self.log)
        self.params = load_params(self.cfg.params_file)
        self.last_bar = {}
        self.last_funding = self.store.get("last_funding_ts", "")
        self.last_heartbeat = 0.0
        self._mkt_daily, self._mkt_ts = None, 0.0
        self.running = True
        self.verify_universe()
        for s in (sigmod.SIGTERM, sigmod.SIGINT):
            sigmod.signal(s, self._stop)

    def _stop(self, *_):
        self.log.info("shutdown signal received")
        self.running = False

    def verify_universe(self):
        """Drop symbols the venue does not actually list, loudly. Silently
        polling a non-existent instrument forever is a failure mode the old
        bot had — FET, for instance, has no OKX perpetual."""
        ok, bad = [], []
        for s in self.cfg.pairs:
            try:
                d = self.feed.bars(s, self.cfg.timeframe, need=60)
            except Exception:
                d = None
            (ok if d is not None and len(d) >= 30 else bad).append(s)
        self.dropped = bad
        if bad:
            self.log.error("no usable %s data for: %s — removed from the "
                           "universe", self.cfg.timeframe, ", ".join(bad))
            self.cfg.pairs = ok
        if not ok:
            raise SystemExit("no tradable symbols — check PAIRS and VENUE")

    # ---- market-wide gate ---------------------------------------------

    def market_filter(self, index):
        """+1 when the market leader's daily trend is up, -1 down, 0 neutral.
        Cached for an hour — it only changes once a day."""
        cfg = self.cfg
        if cfg.market_filter == "none":
            return None
        now = time.time()
        if self._mkt_daily is None or now - self._mkt_ts > 3600:
            d = self.feed.bars(cfg.market_filter_symbol, "1d", need=260)
            if d is None or len(d) < 130:
                self.log.warning("market filter: only %d daily bars for %s — "
                                 "gate disabled this cycle",
                                 0 if d is None else len(d),
                                 cfg.market_filter_symbol)
                return None
            self._mkt_daily, self._mkt_ts = d, now
        return strat.btc_trend_filter(self._mkt_daily, index)

    # ---- signals ------------------------------------------------------

    def signal_for(self, symbol):
        cfg = self.cfg
        df = self.feed.bars(symbol, cfg.timeframe, need=400)
        if df is None or len(df) < 260:
            return None
        bar_ts = str(df.index[-1])
        if self.last_bar.get(symbol) == bar_ts:
            return None                       # already handled this bar
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
                    "be_at_r": float(row.be_at_r), "trail_atr": float(row.trail_atr),
                    "trail_start_r": float(row.trail_start_r),
                    "extra": {"strategy": name}}
        return None

    # ---- funding ------------------------------------------------------

    def maybe_funding(self):
        if not self.cfg.apply_funding:
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
                r = requests.get("https://www.okx.com/api/v5/public/funding-rate",
                                 params={"instId": f"{p['symbol']}-USDT-SWAP"},
                                 timeout=8).json()
                d = (r.get("data") or [{}])[0]
                rates[p["symbol"]] = float(d.get("fundingRate", 0) or 0)
            except Exception:
                continue
        if rates:
            self.broker.accrue_funding(rates)
        self.last_funding = stamp
        self.store.set("last_funding_ts", stamp)

    # ---- loop ---------------------------------------------------------

    def boot_message(self):
        c = self.cfg
        warn = ""
        if self.store.ephemeral_warning:
            warn = f"\n\n*STORAGE WARNING*\n{self.store.ephemeral_warning}"
            self.log.error(self.store.ephemeral_warning)
        if getattr(self, "dropped", None):
            warn += ("\n\n*UNIVERSE*\nNo data for " + ", ".join(self.dropped) +
                     " — removed.")
        eq = self.store.equity()
        self.tg.send(
            f"*Bot started* — {c.mode.upper()}\n"
            f"equity {eq:.2f} USDT  ·  {len(c.pairs)} pairs on {c.timeframe}\n"
            f"strategies: {', '.join(c.strategies)}  "
            f"· market filter: {c.market_filter}\n"
            f"risk {100*c.risk_per_trade:.2f}%/trade · {c.leverage}x · "
            f"max {c.max_open} open · gross cap {c.max_gross_notional}x\n"
            f"kill switches: day {100*c.max_daily_loss:.0f}% · "
            f"DD {100*c.max_drawdown_halt:.0f}% · "
            f"{c.max_consecutive_losses} consecutive losses"
            + warn)

    def heartbeat(self):
        if self.cfg.heartbeat_hours <= 0:
            return
        if time.time() - self.last_heartbeat < self.cfg.heartbeat_hours * 3600:
            return
        self.last_heartbeat = time.time()
        m = metrics.from_store(self.store)
        eq, unreal = self.broker.mark_equity()
        self.tg.send(f"*Heartbeat* equity {eq:.2f} ({unreal:+.2f} open)\n"
                     "```\n" + metrics.fmt(m) + "\n```")

    def run(self):
        c = self.cfg
        tf_min = TF_MS[c.timeframe] / 60000
        self.boot_message()
        self.log.info("universe=%s tf=%s strategies=%s", c.pairs, c.timeframe,
                      c.strategies)

        while self.running:
            t0 = time.time()
            try:
                self.tg.poll()

                # 1. keep the 1m tape warm for every symbol we might hold
                for p in self.store.open_positions():
                    self.feed.bars(p["symbol"], "1m", need=120)

                # 2. manage open risk BEFORE looking for anything new
                for pos, reason, pnl in self.broker.manage(tf_min):
                    eq = self.store.equity()
                    self.tg.send(
                        f"{'🟢' if pnl > 0 else '🔴'} *Closed* {pos['symbol']} "
                        f"{'LONG' if pos['side'] > 0 else 'SHORT'} — {reason}\n"
                        f"{pnl:+.2f} USDT "
                        f"({pnl / pos['r_unit'] if pos['r_unit'] else 0:+.2f}R)  "
                        f"· equity {eq:.2f}")
                    self.log.info("closed %s %s %s pnl=%.2f", pos["symbol"],
                                  pos["side"], reason, pnl)

                self.maybe_funding()

                eq, unreal = self.broker.mark_equity()
                self.risk.roll_day(eq)
                halt = self.risk.halt_check(eq, self.feed)

                # 3. look for new entries
                if not halt:
                    for sym in c.pairs:
                        sig = self.signal_for(sym)
                        if not sig:
                            continue
                        if self.store.seen_signal(sym, sig["bar_ts"], sig["tag"]):
                            continue
                        opens = self.store.open_positions()
                        ok, why = self.risk.can_open(sig, opens, eq)
                        if not ok:
                            self.store.record_signal(sig, "SKIPPED", why)
                            self.log.info("skip %s %s: %s", sym, sig["tag"], why)
                            continue
                        sid = self.store.record_signal(sig, "PENDING")
                        pid, err = self.broker.open(
                            sig, sid, eq, self.broker.gross_notional())
                        if pid is None:
                            self.store.conn.execute(
                                "UPDATE signals SET action='SKIPPED', reason=? "
                                "WHERE id=?", (err, sid))
                            self.store.conn.commit()
                            self.log.info("skip %s: %s", sym, err)
                            continue
                        self.store.conn.execute(
                            "UPDATE signals SET action='OPENED' WHERE id=?", (sid,))
                        self.store.conn.commit()
                        p = self.store.conn.execute(
                            "SELECT * FROM positions WHERE id=?", (pid,)).fetchone()
                        self.tg.send(
                            f"📈 *Opened* {sym} "
                            f"{'LONG' if sig['side'] > 0 else 'SHORT'} "
                            f"[{sig['tag']}] {sig.get('regime', '')}\n"
                            f"entry {p['entry']:.6g}  SL {p['sl']:.6g}  "
                            f"TP {p['tp']:.6g}\n"
                            f"risk {p['r_unit']:.2f} USDT  "
                            f"notional {p['notional']:.0f}  "
                            f"R:R {abs(p['tp']-p['entry'])/abs(p['entry']-p['sl']):.2f}")
                else:
                    if int(time.time()) % 3600 < c.poll_seconds:
                        self.log.warning("HALTED: %s", halt)

                self.store.snapshot(eq, len(self.store.open_positions()),
                                    self.broker.gross_notional())
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

            time.sleep(max(1.0, c.poll_seconds - (time.time() - t0)))

        self.tg.send("Bot stopped.")


if __name__ == "__main__":
    Runner().run()
