from datetime import datetime, timezone

import pandas as pd

from config import BotConfig
from storage import Store
from data_client import fetch_okx, fetch_ticker
from strategy import StrategyBrain, timeframe_for_pair

FUNDING_INTERVAL_SEC = 8 * 3600  # OKX funding settles every 8h (00/08/16 UTC)

def count_funding_events(opened_dt, closed_dt):
    """Number of 8h funding boundaries crossed while a position was open."""
    if closed_dt <= opened_dt:
        return 0
    return max(0, int(closed_dt.timestamp()) // FUNDING_INTERVAL_SEC
                 - int(opened_dt.timestamp()) // FUNDING_INTERVAL_SEC)


class PaperEngine:
    def __init__(self, cfg: BotConfig, store: Store, brain: StrategyBrain):
        self.cfg = cfg
        self.store = store
        self.brain = brain
        self.store.init_balance(cfg.starting_balance)

    def quote(self, pair):
        # Phase 0 fix: live ticker price (was: last CONFIRMED candle close,
        # up to a full timeframe stale).
        return fetch_ticker(pair)

    def used_margin(self):
        return sum(float(p['margin']) for p in self.store.open_positions())

    def calc_size(self, sig):
        equity = self.store.balance()
        risk_mult = float(sig.get('risk_mult', 1.0))
        risk_usdt = equity * (self.cfg.risk_per_trade_pct / 100) * risk_mult
        stop_pct = max(float(sig['risk_pct']), 1e-6)
        notional = risk_usdt / stop_pct
        notional = min(notional, equity * (self.cfg.max_notional_pct / 100))
        margin = notional / self.cfg.leverage
        if margin > equity * (self.cfg.max_margin_per_position_pct / 100):
            margin = equity * (self.cfg.max_margin_per_position_pct / 100)
            notional = margin * self.cfg.leverage
        qty = notional / sig['entry']
        return notional, margin, qty

    def can_open(self, sig, margin):
        equity = self.store.balance()
        if self.cfg.same_pair_lock:
            for p in self.store.open_positions():
                if p['pair'] == sig['pair']:
                    return False, 'pair_already_open'
        if len(self.store.open_positions()) >= self.cfg.max_open_positions:
            return False, 'max_open_positions'
        if self.used_margin() + margin > equity * (self.cfg.max_total_margin_pct / 100):
            return False, 'insufficient_margin_cap'
        return True, ''

    def process_pair(self, pair):
        p_upper = pair.upper().strip()
        tf = timeframe_for_pair(p_upper)
        if tf is None:
            # Uncategorized pair: loudly refuse instead of trading an
            # undocumented fallback config (Phase 0 routing fix).
            print(f"⛔ Skipping {p_upper}: not assigned to any strategy category.")
            return None

        df_candles = fetch_okx(p_upper, tf, 1000)
        if df_candles.empty:
            return None
        sig = self.brain.latest_signal(p_upper, df_candles)
        if not sig:
            return None

        # Deduplicate same pair/side/setup/signal time.
        last_key = self.store.get_state('last_signal_key', {})
        key = f"{p_upper}|{sig['side']}|{sig['setup']}|{sig['signal_time']}"
        if last_key.get(p_upper) == key:
            return None

        notional, margin, qty = self.calc_size(sig)
        sig.update({'notional': notional, 'margin': margin})
        can, reason = self.can_open(sig, margin)
        if self.store.paused() or self.cfg.paused:
            sig.update({'status': 'SKIPPED', 'reason': 'paused'})
            self.store.add_signal(sig)
            return sig
        if not can:
            sig.update({'status': 'SKIPPED', 'reason': reason})
            self.store.add_signal(sig)
            return sig

        sig.update({'status': 'OPENED', 'reason': ''})
        signal_id = self.store.add_signal(sig)
        self.store.add_position({**sig, 'signal_id': signal_id, 'qty': qty, 'leverage': self.cfg.leverage})
        last_key[p_upper] = key
        self.store.set_state('last_signal_key', last_key)
        return sig

    def _check_levels(self, p):
        """Evaluate TP/SL for an open position.

        Phase 0 honesty pass:
        - Walks the 1m candles SINCE ENTRY in chronological order and checks
          each candle's high/low against the stop/target, so intrabar hits are
          caught instead of only the (previously stale) last confirmed close.
        - If a single 1m candle spans BOTH the stop and the target, we assume
          the STOP filled first (pessimistic tie-break, standard for honest
          backtests/paper engines).
        - Falls back to the live ticker when no candle touched a level.
        Returns (reason, exit_price). reason=None means still alive, and
        exit_price is then the current mark price (or None if no data).
        """
        side = p['side']
        sl = float(p['sl'])
        tp = float(p['tp'])

        try:
            opened_ts = pd.Timestamp(p['opened_at'])
        except Exception:
            opened_ts = None

        df1m = fetch_okx(p['pair'], '1m', 5)
        if not df1m.empty:
            if opened_ts is not None:
                df1m = df1m[df1m.index >= opened_ts]
            for _, c in df1m.iterrows():
                high = float(c['high'])
                low = float(c['low'])
                if side == 'LONG':
                    if low <= sl:
                        return 'SL', sl  # pessimistic even if tp also spanned
                    if high >= tp:
                        return 'TP', tp
                else:
                    if high >= sl:
                        return 'SL', sl
                    if low <= tp:
                        return 'TP', tp

        px = self.quote(p['pair'])
        if px is None:
            return None, None
        if side == 'LONG':
            if px <= sl:
                return 'SL', sl
            if px >= tp:
                return 'TP', tp
        else:
            if px >= sl:
                return 'SL', sl
            if px <= tp:
                return 'TP', tp
        return None, px

    def update_positions(self):
        closed = []
        now = datetime.now(timezone.utc)
        for p in self.store.open_positions():
            reason, exit_price = self._check_levels(p)
            if reason is None and exit_price is None:
                continue

            # Time barrier check using max_hold_hours stored in meta if enabled.
            if reason is None and self.cfg.close_on_time_exit:
                try:
                    import json
                    meta = p['meta']
                    m = json.loads(meta) if isinstance(meta, str) else {}
                    max_hold = int(m.get('max_hold_hours', 0) or 0)
                    opened = pd.Timestamp(p['opened_at'])
                    if max_hold > 0 and now >= opened + pd.Timedelta(hours=max_hold):
                        reason = 'TIME'
                except Exception:
                    pass

            if reason:
                notional = float(p['notional'])
                entry = float(p['entry'])
                gross = (exit_price / entry - 1) if p['side'] == 'LONG' else (entry / exit_price - 1)
                drag = 2 * ((self.cfg.taker_fee_pct + self.cfg.slippage_pct) / 100)

                # Phase 0: accrue perp funding for every 8h boundary crossed
                # (previously unmodeled at 10x leverage).
                opened_dt = pd.Timestamp(p['opened_at']).to_pydatetime()
                n_fund = count_funding_events(opened_dt, now)
                funding_cost = notional * (self.cfg.funding_rate_pct_8h / 100) * n_fund

                pnl = notional * (gross - drag) - funding_cost
                pnl_pct = 100 * (pnl / notional)
                fees = notional * 2 * (self.cfg.taker_fee_pct / 100) + funding_cost
                if n_fund > 0:
                    print(f"   💸 funding accrual on {p['pair']}: "
                          f"{n_fund} interval(s) = {funding_cost:.3f} USDT")
                self.store.close_position(p['id'], exit_price, reason, pnl, pnl_pct, fees)
                closed.append((p, reason, pnl))
        return closed

    def cycle(self):
        closed = self.update_positions()
        opened = []
        skipped = []
        for pair in [p.strip().upper() for p in self.cfg.pairs if p.strip()]:
            sig = self.process_pair(pair)
            if sig:
                (opened if sig['status'] == 'OPENED' else skipped).append(sig)
        return {'closed': closed, 'opened': opened, 'skipped': skipped, 'stats': self.store.stats()}
