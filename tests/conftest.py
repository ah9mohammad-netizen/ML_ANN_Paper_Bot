import logging
import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'bot'))
from botconfig import Config
from broker import PaperBroker
from store import Store
import store as store_module


@pytest.fixture(autouse=True)
def no_network(monkeypatch):
    import requests
    def denied(*args, **kwargs):
        raise AssertionError('Unit tests must not contact external services')
    monkeypatch.setattr(requests.sessions.Session, 'request', denied)


class FakeFeed:
    def __init__(self):
        self.px = 100.0
        self.minute = pd.DataFrame(columns=['open', 'high', 'low', 'close'])
        self.strategy = self.minute.copy()
    def price(self, symbol): return self.px
    def bars(self, symbol, tf, **kw):
        return self.minute if tf == '1m' else self.strategy


@pytest.fixture
def account(monkeypatch):
    monkeypatch.setattr(store_module, 'now', lambda: '2024-01-01T00:00:30+00:00')
    cfg = Config(risk_per_trade=.01, leverage=10, timeframe='8h',
                 slip_entry_bps=0, slip_stop_bps=0, taker_fee=.0005)
    store = Store(':memory:'); store.init_account(1000)
    feed = FakeFeed()
    broker = PaperBroker(cfg, store, feed, logging.getLogger('test'))
    yield cfg, store, feed, broker
    store.conn.close()


def signal(side=1, **kw):
    s = dict(symbol='BTC', side=side, ref_price=100., sl=95. if side > 0 else 105.,
             tp=120. if side > 0 else 80., tag='TEST', bar_ts='2024-01-01T00:00:00Z',
             atr=1., max_hold_bars=0, trail_atr=0., trail_start_r=0.)
    s.update(kw)
    return s


def candles(rows):
    # (timestamp, open, high, low, close)
    return pd.DataFrame([r[1:] for r in rows],
                        index=pd.to_datetime([r[0] for r in rows], utc=True),
                        columns=['open', 'high', 'low', 'close'])
