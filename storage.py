import sqlite3
import json
from pathlib import Path
from datetime import datetime, timezone

SCHEMA = """
CREATE TABLE IF NOT EXISTS state (
  key TEXT PRIMARY KEY,
  value TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS signals (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  created_at TEXT NOT NULL,
  pair TEXT NOT NULL,
  side TEXT NOT NULL,
  setup TEXT NOT NULL,
  probability REAL NOT NULL,
  entry REAL NOT NULL,
  sl REAL NOT NULL,
  tp REAL NOT NULL,
  risk_pct REAL NOT NULL,
  notional REAL NOT NULL,
  margin REAL NOT NULL,
  status TEXT NOT NULL,
  reason TEXT,
  meta TEXT
);
CREATE TABLE IF NOT EXISTS positions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  signal_id INTEGER,
  opened_at TEXT NOT NULL,
  pair TEXT NOT NULL,
  side TEXT NOT NULL,
  setup TEXT NOT NULL,
  probability REAL NOT NULL,
  entry REAL NOT NULL,
  sl REAL NOT NULL,
  tp REAL NOT NULL,
  notional REAL NOT NULL,
  margin REAL NOT NULL,
  leverage REAL NOT NULL,
  qty REAL NOT NULL,
  status TEXT NOT NULL,
  closed_at TEXT,
  exit_price REAL,
  exit_reason TEXT,
  pnl REAL,
  pnl_pct REAL,
  fees REAL DEFAULT 0,
  meta TEXT
);
"""

class Store:
    def __init__(self, path):
        self.path = Path(path)
        self.db_path = str(self.path)  # Now properly mapped for TelegramUI!
        self.conn = sqlite3.connect(self.path, check_same_thread=False)

    def now(self):
        return datetime.now(timezone.utc).isoformat()

    def get_state(self,key,default=None):
        r=self.conn.execute('SELECT value FROM state WHERE key=?',(key,)).fetchone()
        if not r: return default
        try: return json.loads(r['value'])
        except Exception: return r['value']

    def set_state(self,key,value):
        self.conn.execute('INSERT OR REPLACE INTO state(key,value) VALUES(?,?)',(key,json.dumps(value)))
        self.conn.commit()

    def init_balance(self, amount):
        if self.get_state('balance') is None:
            self.set_state('balance', float(amount))
        if self.get_state('realized_pnl') is None:
            self.set_state('realized_pnl', 0.0)
        if self.get_state('paused') is None:
            self.set_state('paused', False)

    def balance(self): return float(self.get_state('balance',0.0))
    def set_balance(self,x): self.set_state('balance',float(x))
    def paused(self): return bool(self.get_state('paused',False))
    def set_paused(self,x): self.set_state('paused',bool(x))

    def open_positions(self):
        return self.conn.execute("SELECT * FROM positions WHERE status='OPEN' ORDER BY opened_at").fetchall()

    def add_signal(self, s):
        cur=self.conn.execute('''INSERT INTO signals(created_at,pair,side,setup,probability,entry,sl,tp,risk_pct,notional,margin,status,reason,meta)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',(
            self.now(),s['pair'],s['side'],s['setup'],s['probability'],s['entry'],s['sl'],s['tp'],s['risk_pct'],s['notional'],s['margin'],s['status'],s.get('reason'),json.dumps(s.get('meta',{}))
        ))
        self.conn.commit(); return cur.lastrowid

    def add_position(self, p):
        cur=self.conn.execute('''INSERT INTO positions(signal_id,opened_at,pair,side,setup,probability,entry,sl,tp,notional,margin,leverage,qty,status,meta)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',(
            p.get('signal_id'),self.now(),p['pair'],p['side'],p['setup'],p['probability'],p['entry'],p['sl'],p['tp'],p['notional'],p['margin'],p['leverage'],p['qty'],'OPEN',json.dumps(p.get('meta',{}))
        ))
        self.conn.commit(); return cur.lastrowid

    def close_position(self, pos_id, exit_price, reason, pnl, pnl_pct, fees):
        self.conn.execute('''UPDATE positions SET status='CLOSED', closed_at=?, exit_price=?, exit_reason=?, pnl=?, pnl_pct=?, fees=? WHERE id=?''',(
            self.now(),exit_price,reason,pnl,pnl_pct,fees,pos_id
        ))
        self.set_balance(self.balance()+pnl)
        self.set_state('realized_pnl', float(self.get_state('realized_pnl',0.0))+pnl)
        self.conn.commit()

    def stats(self):
        sig=dict(self.conn.execute("SELECT status, COUNT(*) c FROM signals GROUP BY status").fetchall())
        rows=self.conn.execute("SELECT * FROM positions WHERE status='CLOSED'").fetchall()
        wins=[r for r in rows if (r['pnl'] or 0)>0]; losses=[r for r in rows if (r['pnl'] or 0)<=0]
        gross_win=sum(r['pnl'] for r in wins) if wins else 0
        gross_loss=abs(sum(r['pnl'] for r in losses)) if losses else 0
        return {
            'balance': self.balance(),
            'realized_pnl': self.get_state('realized_pnl',0.0),
            'open_positions': len(self.open_positions()),
            'closed_positions': len(rows),
            'win_rate': 100*len(wins)/len(rows) if rows else 0,
            'profit_factor': gross_win/gross_loss if gross_loss else None,
            'signals': sig,
        }

    def recent(self, table='signals', limit=10):
        if table not in ['signals','positions']: table='signals'
        return self.conn.execute(f'SELECT * FROM {table} ORDER BY id DESC LIMIT ?', (limit,)).fetchall()
