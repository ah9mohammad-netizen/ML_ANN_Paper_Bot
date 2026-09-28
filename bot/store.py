"""SQLite persistence with schema versioning and equity snapshots.

Two things the old store got wrong and this one does not:
  * it warns loudly if DB_PATH is on ephemeral disk, instead of silently
    losing every trade on the next Railway redeploy;
  * it snapshots equity on a schedule, so drawdown is measured on the real
    mark-to-market curve rather than reconstructed from closed trades.
"""
import json, os, sqlite3, time
import math
from contextlib import closing
from datetime import datetime, timezone
from pathlib import Path

SCHEMA_VERSION = 3

DDL = """
CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS state (key TEXT PRIMARY KEY, value TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS signals (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL, bar_ts TEXT NOT NULL,
  symbol TEXT NOT NULL, side INTEGER NOT NULL, tag TEXT NOT NULL,
  ref_price REAL, sl REAL, tp REAL, atr REAL,
  regime TEXT, action TEXT NOT NULL, reason TEXT, extra TEXT,
  UNIQUE(symbol, bar_ts, tag)
);

CREATE TABLE IF NOT EXISTS positions (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  signal_id INTEGER, symbol TEXT NOT NULL, side INTEGER NOT NULL,
  tag TEXT NOT NULL, regime TEXT,
  opened_at TEXT NOT NULL, entry REAL NOT NULL, qty REAL NOT NULL,
  notional REAL NOT NULL, leverage REAL NOT NULL,
  sl REAL NOT NULL, tp REAL NOT NULL, sl0 REAL NOT NULL,
  r_unit REAL NOT NULL, max_hold_bars INTEGER, atr_at_entry REAL,
  entry_fee REAL DEFAULT 0, funding REAL DEFAULT 0,
  status TEXT NOT NULL DEFAULT 'OPEN',
  closed_at TEXT, exit_price REAL, exit_reason TEXT,
  gross REAL, fees REAL, pnl REAL, r_multiple REAL, bars_held INTEGER,
  equity_after REAL, extra TEXT
);
CREATE INDEX IF NOT EXISTS ix_pos_status ON positions(status);
CREATE INDEX IF NOT EXISTS ix_pos_closed ON positions(closed_at);

CREATE TABLE IF NOT EXISTS equity (
  ts TEXT PRIMARY KEY, equity REAL NOT NULL, realized REAL NOT NULL,
  n_open INTEGER NOT NULL, gross_notional REAL NOT NULL
);

CREATE TABLE IF NOT EXISTS events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts TEXT NOT NULL, level TEXT NOT NULL, kind TEXT NOT NULL, msg TEXT
);
"""


def _row(cur, r):
    return {c[0]: r[i] for i, c in enumerate(cur.description)}


def now():
    return datetime.now(timezone.utc).isoformat()


class Store:
    def __init__(self, path):
        self.path = str(path)
        p = Path(self.path)
        p.parent.mkdir(parents=True, exist_ok=True)
        self.ephemeral_warning = self._check_persistence(p)
        self.conn = sqlite3.connect(self.path, check_same_thread=False,
                                    timeout=30)
        self.conn.row_factory = _row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.executescript(DDL)
        v = self.get_meta("schema_version")
        if v is None:
            self.set_meta("schema_version", SCHEMA_VERSION)
        self.conn.commit()

    @staticmethod
    def _check_persistence(p):
        """Railway wipes the container filesystem on every deploy. Only a
        mounted volume survives. Detect the common misconfiguration."""
        parent = str(p.parent.resolve())
        if os.getenv("RAILWAY_ENVIRONMENT") or os.getenv("RAILWAY_PROJECT_ID"):
            mount = os.getenv("RAILWAY_VOLUME_MOUNT_PATH", "")
            if not mount:
                return ("Running on Railway with NO volume attached. "
                        "Every trade record will be destroyed on the next "
                        "deploy or restart. Attach a volume and set DB_PATH "
                        "to a path inside it.")
            if not parent.startswith(str(Path(mount).resolve())):
                return (f"DB_PATH ({parent}) is outside the Railway volume "
                        f"({mount}). History will be lost on redeploy.")
        return None

    # ---- kv ----------------------------------------------------------
    def get_meta(self, k, d=None):
        r = self.conn.execute("SELECT value FROM meta WHERE key=?", (k,)).fetchone()
        return json.loads(r["value"]) if r else d

    def set_meta(self, k, v):
        self.conn.execute("INSERT OR REPLACE INTO meta VALUES(?,?)", (k, json.dumps(v)))
        self.conn.commit()

    def get(self, k, d=None):
        r = self.conn.execute("SELECT value FROM state WHERE key=?", (k,)).fetchone()
        return json.loads(r["value"]) if r else d

    def _set(self, k, v):
        """Set state within the caller's transaction."""
        self.conn.execute("INSERT OR REPLACE INTO state VALUES(?,?)", (k, json.dumps(v)))

    def set(self, k, v):
        with self.conn:
            self._set(k, v)

    # ---- lifecycle ---------------------------------------------------
    def init_account(self, equity):
        if self.get("equity") is None:
            self.set("equity", float(equity))
            self.set("realized", 0.0)
            self.set("paused", False)
            self.set("halt_reason", "")
            self.set("consecutive_losses", 0)
            self.set("day", "")
            self.set("day_start_equity", float(equity))
            self.set("peak_equity", float(equity))

    def equity(self):
        return float(self.get("equity", 0.0))

    def add_equity(self, delta):
        e = self.equity() + delta
        self.set("equity", e)
        return e

    def log(self, level, kind, msg):
        self.conn.execute("INSERT INTO events(ts,level,kind,msg) VALUES(?,?,?,?)",
                          (now(), level, kind, msg))
        self.conn.commit()

    def snapshot(self, equity, n_open, gross):
        self.conn.execute(
            "INSERT OR REPLACE INTO equity VALUES(?,?,?,?,?)",
            (now(), float(equity), float(self.get("realized", 0.0)),
             int(n_open), float(gross)))
        self.conn.commit()

    # ---- signals & positions ----------------------------------------
    def record_signal(self, s, action, reason=""):
        try:
            cur = self.conn.execute(
                """INSERT INTO signals(ts,bar_ts,symbol,side,tag,ref_price,sl,tp,
                   atr,regime,action,reason,extra) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)""",
                (now(), s["bar_ts"], s["symbol"], int(s["side"]), s["tag"],
                 s.get("ref_price"), s.get("sl"), s.get("tp"), s.get("atr"),
                 s.get("regime", ""), action, reason,
                 json.dumps(s.get("extra", {}))))
            self.conn.commit()
            return cur.lastrowid
        except sqlite3.IntegrityError:
            return None                      # already seen this bar

    def seen_signal(self, symbol, bar_ts, tag):
        return self.conn.execute(
            "SELECT 1 FROM signals WHERE symbol=? AND bar_ts=? AND tag=?",
            (symbol, bar_ts, tag)).fetchone() is not None

    def open_positions(self):
        return self.conn.execute(
            "SELECT * FROM positions WHERE status='OPEN' ORDER BY opened_at").fetchall()

    def open_position(self, p):
        with self.conn:
            return self._open_position(p)

    def _open_position(self, p):
        cur = self.conn.execute(
            """INSERT INTO positions(signal_id,symbol,side,tag,regime,opened_at,
               entry,qty,notional,leverage,sl,tp,sl0,r_unit,max_hold_bars,
               atr_at_entry,entry_fee,status,extra)
               VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?, 'OPEN', ?)""",
            (p.get("signal_id"), p["symbol"], int(p["side"]), p["tag"],
             p.get("regime", ""), now(), p["entry"], p["qty"], p["notional"],
             p["leverage"], p["sl"], p["tp"], p["sl"], p["r_unit"],
             int(p.get("max_hold_bars", 0)), p.get("atr_at_entry", 0.0),
             p.get("entry_fee", 0.0), json.dumps(p.get("extra", {}))))
        # Commit the fee and position together; a restart cannot lose one leg.
        self._set("equity", self.equity() - float(p.get("entry_fee", 0)))
        return cur.lastrowid

    def update_position(self, pid, **kw):
        if not kw:
            return
        sets = ",".join(f"{k}=?" for k in kw)
        self.conn.execute(f"UPDATE positions SET {sets} WHERE id=?",
                          (*kw.values(), pid))
        self.conn.commit()

    def close_position(self, pid, cash_delta=None, **kw):
        """Atomically settle once. Returns False if already closed/not found."""
        with self.conn:
            row = self.conn.execute("SELECT * FROM positions WHERE id=? AND status='OPEN'",
                                    (pid,)).fetchone()
            if row is None:
                return False
            kw["status"] = "CLOSED"
            kw["closed_at"] = now()
            if cash_delta is not None:
                e = self.equity() + float(cash_delta)
                self._set("equity", e)
                kw["equity_after"] = e
            sets = ",".join(f"{k}=?" for k in kw)
            self.conn.execute(f"UPDATE positions SET {sets} WHERE id=?",
                              (*kw.values(), pid))
            pnl = float(kw.get("pnl", 0.0))
            # Recompute the derived aggregate, including normalized legacy rows.
            self._set("realized", sum(float(p["pnl"] or 0) for p in self.closed()))
            c = int(self.get("consecutive_losses", 0))
            self._set("consecutive_losses", c + 1 if pnl <= 0 else 0)
        return True

    @staticmethod
    def _reported_trade(p):
        """Normalize accounting in memory; never rewrite historical fills.

        Legacy pnl omitted entry fees. gross/fees/funding are enough to repair
        that arithmetic, but NOT erroneous execution. Preserve the raw fields
        and explicitly mark old trades as unvalidated.
        """
        extra = json.loads(p.get("extra") or "{}")
        p["legacy_execution"] = int(extra.get("execution_version", 0)) < 3
        p["history_warning"] = extra.get("history_warning", "")
        p["raw_pnl"], p["raw_r_multiple"] = p.get("pnl"), p.get("r_multiple")
        components = [p.get(k) for k in ("gross", "fees", "funding")]
        if all(x is not None and math.isfinite(float(x)) for x in components):
            p["pnl"] = float(p["gross"]) - float(p["fees"]) - float(p["funding"])
            p["r_multiple"] = p["pnl"] / p["r_unit"] if p["r_unit"] else 0.0
        return p

    def closed(self, limit=None):
        q = "SELECT * FROM positions WHERE status='CLOSED' ORDER BY closed_at"
        if limit:
            q += f" DESC LIMIT {int(limit)}"
        return [self._reported_trade(p) for p in self.conn.execute(q).fetchall()]

    def backup(self, path):
        """SQLite online backup includes committed WAL pages."""
        with closing(sqlite3.connect(str(path))) as dest:
            self.conn.backup(dest)

    def equity_curve(self):
        return self.conn.execute("SELECT * FROM equity ORDER BY ts").fetchall()
