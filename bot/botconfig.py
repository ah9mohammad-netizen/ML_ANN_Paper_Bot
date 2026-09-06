"""Environment-driven configuration with validation.

Every knob is an env var so Railway can drive it, but every knob also has a
sane default and is range-checked at boot. A bot that starts with a nonsense
config is worse than one that refuses to start.
"""
import os
from dataclasses import dataclass, field, asdict
from typing import List


def _f(k, d): return float(os.getenv(k, d))
def _i(k, d): return int(float(os.getenv(k, d)))
def _b(k, d): return os.getenv(k, d).strip().lower() in ("1", "true", "yes", "on")
def _l(k, d): return [x.strip().upper() for x in os.getenv(k, d).split(",") if x.strip()]
def _ll(k, d): return [x.strip().lower() for x in os.getenv(k, d).split(",") if x.strip()]


@dataclass
class Config:
    # ---- universe & timeframes
    pairs: List[str] = field(default_factory=lambda: _l(
        "PAIRS", "BTC,ETH,SOL,XRP,BNB,ADA,DOGE,LINK,AVAX,DOT,LTC,BCH,TRX,XLM,UNI,AAVE,FIL,NEAR,APT,SUI,PEPE,SHIB,WLD,STX,TURBO,WIF,ZRO,ONDO,TAO,PENGU,FARTCOIN,TRUMP,KAITO,HYPE,BICO,XAU"))
    timeframe: str = os.getenv("TIMEFRAME", "8h")
    exit_timeframe: str = os.getenv("EXIT_TIMEFRAME", "1m")   # wick-accurate exits
    venue: str = os.getenv("VENUE", "okx")
    inst_type: str = os.getenv("INST_TYPE", "SWAP")

    # ---- strategy selection (comma separated, evaluated in order)
    strategies: List[str] = field(default_factory=lambda: _ll("STRATEGIES", "donchian"))
    params_file: str = os.getenv("PARAMS_FILE", "params.json")
    # market-wide gate: 'align' = only trade with BTC's daily trend, 'none' = off
    market_filter: str = os.getenv("MARKET_FILTER", "align")
    market_filter_symbol: str = os.getenv("MARKET_FILTER_SYMBOL", "BTC")

    # ---- account & risk
    starting_equity: float = _f("STARTING_EQUITY", "1000")
    risk_per_trade: float = _f("RISK_PER_TRADE", "0.005")     # fraction of equity
    leverage: float = _f("LEVERAGE", "5")
    max_open: int = _i("MAX_OPEN", "8")
    max_gross_notional: float = _f("MAX_GROSS_NOTIONAL", "2.0")   # x equity
    max_notional_per_trade: float = _f("MAX_NOTIONAL_PER_TRADE", "0.35")
    same_symbol_lock: bool = _b("SAME_SYMBOL_LOCK", "true")
    max_correlated: int = _i("MAX_CORRELATED", "8")   # set == MAX_OPEN to match the
    # tested configuration; the 16% backtest drawdown already reflects an
    # uncapped, fully-correlated book. Lower it only if you accept live != test.

    # ---- kill switches
    max_daily_loss: float = _f("MAX_DAILY_LOSS", "0.05")
    max_drawdown_halt: float = _f("MAX_DRAWDOWN_HALT", "0.25")
    max_consecutive_losses: int = _i("MAX_CONSECUTIVE_LOSSES", "8")
    stale_data_halt_min: int = _i("STALE_DATA_HALT_MIN", "20")

    # ---- costs
    taker_fee: float = _f("TAKER_FEE", "0.0005")
    slip_entry_bps: float = _f("SLIP_ENTRY_BPS", "2")
    slip_stop_bps: float = _f("SLIP_STOP_BPS", "5")
    apply_funding: bool = _b("APPLY_FUNDING", "true")

    # ---- operations
    mode: str = os.getenv("MODE", "paper")           # paper | live (live = not built)
    poll_seconds: int = _i("POLL_SECONDS", "30")        # with open positions
    idle_poll_seconds: int = _i("IDLE_POLL_SECONDS", "60")  # flat, waiting
    api_rate_per_sec: float = _f("API_RATE_PER_SEC", "6")   # token bucket
    api_workers: int = _i("API_WORKERS", "6")
    snapshot_seconds: int = _i("SNAPSHOT_SECONDS", "300")   # equity curve rows
    db_path: str = os.getenv("DB_PATH", "/data/bot.db")
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    paused: bool = _b("PAUSED", "false")
    telegram_token: str = os.getenv("TELEGRAM_BOT_TOKEN", "")
    telegram_chat_id: str = os.getenv("TELEGRAM_CHAT_ID", "")
    heartbeat_hours: int = _i("HEARTBEAT_HOURS", "6")

    def validate(self):
        errs = []
        if self.mode not in ("paper",):
            errs.append(f"MODE={self.mode!r} — only 'paper' is implemented. "
                        "Live execution is deliberately not wired up.")
        if not 0 < self.risk_per_trade <= 0.05:
            errs.append("RISK_PER_TRADE must be in (0, 0.05]")
        if not 1 <= self.leverage <= 20:
            errs.append("LEVERAGE must be in [1, 20]")
        if self.max_open < 1:
            errs.append("MAX_OPEN must be >= 1")
        gross = self.max_open * self.max_notional_per_trade
        if gross > self.max_gross_notional * 1.5:
            errs.append(f"MAX_OPEN×MAX_NOTIONAL_PER_TRADE={gross:.1f}x equity "
                        f"but MAX_GROSS_NOTIONAL={self.max_gross_notional}x — "
                        "the per-trade cap can never bind. Tighten one of them.")
        if self.max_gross_notional > 4:
            errs.append("MAX_GROSS_NOTIONAL > 4x equity on correlated crypto "
                        "is not a risk limit, it is a leveraged directional bet.")
        if not 0.5 <= self.api_rate_per_sec <= 15:
            errs.append("API_RATE_PER_SEC must be in [0.5, 15] — OKX public "
                        "market endpoints allow 20/s and headroom is free")
        if self.taker_fee < 0.0002:
            errs.append("TAKER_FEE below 2bps is not a retail fee tier.")
        if not self.pairs:
            errs.append("PAIRS is empty")
        if self.market_filter not in ("align", "none"):
            errs.append("MARKET_FILTER must be 'align' or 'none'")
        if self.timeframe not in ("15m", "30m", "1h", "2h", "4h", "8h", "1d"):
            errs.append(f"TIMEFRAME={self.timeframe!r} is not supported")
        known = {"donchian", "squeeze", "sweep", "pullback", "meanrev",
                 "legacy_smc", "legacy_cci"}
        bad = [s for s in self.strategies if s not in known]
        if bad:
            errs.append(f"unknown STRATEGIES: {bad} — known: {sorted(known)}")
        return errs

    def dump(self):
        return asdict(self)
