import os
from dataclasses import dataclass, field
from typing import List

@dataclass
class BotConfig:
    """Railway/env configuration for the MTF Local-Opt Paper Bot."""

    # Final paper universe from research. HYPE omitted. BNB watch-only and disabled in strategy.
    pairs: List[str] = field(default_factory=lambda: os.getenv('PAIRS', 'BTC,ETH,SOL,AVAX').split(','))

    starting_balance: float = float(os.getenv('STARTING_BALANCE', '100'))

    # Conservative profile from Phase 17/18 backtests.
    risk_per_trade_pct: float = float(os.getenv('RISK_PER_TRADE_PCT', '1.0'))
    leverage: float = float(os.getenv('LEVERAGE', '3'))
    max_open_positions: int = int(os.getenv('MAX_OPEN_POSITIONS', '2'))
    max_total_margin_pct: float = float(os.getenv('MAX_TOTAL_MARGIN_PCT', '70'))
    max_margin_per_position_pct: float = float(os.getenv('MAX_MARGIN_PER_POSITION_PCT', '35'))
    max_notional_pct: float = float(os.getenv('MAX_NOTIONAL_PCT', '150'))

    # Paper assumptions. Keep conservative while validating live for a month.
    taker_fee_pct: float = float(os.getenv('TAKER_FEE_PCT', '0.04'))
    slippage_pct: float = float(os.getenv('SLIPPAGE_PCT', '0.02'))

    # Not used by the strategy directly; TP/SL are pair-direction config based.
    tp_r_multiple: float = float(os.getenv('TP_R_MULTIPLE', '1.8'))

    # Main loop. 4h strategy does not need 60s polling, but Telegram command responsiveness benefits.
    poll_seconds: int = int(os.getenv('POLL_SECONDS', '120'))

    # Basic risk controls.
    same_pair_lock: bool = os.getenv('SAME_PAIR_LOCK', 'true').lower() == 'true'
    close_on_time_exit: bool = os.getenv('CLOSE_ON_TIME_EXIT', 'true').lower() == 'true'

    db_path: str = os.getenv('DB_PATH', 'paper_bot.db')
    telegram_token: str = os.getenv('TELEGRAM_BOT_TOKEN', '')
    telegram_chat_id: str = os.getenv('TELEGRAM_CHAT_ID', '')

    # Kept only for compatibility with old envs; strategy no longer loads ANN artifacts.
    model_path: str = os.getenv('MODEL_PATH', '')
    meta_path: str = os.getenv('META_PATH', '')

    strategy_mode: str = os.getenv('STRATEGY_MODE', 'MTF_LOCAL_OPT')
    timeframe: str = os.getenv('TIMEFRAME', '')

    paused: bool = os.getenv('PAUSED', 'false').lower() == 'true'
