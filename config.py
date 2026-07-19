import os
from dataclasses import dataclass, field
from typing import List

@dataclass
class BotConfig:
    """Railway/env configuration for the Dynamic Portfolio Unified Bot."""

    # Dynamic fixed asset universe. Handled automatically by category.
    pairs: List[str] = field(default_factory=lambda: os.getenv(
        'PAIRS', 'BTC,ETH,SOL,AVAX,BNB,LINK,NEAR,HYPE,PEPE,WIF,FET'
    ).split(','))

    starting_balance: float = float(os.getenv('STARTING_BALANCE', '100'))

    # Risk parameters
    risk_per_trade_pct: float = float(os.getenv('RISK_PER_TRADE_PCT', '1.0'))
    leverage: float = float(os.getenv('LEVERAGE', '3'))
    max_open_positions: int = int(os.getenv('MAX_OPEN_POSITIONS', '4'))
    max_total_margin_pct: float = float(os.getenv('MAX_TOTAL_MARGIN_PCT', '70'))
    max_margin_per_position_pct: float = float(os.getenv('MAX_MARGIN_PER_POSITION_PCT', '35'))
    max_notional_pct: float = float(os.getenv('MAX_NOTIONAL_PCT', '150'))

    # Paper trading fees & slippage buffers
    taker_fee_pct: float = float(os.getenv('TAKER_FEE_PCT', '0.04'))
    slippage_pct: float = float(os.getenv('SLIPPAGE_PCT', '0.02'))

    # Polling loop (60s is ideal for 5m scalper mode)
    poll_seconds: int = int(os.getenv('POLL_SECONDS', '60'))

    same_pair_lock: bool = os.getenv('SAME_PAIR_LOCK', 'true').lower() == 'true'
    close_on_time_exit: bool = os.getenv('CLOSE_ON_TIME_EXIT', 'true').lower() == 'true'

    # Database Path (Ensure this points to the persistent volume mount)
    db_path: str = os.getenv('DB_PATH', '/data/paper_bot.db')
    
    telegram_token: str = os.getenv('TELEGRAM_BOT_TOKEN', '')
    telegram_chat_id: str = os.getenv('TELEGRAM_CHAT_ID', '')

    paused: bool = os.getenv('PAUSED', 'false').lower() == 'true'
