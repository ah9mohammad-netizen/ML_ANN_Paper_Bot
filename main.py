import time
import traceback
from config import BotConfig
from storage import Store
from strategy import StrategyBrain
from paper_engine import PaperEngine
from telegram_ui import TelegramUI

def main():
    cfg = BotConfig()
    store = Store(cfg.db_path)
    brain = StrategyBrain(cfg)
    engine = PaperEngine(cfg, store, brain)
    tg = TelegramUI(cfg.telegram_token, cfg.telegram_chat_id, store)

    start_msg = (
        f"🤖 Unified Paper Trading Bot Started (Phase 0: honest-sim build)\n"
        f"💰 Balance: {store.balance():.2f} USDT\n"
        f"📈 Pairs: {', '.join([p.strip().upper() for p in cfg.pairs if p.strip()])}\n"
        f"⚙️ Risk: {cfg.risk_per_trade_pct}% | Leverage: {cfg.leverage}x | MaxOpen: {cfg.max_open_positions}\n"
        f"💸 Costs modeled: taker {cfg.taker_fee_pct}%/side + {cfg.slippage_pct}% slippage + funding {cfg.funding_rate_pct_8h}%/8h\n"
        f"🧠 Strategy: Dynamic Portfolio (Cat 1+2: SMC sweep 15m | Cat 3: CCI+BB 5m)"
    )
    tg.send(start_msg)
    print(start_msg)

    while True:
        try:
            tg.poll_once()
            result = engine.cycle()

            for pos, reason, pnl in result.get('closed', []):
                msg = (
                    f"✅ Closed {pos['pair']} {pos['side']}\n"
                    f"Setup: {pos['setup']}\n"
                    f"Reason: {reason}\n"
                    f"PnL: {pnl:.2f} USDT\n"
                    f"New Balance: {store.balance():.2f} USDT"
                )
                tg.send(msg)

            for sig in result.get('opened', []):
                # Honest R:R readout (there is no probability model in this build).
                risk = abs(sig['entry'] - sig['sl'])
                reward = abs(sig['tp'] - sig['entry'])
                rr = reward / risk if risk > 0 else 0.0
                msg = (
                    f"🚀 New Paper Trade\n"
                    f"Pair: {sig['pair']} {sig['side']}\n"
                    f"Setup: {sig['setup']}\n"
                    f"R:R: {rr:.2f}:1 | Risk: {sig['risk_pct']*100:.2f}% of price\n"
                    f"Entry: {sig['entry']:.4f}\n"
                    f"SL: {sig['sl']:.4f} | TP: {sig['tp']:.4f}\n"
                    f"Margin: {sig['margin']:.2f} USDT"
                )
                tg.send(msg)

            for skip in result.get('skipped', []):
                print(f"Skipped {skip['pair']} {skip['side']}: {skip['reason']} prob={skip['probability']:.2f}")

        except Exception as e:
            error_msg = f"⚠️ Bot Error: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            try:
                tg.send(error_msg)
            except Exception:
                pass
            time.sleep(10)

        time.sleep(cfg.poll_seconds)

if __name__ == '__main__':
    main()
