import requests
import time
import os

class TelegramUI:
    def __init__(self, token, chat_id, store=None):
        self.token = token
        self.chat_id = str(chat_id) if chat_id else ''
        self.store = store
        self.offset = 0
        self.base = f'https://api.telegram.org/bot{token}' if token else ''

    def enabled(self): 
        return bool(self.token and self.chat_id)

    def send(self, text):
        if not self.enabled():
            print('[TELEGRAM disabled]', text)
            return
        try:
            requests.post(self.base + '/sendMessage', json={'chat_id': self.chat_id, 'text': text[:3900]}, timeout=10)
        except Exception as e:
            print('telegram send error', e)

    def send_document(self, file_path, caption=""):
        if not self.enabled(): return
        if not os.path.exists(file_path):
            self.send(f"❌ File not found: {file_path}")
            return
        try:
            with open(file_path, 'rb') as f:
                requests.post(
                    self.base + '/sendDocument',
                    data={'chat_id': self.chat_id, 'caption': caption},
                    files={'document': f},
                    timeout=60
                )
        except Exception as e:
            self.send(f"❌ Failed to send document: {e}")

    def format_stats(self):
        s = self.store.stats()
        pf = s['profit_factor']
        pf_str = '∞' if pf is None else f'{pf:.2f}'
        
        # Safe unpacking of advanced counters
        try:
            reg_r_total = s['regime_stats']['Ranging']['wins'] + s['regime_stats']['Ranging']['losses']
            reg_r_wr = (s['regime_stats']['Ranging']['wins'] / reg_r_total) * 100 if reg_r_total > 0 else 0.0
            
            reg_t_total = s['regime_stats']['Trending']['wins'] + s['regime_stats']['Trending']['losses']
            reg_t_wr = (s['regime_stats']['Trending']['wins'] / reg_t_total) * 100 if reg_t_total > 0 else 0.0
            
            cat1_total = s['category_stats']['Category 1 (Majors)']['wins'] + s['category_stats']['Category 1 (Majors)']['losses']
            cat1_wr = (s['category_stats']['Category 1 (Majors)']['wins'] / cat1_total) * 100 if cat1_total > 0 else 0.0
            
            cat2_total = s['category_stats']['Category 2 (Mid-Vol)']['wins'] + s['category_stats']['Category 2 (Mid-Vol)']['losses']
            cat2_wr = (s['category_stats']['Category 2 (Mid-Vol)']['wins'] / cat2_total) * 100 if cat2_total > 0 else 0.0
            
            cat3_total = s['category_stats']['Category 3 (High-Vol)']['wins'] + s['category_stats']['Category 3 (High-Vol)']['losses']
            cat3_wr = (s['category_stats']['Category 3 (High-Vol)']['wins'] / cat3_total) * 100 if cat3_total > 0 else 0.0
            
            ratio_str = f"{s['wl_ratio']:.2f}"
            avg_win_str = f"{s['avg_win']:.2f}"
            avg_loss_str = f"{s['avg_loss']:.2f}"
            
            advanced_msg = (
                f"\n⚖️ Avg Win/Loss: {avg_win_str}/{avg_loss_str} (Ratio: {ratio_str})\n\n"
                f"🔄 **REGIME AUDIT COHORT:**\n"
                f"  • Ranging  : {reg_r_wr:.1f}% ({s['regime_stats']['Ranging']['wins']}W/{s['regime_stats']['Ranging']['losses']}L)\n"
                f"  • Trending : {reg_t_wr:.1f}% ({s['regime_stats']['Trending']['wins']}W/{s['regime_stats']['Trending']['losses']}L)\n\n"
                f"📊 **CATEGORY AUDIT COHORT:**\n"
                f"  • Cat 1 (Majors) : {cat1_wr:.1f}% ({s['category_stats']['Category 1 (Majors)']['wins']}W/{cat1_total - s['category_stats']['Category 1 (Majors)']['wins']}L)\n"
                f"  • Cat 2 (Mid-Vol): {cat2_wr:.1f}% ({s['category_stats']['Category 2 (Mid-Vol)']['wins']}W/{cat2_total - s['category_stats']['Category 2 (Mid-Vol)']['wins']}L)\n"
                f"  • Cat 3 (High-Vol): {cat3_wr:.1f}% ({s['category_stats']['Category 3 (High-Vol)']['wins']}W/{cat3_total - s['category_stats']['Category 3 (High-Vol)']['wins']}L)"
            )
        except Exception:
            advanced_msg = ""

        return (f"📊 **PORTFOLIO QUANT AUDIT REPORT**\n"
                f"--------------------------------------------------\n"
                f"💰 Balance: {s['balance']:.2f} USDT\n"
                f"📈 Realized PnL: {s['realized_pnl']:.2f}\n"
                f"🔄 Open: {s['open_positions']} | ✅ Closed: {s['closed_positions']}\n"
                f"🎯 Win Rate: {s['win_rate']:.1f}% | PF: {pf_str}"
                f"{advanced_msg}\n"
                f"--------------------------------------------------\n"
                f"📡 Signals Checked: {s['signals']}")

    def handle_text(self, text):
        t = text.strip().lower()
        if t in ['/start', '/help']:
            self.send('🤖 Commands:\n/stats - Show performance\n/open - View open trades\n/recent - View last 10 signals\n/backup - Download the .db file\n/pause - Stop new entries\n/resume - Start new entries\n/reset [amount] - WIPE all history & reset balance (e.g. /reset 1000)')
        
        elif t == '/stats':
            self.send(self.format_stats())
            
        elif t.startswith('/reset'):
            try:
                parts = text.strip().split()
                if len(parts) > 1:
                    amount = float(parts[1])
                    # Phase 0 fix: fully wipe signals + positions too, otherwise
                    # old trades keep contaminating /stats across test phases.
                    self.store.reset_all(amount)
                    self.send(f"✅ Full reset complete.\n💰 Balance: {amount:.2f} USDT\n🧹 All signals & positions wiped. Stats now start from a clean cohort.")
                else:
                    self.send("⚠️ Usage: /reset [amount] (e.g. /reset 1000)")
            except Exception as e:
                self.send(f"❌ Failed to reset balance: {e}")
        
        elif t == '/pause':
            self.store.set_paused(True)
            self.send('⏸ Paused: No new trades will be opened.')
        
        elif t == '/resume':
            self.store.set_paused(False)
            self.send('▶️ Resumed: Bot is looking for signals.')
            
        elif t == '/backup':
            self.send("📦 Preparing database backup...")
            # This reads the file from the path defined in your DB_PATH variable
            self.send_document(self.store.db_path, f"Latest Backup: {time.strftime('%Y-%m-%d %H:%M:%S')}")

        elif t == '/open':
            rows = self.store.open_positions()
            if not rows:
                self.send('📭 No open positions.')
                return
            msg = '📂 **Open Positions:**\n' + '\n'.join([
                f"#{r['id']} {r['pair']} {r['side']} ({r['setup']})\nEntry: {r['entry']:.4g} | SL: {r['sl']:.4g} | Margin: {r['margin']:.2f}" 
                for r in rows
            ])
            self.send(msg)

        elif t == '/recent':
            rows = self.store.recent('signals', 10)
            if not rows:
                self.send('📭 No recent signals in database.')
                return
            msg = '📡 **Recent Signals:**\n' + '\n'.join([
                f"#{r['id']} {r['pair']} {r['side']} {r['setup']}\nProb: {r['probability']:.2f} | Result: {r['status']} {r['reason'] or ''}" 
                for r in rows
            ])
            self.send(msg)

    def poll_once(self):
        if not self.enabled(): return
        try:
            r = requests.get(self.base + '/getUpdates', params={'timeout': 1, 'offset': self.offset}, timeout=5).json()
            for u in r.get('result', []):
                self.offset = max(self.offset, u['update_id'] + 1)
                msg = u.get('message') or {}
                chat = str((msg.get('chat') or {}).get('id', ''))
                if chat != self.chat_id: continue
                if 'text' in msg:
                    self.handle_text(msg['text'])
        except Exception as e:
            print('telegram poll error', e)
