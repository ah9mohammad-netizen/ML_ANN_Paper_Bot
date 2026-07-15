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
        return (f"📊 **Paper Bot Stats**\n"
                f"💰 Balance: {s['balance']:.2f} USDT\n"
                f"📉 Realized PnL: {s['realized_pnl']:.2f}\n"
                f"🔄 Open: {s['open_positions']} | ✅ Closed: {s['closed_positions']}\n"
                f"🎯 Win Rate: {s['win_rate']:.1f}% | PF: {pf_str}\n"
                f"📡 Signals Checked: {s['signals']}")

    def handle_text(self, text):
        t = text.strip().lower()
        if t in ['/start', '/help']:
            self.send('🤖 Commands:\n/stats - Show performance\n/open - View open trades\n/recent - View last 10 signals\n/backup - Download the .db file\n/pause - Stop new entries\n/resume - Start new entries')
        
        elif t == '/stats':
            self.send(self.format_stats())
        
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
