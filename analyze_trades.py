import os
import sqlite3
import pandas as pd
import numpy as np
from pathlib import Path

def analyze_dataframe(df):
    if df.empty:
        print("❌ No closed trades found in the positions dataset.")
        return
    
    # Ensure proper data types
    df['pnl'] = pd.to_numeric(df['pnl'], errors='coerce').fillna(0.0)
    df['pnl_pct'] = pd.to_numeric(df['pnl_pct'], errors='coerce').fillna(0.0)
    df['fees'] = pd.to_numeric(df['fees'], errors='coerce').fillna(0.0)
    df['notional'] = pd.to_numeric(df['notional'], errors='coerce').fillna(0.0)
    
    # 1. High-Level Metrics
    total_trades = len(df)
    closed_df = df[df['status'] == 'CLOSED'].copy()
    open_trades = len(df[df['status'] == 'OPEN'])
    
    print("==================================================")
    print("📈 AUTOMATED TRADING PERFORMANCE REPORT")
    print("==================================================")
    print(f"Total Logged Positions : {total_trades}")
    print(f"Closed Positions       : {len(closed_df)}")
    print(f"Currently Open         : {open_trades}")
    
    if closed_df.empty:
        print("\nℹ️ No closed positions to analyze yet. Let the bot run a bit longer!")
        return

    # 2. Profit & Loss Metrics
    wins_df = closed_df[closed_df['pnl'] > 0]
    losses_df = closed_df[closed_df['pnl'] <= 0]
    
    win_count = len(wins_df)
    loss_count = len(losses_df)
    win_rate = (win_count / len(closed_df)) * 100
    
    gross_profit = wins_df['pnl'].sum()
    gross_loss = abs(losses_df['pnl'].sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    net_pnl = closed_df['pnl'].sum()
    total_fees = closed_df['fees'].sum()
    
    print("\n💡 CORE PROFIT & LOSS METRICS:")
    print(f"--------------------------------------------------")
    print(f"Win Rate               : {win_rate:.2f}% ({win_count} Wins / {loss_count} Losses)")
    print(f"Net Realized PnL       : {net_pnl:.2f} USDT")
    print(f"Gross Profit           : {gross_profit:.2f} USDT")
    print(f"Gross Loss             : {gross_loss:.2f} USDT")
    print(f"Profit Factor          : {profit_factor:.2f}" if profit_factor != np.inf else "Profit Factor          : ∞")
    print(f"Total Fees Paid        : {total_fees:.2f} USDT")
    
    # 3. Average Hold Time
    try:
        closed_df['opened_at_dt'] = pd.to_datetime(closed_df['opened_at'])
        closed_df['closed_at_dt'] = pd.to_datetime(closed_df['closed_at'])
        closed_df['hold_duration'] = closed_df['closed_at_dt'] - closed_df['opened_at_dt']
        avg_hold = closed_df['hold_duration'].mean()
        print(f"Average Trade Hold Time: {avg_hold}")
    except Exception:
        pass

    # 4. Drawdown Calculation
    # Assume a starting balance of 100 if not easily retrievable
    starting_balance = 100.0
    closed_df = closed_df.sort_values('closed_at')
    balance_curve = starting_balance + closed_df['pnl'].cumsum()
    running_max = balance_curve.cummax()
    drawdowns = (running_max - balance_curve) / running_max * 100
    max_drawdown = drawdowns.max()
    print(f"Maximum Equity Drawdown: {max_drawdown:.2f}%")

    # 5. Asset Performance Breakdown
    print("\n📊 PERFORMANCE BY ASSET (PAIR):")
    print(f"--------------------------------------------------")
    asset_groups = closed_df.groupby('pair')
    asset_report = []
    for pair, group in asset_groups:
        p_pnl = group['pnl'].sum()
        p_total = len(group)
        p_wins = len(group[group['pnl'] > 0])
        p_wr = (p_wins / p_total) * 100 if p_total > 0 else 0.0
        p_fees = group['fees'].sum()
        asset_report.append({
            'Pair': pair, 'Trades': p_total, 'Win Rate': f"{p_wr:.1f}%", 
            'Net PnL (USDT)': round(p_pnl, 2), 'Fees (USDT)': round(p_fees, 2)
        })
    asset_df = pd.DataFrame(asset_report)
    print(asset_df.to_string(index=False))

    # 6. Directional Performance Breakdown
    print("\n🔄 PERFORMANCE BY DIRECTION (LONG vs SHORT):")
    print(f"--------------------------------------------------")
    dir_groups = closed_df.groupby('side')
    dir_report = []
    for side, group in dir_groups:
        d_pnl = group['pnl'].sum()
        d_total = len(group)
        d_wins = len(group[group['pnl'] > 0])
        d_wr = (d_wins / d_total) * 100 if d_total > 0 else 0.0
        dir_report.append({
            'Direction': side, 'Trades': d_total, 'Win Rate': f"{d_wr:.1f}%", 'Net PnL (USDT)': round(d_pnl, 2)
        })
    dir_df = pd.DataFrame(dir_report)
    print(dir_df.to_string(index=False))

    # 7. Live API Deployment Recommendation
    print("\n🤖 LIVE API RECOMMENDATION ASSESSMENT:")
    print(f"--------------------------------------------------")
    rec_points = 0
    reasons = []
    
    if win_rate >= 70.0:
        rec_points += 1
        reasons.append("✅ Strategy holds the target win rate (>70%).")
    else:
        reasons.append("❌ Win rate is below the target 70% threshold.")
        
    if profit_factor >= 1.5:
        rec_points += 1
        reasons.append("✅ Profit factor is highly viable (>= 1.5).")
    else:
        reasons.append("❌ Profit factor is too thin (< 1.5) to survive live market slippage.")
        
    if max_drawdown <= 15.0:
        rec_points += 1
        reasons.append("✅ Drawdown risk is well contained (<= 15%).")
    else:
        reasons.append("❌ Drawdown risk is too high (> 15%). Consider lowering leverage.")
        
    if total_trades >= 30:
        rec_points += 1
        reasons.append("✅ Sample size is statistically significant (>= 30 trades).")
    else:
        reasons.append("⚠️ Sample size is too small (< 30 trades). Let it paper-trade longer for high-confidence validation.")

    for r in reasons:
        print(f"  {r}")
        
    print(f"\nFinal Score: {rec_points}/4")
    if rec_points == 4:
        print("🟢 STATUS: APPROVED FOR LIVE DEPLOYMENT. You are ready to configure live API keys!")
    elif rec_points == 3:
        print("🟡 STATUS: VIABLE WITH RISK. Optimize low-performing pairs before putting real funds.")
    else:
        print("🔴 STATUS: HIGH DANGER. Maintain paper trading. Optimize indicators and wait for a full statistical cycle.")
    print("==================================================")

def main():
    # Search for files dynamically
    csv_path = Path('/home/user/uploads/positions.csv')
    local_csv_path = Path('positions.csv')
    db_path = Path('paper_bot.db')
    
    if csv_path.exists():
        print(f"Found uploaded CSV at: {csv_path}")
        df = pd.read_csv(csv_path)
        analyze_dataframe(df)
    elif local_csv_path.exists():
        print(f"Found local CSV at: {local_csv_path}")
        df = pd.read_csv(local_csv_path)
        analyze_dataframe(df)
    elif db_path.exists():
        print(f"Found local SQLite database: {db_path}")
        try:
            conn = sqlite3.connect(db_path)
            df = pd.read_sql_query("SELECT * FROM positions", conn)
            conn.close()
            analyze_dataframe(df)
        except Exception as e:
            print(f"❌ Failed to read database: {e}")
    else:
        # Check parent folder or any uploaded positions.csv
        found = False
        for root, dirs, files in os.walk('/home/user'):
            if 'positions.csv' in files:
                p = Path(root) / 'positions.csv'
                print(f"Found positions.csv dynamically at: {p}")
                df = pd.read_csv(p)
                analyze_dataframe(df)
                found = True
                break
        if not found:
            print("❌ No database file (paper_bot.db) or positions.csv found in workspace.")
            print("💡 Please download your .db file using Telegram /backup, extract positions to positions.csv,")
            print("   and place it in this workspace to trigger the analysis.")

if __name__ == '__main__':
    main()
