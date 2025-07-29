#!/usr/bin/env python3
"""
KellyCondor Trade Monitor
Monitor live trading activity, open positions, and trade history
"""

import redis
import json
import pandas as pd
from datetime import datetime
import time
import sys

def connect_to_redis():
    """Connect to Redis to get trading data"""
    try:
        r = redis.Redis(host='localhost', port=6379, db=0)
        r.ping()
        return r
    except:
        print("❌ Could not connect to Redis. Make sure Redis is running.")
        return None

def get_trade_history(redis_client):
    """Get all trade history from Redis"""
    try:
        # Get all trade keys
        trade_keys = redis_client.keys("trade:*")
        trades = []
        
        for key in trade_keys:
            trade_data = redis_client.hgetall(key)
            if trade_data:
                # Decode bytes to strings
                trade = {k.decode('utf-8'): v.decode('utf-8') for k, v in trade_data.items()}
                trades.append(trade)
        
        return trades
    except Exception as e:
        print(f"❌ Error getting trade history: {e}")
        return []

def get_open_positions(redis_client):
    """Get current open positions"""
    try:
        position_keys = redis_client.keys("position:*")
        positions = []
        
        for key in position_keys:
            position_data = redis_client.hgetall(key)
            if position_data:
                position = {k.decode('utf-8'): v.decode('utf-8') for k, v in position_data.items()}
                positions.append(position)
        
        return positions
    except Exception as e:
        print(f"❌ Error getting open positions: {e}")
        return []

def get_trading_status(redis_client):
    """Get current trading system status"""
    try:
        status = redis_client.get("trading_status")
        if status:
            return status.decode('utf-8')
        return "Unknown"
    except:
        return "Unknown"

def display_trade_summary(trades):
    """Display a summary of all trades"""
    if not trades:
        print("📊 No trades found in history")
        return
    
    print(f"\n📊 Trade History Summary ({len(trades)} trades)")
    print("=" * 80)
    
    # Convert to DataFrame for easier analysis
    df = pd.DataFrame(trades)
    
    if 'pnl' in df.columns:
        df['pnl'] = pd.to_numeric(df['pnl'], errors='coerce')
        total_pnl = df['pnl'].sum()
        winning_trades = len(df[df['pnl'] > 0])
        losing_trades = len(df[df['pnl'] < 0])
        
        print(f"💰 Total PnL: ${total_pnl:,.2f}")
        print(f"✅ Winning Trades: {winning_trades}")
        print(f"❌ Losing Trades: {losing_trades}")
        print(f"📈 Win Rate: {winning_trades/len(trades)*100:.1f}%")
    
    print("\n📋 Recent Trades:")
    print("-" * 80)
    
    # Show last 10 trades
    for i, trade in enumerate(trades[-10:], 1):
        date = trade.get('date', 'Unknown')
        trade_type = trade.get('type', 'Unknown')
        pnl = trade.get('pnl', '0')
        size = trade.get('size', '0')
        status = trade.get('status', 'Unknown')
        
        pnl_color = "🟢" if float(pnl) > 0 else "🔴" if float(pnl) < 0 else "⚪"
        print(f"{i:2d}. {date} | {trade_type:15s} | {pnl_color} ${pnl:>8s} | Size: {size:>6s} | {status}")

def display_open_positions(positions):
    """Display current open positions"""
    if not positions:
        print("\n📦 No open positions")
        return
    
    print(f"\n📦 Open Positions ({len(positions)} positions)")
    print("=" * 80)
    
    for i, pos in enumerate(positions, 1):
        symbol = pos.get('symbol', 'Unknown')
        quantity = pos.get('quantity', '0')
        avg_price = pos.get('avg_price', '0')
        market_value = pos.get('market_value', '0')
        unrealized_pnl = pos.get('unrealized_pnl', '0')
        
        print(f"{i}. {symbol:10s} | Qty: {quantity:>6s} | Avg: ${avg_price:>8s} | Value: ${market_value:>10s} | PnL: ${unrealized_pnl:>8s}")

def display_system_status(status):
    """Display current system status"""
    print(f"\n🔄 Trading System Status: {status}")
    print("=" * 80)

def main():
    """Main monitoring function"""
    print("🔍 KellyCondor Trade Monitor")
    print("=" * 80)
    
    # Connect to Redis
    redis_client = connect_to_redis()
    if not redis_client:
        return
    
    # Get current data
    trades = get_trade_history(redis_client)
    positions = get_open_positions(redis_client)
    status = get_trading_status(redis_client)
    
    # Display information
    display_system_status(status)
    display_open_positions(positions)
    display_trade_summary(trades)
    
    print(f"\n⏰ Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n💡 Tip: Run this script periodically to monitor your trading activity")

if __name__ == "__main__":
    main() 