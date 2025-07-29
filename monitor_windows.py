#!/usr/bin/env python3
"""
Monitor entry window performance for KellyCondor trading system.
"""

import redis
import json
import pandas as pd
from datetime import datetime, time
import pytz
from typing import Dict, List, Optional
import time as time_module


def connect_to_redis():
    """Connect to Redis and return client."""
    try:
        r = redis.Redis(host='localhost', port=6379, db=0)
        r.ping()
        return r
    except Exception as e:
        print(f"❌ Redis: NOT RUNNING (Error: {e})")
        return None


def get_window_trades(redis_client, window_name: str) -> List[Dict]:
    """Get all trades for a specific window."""
    trades = []
    try:
        # Get all trade keys
        trade_keys = redis_client.keys("trade:*")
        
        for key in trade_keys:
            trade_data = redis_client.hgetall(key)
            if trade_data:
                # Convert bytes to strings
                trade = {k.decode(): v.decode() for k, v in trade_data.items()}
                
                # Check if trade has window information
                if 'window_name' in trade and trade['window_name'] == window_name:
                    trades.append(trade)
                elif 'entry_window' in trade and trade['entry_window'] == window_name:
                    trades.append(trade)
    except Exception as e:
        print(f"Error getting window trades: {e}")
    
    return trades


def get_current_window_status() -> Dict:
    """Get current window status based on time."""
    et_tz = pytz.timezone('US/Eastern')
    current_time = datetime.now(et_tz)
    current_time_only = current_time.time()
    
    windows = {
        "Morning": (time(9, 30), time(10, 30)),
        "Mid-Morning": (time(11, 0), time(12, 0)),
        "Afternoon": (time(14, 0), time(15, 0)),
        "Close": (time(15, 30), time(16, 0))
    }
    
    status = {}
    for window_name, (start, end) in windows.items():
        if start <= current_time_only <= end:
            status[window_name] = "🟢 ACTIVE"
        elif current_time_only > end:
            status[window_name] = "🔴 EXPIRED"
        elif current_time_only < start:
            status[window_name] = "⚪ UPCOMING"
        else:
            status[window_name] = "❓ UNKNOWN"
    
    return status


def calculate_window_performance(trades: List[Dict]) -> Dict:
    """Calculate performance metrics for a window."""
    if not trades:
        return {
            "trades_placed": 0,
            "total_pnl": 0.0,
            "win_rate": 0.0,
            "avg_trade_size": 0.0,
            "avg_pnl_per_trade": 0.0
        }
    
    total_pnl = 0.0
    total_size = 0.0
    wins = 0
    
    for trade in trades:
        # Extract PnL (would need to be calculated from fill prices)
        pnl = float(trade.get('pnl', 0.0))
        total_pnl += pnl
        
        # Extract trade size
        size = float(trade.get('quantity', 0)) * float(trade.get('fill_price', 0))
        total_size += size
        
        # Count wins (positive PnL)
        if pnl > 0:
            wins += 1
    
    trades_placed = len(trades)
    win_rate = wins / trades_placed if trades_placed > 0 else 0.0
    avg_trade_size = total_size / trades_placed if trades_placed > 0 else 0.0
    avg_pnl_per_trade = total_pnl / trades_placed if trades_placed > 0 else 0.0
    
    return {
        "trades_placed": trades_placed,
        "total_pnl": total_pnl,
        "win_rate": win_rate,
        "avg_trade_size": avg_trade_size,
        "avg_pnl_per_trade": avg_pnl_per_trade
    }


def display_window_summary(redis_client, window_status: Dict):
    """Display a summary of all windows and their performance."""
    print("\n" + "="*60)
    print("📊 ENTRY WINDOW PERFORMANCE MONITOR")
    print("="*60)
    
    # Current time
    et_tz = pytz.timezone('US/Eastern')
    current_time = datetime.now(et_tz)
    print(f"🕐 Current Time (ET): {current_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Window status and performance
    for window_name, status in window_status.items():
        print(f"{status} {window_name}")
        
        # Get trades for this window
        trades = get_window_trades(redis_client, window_name)
        performance = calculate_window_performance(trades)
        
        print(f"   📈 Trades: {performance['trades_placed']}")
        print(f"   💰 Total PnL: ${performance['total_pnl']:.2f}")
        print(f"   🎯 Win Rate: {performance['win_rate']:.1%}")
        print(f"   📊 Avg Trade Size: ${performance['avg_trade_size']:.2f}")
        print(f"   📈 Avg PnL/Trade: ${performance['avg_pnl_per_trade']:.2f}")
        
        # Show recent trades
        if trades:
            print(f"   📋 Recent Trades:")
            for trade in trades[-3:]:  # Last 3 trades
                timestamp = trade.get('timestamp', 'Unknown')
                status = trade.get('status', 'Unknown')
                print(f"      • {timestamp} - {status}")
        print()


def display_window_recommendations(performance_data: Dict):
    """Display recommendations based on window performance."""
    print("\n" + "="*60)
    print("💡 WINDOW PERFORMANCE RECOMMENDATIONS")
    print("="*60)
    
    best_window = None
    best_pnl = float('-inf')
    worst_window = None
    worst_pnl = float('inf')
    
    for window_name, performance in performance_data.items():
        pnl = performance['total_pnl']
        win_rate = performance['win_rate']
        
        if pnl > best_pnl:
            best_pnl = pnl
            best_window = window_name
        
        if pnl < worst_pnl:
            worst_pnl = pnl
            worst_window = window_name
    
    if best_window and best_pnl > 0:
        print(f"🏆 Best Performing Window: {best_window}")
        print(f"   Total PnL: ${best_pnl:.2f}")
        print(f"   Recommendation: Consider increasing position sizes")
    
    if worst_window and worst_pnl < 0:
        print(f"⚠️  Worst Performing Window: {worst_window}")
        print(f"   Total PnL: ${worst_pnl:.2f}")
        print(f"   Recommendation: Consider reducing position sizes or skipping")
    
    # Overall recommendations
    total_trades = sum(p['trades_placed'] for p in performance_data.values())
    if total_trades > 0:
        print(f"\n📊 Overall Statistics:")
        print(f"   Total Trades Across All Windows: {total_trades}")
        print(f"   Windows with Trades: {len([w for w, p in performance_data.items() if p['trades_placed'] > 0])}")
        
        # Check for windows with no trades
        inactive_windows = [w for w, p in performance_data.items() if p['trades_placed'] == 0]
        if inactive_windows:
            print(f"   ⚠️  Inactive Windows: {', '.join(inactive_windows)}")
            print(f"   💡 Consider reviewing market conditions during these times")


def main():
    """Main monitoring function."""
    print("🔍 KellyCondor Entry Window Monitor")
    print("Monitoring entry window performance and trade activity...")
    
    # Connect to Redis
    redis_client = connect_to_redis()
    if not redis_client:
        print("❌ Cannot connect to Redis. Make sure Redis server is running.")
        print("   Run: sudo systemctl start redis-server")
        return
    
    print("✅ Connected to Redis")
    
    try:
        while True:
            # Get current window status
            window_status = get_current_window_status()
            
            # Display summary
            display_window_summary(redis_client, window_status)
            
            # Get performance data for recommendations
            performance_data = {}
            for window_name in window_status.keys():
                trades = get_window_trades(redis_client, window_name)
                performance_data[window_name] = calculate_window_performance(trades)
            
            # Display recommendations
            display_window_recommendations(performance_data)
            
            print("\n" + "="*60)
            print("⏰ Refreshing in 30 seconds... (Press Ctrl+C to exit)")
            print("="*60)
            
            # Wait before next update
            time_module.sleep(30)
            
    except KeyboardInterrupt:
        print("\n👋 Window monitor stopped.")
    except Exception as e:
        print(f"❌ Error in window monitor: {e}")


if __name__ == "__main__":
    main() 