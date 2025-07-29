#!/usr/bin/env python3
"""
Simple KellyCondor Trade Monitor
Monitor trading activity from log files and process status
"""

import os
import json
import time
from datetime import datetime
import subprocess
import psutil

def check_process_running(process_name):
    """Check if a process is running"""
    for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
        try:
            if process_name in ' '.join(proc.info['cmdline'] or []):
                return proc.info
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            pass
    return None

def get_recent_logs(log_file="trading.log", lines=50):
    """Get recent log entries"""
    if not os.path.exists(log_file):
        return []
    
    try:
        with open(log_file, 'r') as f:
            return f.readlines()[-lines:]
    except:
        return []

def parse_trade_entries(logs):
    """Parse trade-related log entries"""
    trades = []
    for line in logs:
        if any(keyword in line for keyword in ['ORDER', 'FILL', 'TRADE', 'IRON CONDOR']):
            trades.append(line.strip())
    return trades

def check_tws_connection():
    """Check if TWS is accessible"""
    try:
        import socket
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        result = sock.connect_ex(('127.0.0.1', 7497))
        sock.close()
        return result == 0
    except:
        return False

def main():
    """Main monitoring function"""
    print("🔍 KellyCondor Simple Trade Monitor")
    print("=" * 80)
    
    # Check trading process
    print("\n📊 Trading Process Status:")
    print("-" * 40)
    
    trading_proc = check_process_running("kelly-live")
    if trading_proc:
        print(f"✅ KellyCondor Live Trading: RUNNING")
        print(f"   PID: {trading_proc['pid']}")
        print(f"   Command: {trading_proc['cmdline']}")
        
        # Get process info
        try:
            proc = psutil.Process(trading_proc['pid'])
            print(f"   CPU Usage: {proc.cpu_percent():.1f}%")
            print(f"   Memory Usage: {proc.memory_percent():.1f}%")
            print(f"   Started: {datetime.fromtimestamp(proc.create_time()).strftime('%Y-%m-%d %H:%M:%S')}")
        except:
            pass
    else:
        print("❌ KellyCondor Live Trading: NOT RUNNING")
    
    # Check dashboard process
    print("\n📈 Dashboard Status:")
    print("-" * 40)
    
    dashboard_proc = check_process_running("app.py")
    if dashboard_proc:
        print(f"✅ Dashboard: RUNNING")
        print(f"   PID: {dashboard_proc['pid']}")
        print(f"   🌐 Dashboard accessible at: http://localhost:8050")
    else:
        print("❌ Dashboard: NOT RUNNING")
    
    # Check TWS connection
    print("\n🏦 TWS Connection Status:")
    print("-" * 40)
    
    if check_tws_connection():
        print("✅ TWS Connection: AVAILABLE (port 7497)")
    else:
        print("❌ TWS Connection: NOT AVAILABLE (port 7497)")
        print("   Make sure TWS is running in paper trading mode")
    
    # Check recent activity
    print("\n📋 Recent Trading Activity:")
    print("-" * 40)
    
    logs = get_recent_logs()
    if logs:
        trade_entries = parse_trade_entries(logs)
        if trade_entries:
            print(f"Found {len(trade_entries)} recent trade-related entries:")
            for entry in trade_entries[-10:]:  # Show last 10
                print(f"   {entry}")
        else:
            print("No recent trade activity found in logs")
    else:
        print("No log file found. Trading may not have started yet.")
    
    print(f"\n⏰ Checked at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\n💡 Commands:")
    print("   Start trading: kelly-live --paper --verbose")
    print("   Start dashboard: python dashboard/app.py")
    print("   Check status: python simple_trade_monitor.py")
    print("   Stop trading: kill <trading_pid>")

if __name__ == "__main__":
    main() 