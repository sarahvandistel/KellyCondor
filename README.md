# KellyCondor 🚀

A sophisticated options trading system that implements the Kelly Criterion for iron condor strategies with live IBKR integration, real-time monitoring, and comprehensive backtesting capabilities.

## 🌟 Features

### 📊 **Live Paper Trading**
- **Interactive Brokers Integration**: Real-time connection to TWS/IB Gateway
- **Iron Condor Strategy**: Automated strike selection and position sizing
- **Kelly Criterion**: Dynamic position sizing based on IV rank and volatility skew
- **Redis Trade Tracking**: Complete trade history and position monitoring
- **Entry Window Management**: Multiple intraday entry windows with performance tracking

### 📈 **Real-time Dashboard**
- **Live Performance Metrics**: Equity curves, PnL distributions, trade history
- **Interactive Charts**: Plotly-powered visualizations
- **Real-time Updates**: Web-based dashboard at `http://localhost:8050`

### 🔄 **Backtesting Engine**
- **Historical Data**: Databento integration for market data
- **Strategy Validation**: Comprehensive backtesting with performance metrics
- **Risk Analysis**: Drawdown analysis and statistical validation
- **Window Analysis**: Backtesting with entry window performance tracking

### 🛠️ **Monitoring & Management**
- **Trade Monitor**: Real-time trade tracking and status updates
- **System Status**: Process monitoring and health checks
- **Redis Integration**: Persistent trade data and position tracking
- **Window Monitor**: Entry window performance monitoring and recommendations

## 🚀 Quick Start

### Prerequisites
```bash
# Install Redis
sudo apt install redis-server
sudo systemctl start redis-server

# Install Interactive Brokers TWS (Paper Trading)
# Download from: https://www.interactivebrokers.com/en/trading/tws.php
```

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/KellyCondor.git
cd KellyCondor

# Create and activate conda environment
mamba create -n sarah313 python=3.13
mamba activate sarah313

# Install the package
pip install -e .
```

### Live Paper Trading
```bash
# Start TWS in paper trading mode (port 7497)
# Enable API connections in TWS settings

# Start live trading with default entry windows
kelly-live --paper --enable-windows

# Start live trading with custom window configuration
kelly-live --paper --enable-windows --window-config sample_windows.json

# Monitor entry window performance
python monitor_windows.py

# Monitor trades
python monitor_trades.py

# Check system status
python simple_trade_monitor.py
```

### Dashboard
```bash
# Start the dashboard
python dashboard/app.py

# Access at: http://localhost:8050
```

### Backtesting with Entry Windows
```bash
# Run backtest with entry window analysis
kelly-live --backtest --enable-windows --data-file historical_data.csv

# Run backtest with custom windows and save results
kelly-live --backtest --enable-windows --window-config custom_windows.json --data-file data.csv --output-file results.json
```

## 📁 Project Structure

```
KellyCondor/
├── src/kellycondor/
│   ├── __init__.py          # Package initialization
│   ├── processor.py         # Market data processing
│   ├── sizer.py            # Kelly criterion sizing
│   ├── execution.py        # Live IBKR trading
│   ├── entry_windows.py    # Entry window management
│   ├── replay.py           # Backtesting engine
│   └── cli.py              # Command line interface
├── dashboard/
│   └── app.py              # Dash web dashboard
├── tests/
│   ├── test_processor.py   # Unit tests
│   ├── test_sizer.py       # Unit tests
│   ├── test_execution.py   # Unit tests
│   └── test_entry_windows.py # Entry window tests
├── scripts/
│   └── kelly_replay.py     # Backtesting script
├── monitor_trades.py        # Trade monitoring
├── monitor_windows.py       # Entry window monitoring
├── simple_trade_monitor.py  # System status
├── force_trade.py          # Test trade submission
├── sample_windows.json     # Sample window configuration
└── README.md               # This file
```

## 🔧 Configuration

### IBKR Connection
- **Paper Trading Port**: 7497
- **Live Trading Port**: 7496
- **Host**: 127.0.0.1
- **Client ID**: 1

### Trading Parameters
- **IV Rank Threshold**: > 0.5
- **Skew Threshold**: < 0.1
- **Default Expiry**: 20241220
- **Strike Selection**: ±50 from current price
- **Spread Width**: 25 points

### Entry Windows
- **Morning**: 9:30-10:30 AM ET
- **Mid-Morning**: 11:00-12:00 PM ET
- **Afternoon**: 2:00-3:00 PM ET
- **Close**: 3:30-4:00 PM ET

## 📊 Monitoring Tools

### Trade Monitor
```bash
python monitor_trades.py
```
Shows:
- Current open positions
- Trade history summary
- PnL statistics
- Recent trades

### Entry Window Monitor
```bash
python monitor_windows.py
```
Shows:
- Current window status (Active/Expired/Upcoming)
- Window-specific performance metrics
- Trade recommendations per window
- Historical performance analysis

### System Status
```bash
python simple_trade_monitor.py
```
Shows:
- Trading process status
- Dashboard status
- Redis connection
- TWS connection

### Force Trade (Testing)
```bash
python force_trade.py
```
Forces a trade submission for testing Redis logging.

## 🧪 Testing

### Unit Tests
```bash
pytest tests/
```

### Entry Window Tests
```bash
pytest tests/test_entry_windows.py -v
```

### Integration Tests
```bash
# Test IBKR connection
kelly-live --simulate --verbose

# Test trade submission
python force_trade.py

# Test entry windows
kelly-live --simulate --enable-windows --verbose
```

## 📈 Strategy Details

### Kelly Criterion Implementation
The system uses the Kelly Criterion for position sizing:
- **Win Rate**: Based on historical IV rank buckets
- **Risk Management**: Maximum position size limits
- **Dynamic Sizing**: Adjusts based on current market conditions
- **Window Adjustments**: Position sizes adjusted based on entry window performance

### Iron Condor Strategy
- **Call Spread**: Sell call + Buy higher call
- **Put Spread**: Sell put + Buy lower put
- **Strike Selection**: Based on current SPX price
- **Expiry**: Monthly options (typically 30-45 DTE)

### Entry Window Strategy
- **Multiple Windows**: Trade only during specific time windows
- **Performance Tracking**: Monitor win rates and PnL per window
- **Dynamic Adjustments**: Reduce position sizes for underperforming windows
- **Time-based Filtering**: Avoid trading during low-probability periods

### Market Data Processing
- **IV Rank**: Historical percentile of current IV
- **Volatility Skew**: Difference between call and put IV
- **Real-time Updates**: Continuous market data processing
- **Window Awareness**: Process data with entry window context

## 🔒 Security & Risk

### Paper Trading Only
⚠️ **This system is designed for paper trading only.**
- No real money trading
- Educational and research purposes
- Always test thoroughly before live trading

### Risk Management
- Position size limits
- Maximum drawdown controls
- Stop-loss mechanisms
- Diversification across expiries
- Entry window performance monitoring

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📝 License

This project is for educational and research purposes. Use at your own risk.

## 🆘 Troubleshooting

### Common Issues

**TWS Connection Failed**
```bash
# Check TWS is running on correct port
# Paper trading: 7497
# Live trading: 7496
```

**Redis Connection Error**
```bash
# Start Redis server
sudo systemctl start redis-server
```

**Dashboard Not Loading**
```bash
# Check if dashboard is running
python simple_trade_monitor.py
```

**Entry Windows Not Working**
```bash
# Check timezone settings
# Ensure pytz is installed
pip install pytz

# Verify window configuration
cat sample_windows.json
```

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review the logs
3. Open an issue on GitHub

---

**Happy Trading! 📈**

*Remember: This is for paper trading and educational purposes only.* 