# KellyCondor 🚀

A sophisticated options trading system that implements the Kelly Criterion for iron condor strategies with live IBKR integration, real-time monitoring, and comprehensive backtesting capabilities.

## 🌟 Features

### 📊 **Live Paper Trading**
- **Interactive Brokers Integration**: Real-time connection to TWS/IB Gateway
- **Iron Condor Strategy**: Automated strike selection and position sizing
- **Kelly Criterion**: Dynamic position sizing based on IV rank and volatility skew
- **Redis Trade Tracking**: Complete trade history and position monitoring

### 📈 **Real-time Dashboard**
- **Live Performance Metrics**: Equity curves, PnL distributions, trade history
- **Interactive Charts**: Plotly-powered visualizations
- **Real-time Updates**: Web-based dashboard at `http://localhost:8050`

### 🔄 **Backtesting Engine**
- **Historical Data**: Databento integration for market data
- **Strategy Validation**: Comprehensive backtesting with performance metrics
- **Risk Analysis**: Drawdown analysis and statistical validation

### 🛠️ **Monitoring & Management**
- **Trade Monitor**: Real-time trade tracking and status updates
- **System Status**: Process monitoring and health checks
- **Redis Integration**: Persistent trade data and position tracking

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

# Start live trading
kelly-live --paper --verbose

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

## 📁 Project Structure

```
KellyCondor/
├── src/kellycondor/
│   ├── __init__.py          # Package initialization
│   ├── processor.py         # Market data processing
│   ├── sizer.py            # Kelly criterion sizing
│   ├── execution.py        # Live IBKR trading
│   ├── replay.py           # Backtesting engine
│   └── cli.py              # Command line interface
├── dashboard/
│   └── app.py              # Dash web dashboard
├── tests/
│   ├── test_processor.py   # Unit tests
│   ├── test_sizer.py       # Unit tests
│   └── test_execution.py   # Unit tests
├── scripts/
│   └── kelly_replay.py     # Backtesting script
├── monitor_trades.py        # Trade monitoring
├── simple_trade_monitor.py  # System status
├── force_trade.py          # Test trade submission
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

### Integration Tests
```bash
# Test IBKR connection
kelly-live --simulate --verbose

# Test trade submission
python force_trade.py
```

## 📈 Strategy Details

### Kelly Criterion Implementation
The system uses the Kelly Criterion for position sizing:
- **Win Rate**: Based on historical IV rank buckets
- **Risk Management**: Maximum position size limits
- **Dynamic Sizing**: Adjusts based on current market conditions

### Iron Condor Strategy
- **Call Spread**: Sell call + Buy higher call
- **Put Spread**: Sell put + Buy lower put
- **Strike Selection**: Based on current SPX price
- **Expiry**: Monthly options (typically 30-45 DTE)

### Market Data Processing
- **IV Rank**: Historical percentile of current IV
- **Volatility Skew**: Difference between call and put IV
- **Real-time Updates**: Continuous market data processing

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

## 📞 Support

For questions or issues:
1. Check the troubleshooting section
2. Review the logs
3. Open an issue on GitHub

---

**Happy Trading! 📈**

*Remember: This is for paper trading and educational purposes only.* 