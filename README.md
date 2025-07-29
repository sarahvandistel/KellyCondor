# KellyCondor

[![Build Status](https://img.shields.io/github/workflow/status/YourUser/KellyCondor/Kelly%20Sizer%20Regression%20Test)](https://github.com/YourUser/KellyCondor/actions)
[![PyPI version](https://img.shields.io/pypi/v/kellycondor)](https://pypi.org/project/kellycondor)

A live‑paper‑ready SPX 0DTE iron‑condor engine using a Kelly‐criterion‑based sizer that dynamically adapts to IV Rank and skew.  
Supports historical replay, CI regression testing, Redis‐backed persistence, and a Dash dashboard.

---

## 📦 Installation

```bash
git clone https://github.com/YourUser/KellyCondor.git
cd KellyCondor
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
pip install -r requirements.txt
```

### Using Mamba Environment

If you're using Mamba with the `sarah313` environment:

```bash
mamba activate sarah313
pip install -e .
pip install -r requirements.txt
```

## 🚀 Quick Start

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ -v --cov=kellycondor --cov-report=html
```

### Running the Dashboard

```bash
# Start the Dash dashboard
python dashboard/app.py
```

Then open your browser to `http://localhost:8050`

### Running Historical Replay

```bash
# Using the CLI script
./scripts/run_replay.sh --account-size 100000

# Or directly with Python
python -m kellycondor.replay --account-size 100000
```

### 📈 Live Paper Trading

KellyCondor supports live paper trading through Interactive Brokers. Here's how to get started:

#### Prerequisites

1. **Install TWS or IB Gateway**
   - Download from [Interactive Brokers](https://www.interactivebrokers.com/en/trading/tws.php)
   - Enable API connections in TWS settings
   - Set port to 7496 for paper trading

2. **Configure API Permissions**
   - In TWS: File → Global Configuration → API → Settings
   - Enable "Enable ActiveX and Socket Clients"
   - Add your IP to "Trusted IPs" or check "Allow connections from localhost"
   - Set Socket Port to 7497 for paper trading

#### Running Paper Trading

```bash
# Basic paper trading
kelly-live --paper

# Custom configuration
kelly-live --paper --host 127.0.0.1 --port 7497 --account-size 50000

# Simulation mode (no IBKR connection)
kelly-live --simulate

# Verbose logging
kelly-live --paper --verbose
```

#### Paper Trading Workflow

1. **Launch TWS in Paper Mode**
   ```bash
   # TWS should be running on port 7497 for paper trading
   # Make sure API connections are enabled
   ```

2. **Start KellyCondor**
   ```bash
   kelly-live --paper --verbose
   ```

3. **Monitor in Dashboard**
   ```bash
   # In another terminal
   python dashboard/app.py
   # Open http://localhost:8050
   ```

4. **Check Order Status**
   - Orders will be logged to console
   - Dashboard shows real-time updates
   - TWS shows order status and fills

#### Configuration Options

- `--paper`: Enable paper trading mode
- `--simulate`: Run in simulation mode (no orders)
- `--host`: IBKR host (default: 127.0.0.1)
- `--port`: IBKR port (7497=paper, 7496=live)
- `--account-size`: Account size for Kelly sizing
- `--verbose`: Enable detailed logging
- `--dry-run`: Test without submitting orders

## 📁 Project Structure

```
KellyCondor/
├── .github/                   # CI configs
│   └── workflows/
│       └── regression.yml
├── src/
│   └── kellycondor/           # your package
│       ├── __init__.py
│       ├── processor.py
│       ├── sizer.py
│       └── replay.py
├── tests/                     # pytest suite
│   ├── test_processor.py
│   └── test_sizer.py
├── dashboard/                 # Dash/Plotly front end
│   └── app.py
├── scripts/                   # helper CLI scripts
│   └── run_replay.sh
├── requirements.txt
├── setup.py                   # editable install
├── README.md
└── .gitignore
```

## 🔧 Core Components

### Processor (`src/kellycondor/processor.py`)
- Processes SPX options data and market information
- Calculates IV Rank from historical implied volatility
- Computes volatility skew metrics

### KellySizer (`src/kellycondor/sizer.py`)
- Implements Kelly criterion-based position sizing
- Dynamically adjusts sizing based on IV Rank and skew
- Provides conservative position sizing for iron condors

### ReplayEngine (`src/kellycondor/replay.py`)
- Historical backtesting engine
- Performance metrics calculation
- Trade simulation and analysis

## 📊 Dashboard Features

The Dash dashboard provides:
- Real-time equity curve visualization
- Trade PnL distribution analysis
- Kelly sizing metrics display
- Recent trades table
- Performance statistics

## 🧪 Testing

The project includes comprehensive unit tests:

```bash
# Run all tests
pytest tests/

# Run specific test file
pytest tests/test_sizer.py

# Run with verbose output
pytest tests/ -v
```

## 🔄 CI/CD

GitHub Actions workflow includes:
- Multi-Python version testing (3.11, 3.12)
- Code linting with flake8, black, and isort
- Regression testing
- Coverage reporting

## 📈 Usage Examples

### Basic Kelly Sizing

```python
from kellycondor import KellySizer

sizer = KellySizer(max_kelly_fraction=0.25)
result = sizer.size_position(
    iv_rank=0.65,
    skew=0.12,
    account_size=100000
)
print(f"Position size: ${result['position_size']:,.2f}")
```

### Historical Replay

```python
from kellycondor import ReplayEngine

engine = ReplayEngine(account_size=100000)
results = engine.run_replay(historical_data)
metrics = engine.calculate_performance_metrics(results['trades'])
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

This software is for educational and research purposes only. It is not intended for actual trading. Use at your own risk. The authors are not responsible for any financial losses incurred through the use of this software.

## 🔗 Dependencies

- **numpy**: Numerical computing
- **pandas**: Data manipulation and analysis
- **backtrader**: Backtesting framework
- **redis**: Data persistence
- **databento**: Market data provider
- **ibapi**: Interactive Brokers API
- **dash**: Web dashboard framework
- **plotly**: Interactive plotting
- **pytest**: Testing framework

---

**KellyCondor** - Where Kelly meets Condor for optimal SPX iron condor sizing. 