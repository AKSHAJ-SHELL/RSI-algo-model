# 📈 RSI Mean-Reversion Trading Scanner

A clean, production-ready quantitative trading scanner that identifies oversold conditions using RSI (Relative Strength Index) and implements mean-reversion strategies.

## 🚀 Quick Start

### Prerequisites
- Python 3.8+
- pip package manager

### Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AKSHAJ-SHELL/RSI-algo-model.git
   cd rsi-algo-model
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

3. **Run the application:**
   ```bash
   streamlit run app.py
   ```

## 📊 Features

- **RSI Mean-Reversion Strategies**: Conservative, Aggressive, and advanced strategies
- **Live Signal Scanning**: Real-time analysis across multiple tickers
- **Comprehensive Backtesting**: Historical performance testing with detailed metrics
- **Interactive Charts**: Price/RSI overlays and equity curves
- **Risk Management**: Position sizing, drawdown controls, and stop losses
- **Market Regime Detection**: Adapts strategies based on market conditions

## 🎯 Strategy Overview

The scanner implements **RSI Mean-Reversion Trading**:
- **Entry**: Buy when RSI < 30 (oversold)
- **Exit**: Sell when RSI > 50-70 (mean reversion)
- **Best for**: Range-bound markets like 2023

### Available Strategies
- **Conservative**: RSI < 30 entry, < 50 exit, 1% position size
- **Aggressive**: RSI < 25 entry, < 70 exit, 2% position size
- **Divergence**: Advanced strategy using RSI divergences

## 📱 Usage

### Web Interface
Run `streamlit run app.py` and access the web interface with three main tabs:

1. **Live Scanner**: Scan tickers for current signals
2. **Backtest**: Test strategies on historical data
3. **Charts**: Visualize price action and indicators

### CLI Interface
```bash
# Run backtest
python -m src.main backtest --ticker SPY --start 2023-01-01 --end 2025-12-31

# Scan for signals
python -m src.main scan --tickers SPY,QQQ
```

## 🏗️ Architecture

```
src/
├── core/           # Core trading logic
│   ├── indicators.py    # RSI, moving averages, regime detection
│   ├── signals.py       # Entry/exit signal generation
│   ├── strategies.py    # Trading strategy implementations
│   └── backtest.py      # Backtesting engine
├── data/           # Data handling
│   ├── fetcher.py       # Yahoo Finance data fetching
│   ├── database.py      # SQLite storage
│   └── validators.py    # Data validation
├── trading/        # Trading execution
│   ├── scanner.py       # Live signal scanning
│   └── executor.py      # Trade execution (future)
├── utils/          # Utilities
│   ├── config.py        # Configuration management
│   ├── logging.py       # Logging setup
│   └── errors.py        # Custom exceptions
└── web/            # Streamlit interface
    ├── app.py           # Main web application
    └── charts.py        # Chart visualizations
```

## ⚠️ Disclaimer

**This software is for educational and research purposes only.**

- Not financial advice
- Past performance ≠ future results
- Trading involves substantial risk of loss
- Always backtest thoroughly
- RSI works best in range-bound markets, not trending markets

The authors are not responsible for any financial losses incurred through the use of this software.

## 📊 Performance Metrics

The scanner calculates comprehensive performance metrics:
- **CAGR**: Compound Annual Growth Rate
- **Sharpe Ratio**: Risk-adjusted returns
- **Max Drawdown**: Largest peak-to-trough decline
- **Win Rate**: Percentage of profitable trades
- **Profit Factor**: Gross profits / Gross losses

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

---

**Happy Trading! 🚀📈**
