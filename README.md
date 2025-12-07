# AI Trading Bot XAUUSD

An ML-based trading bot for XAUUSD (Gold) on MetaTrader 5, featuring automated trading, backtesting, and self-learning capabilities.

## Features

- **ML-Based Predictions**: LightGBM classifier for price direction prediction
- **Comprehensive Feature Engineering**: 50+ technical and time-based features
- **Risk Management**: Multi-layer risk controls with circuit breakers
- **Backtesting Engine**: Vectorized backtester with realistic trading costs
- **Self-Learning**: Scheduled retraining with safe deployment criteria
- **MT5 Integration**: Full trading execution via MetaTrader 5

## Quick Start

### 1. Installation

```bash
# Clone/navigate to project directory
cd TradingBot

# Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### 2. Configuration

Edit `config/config.yaml`:

```yaml
broker:
  terminal_path: "C:/Program Files/MetaTrader 5/terminal64.exe"
  login: ${MT5_LOGIN}      # Set via environment variable
  password: ${MT5_PASSWORD}
  server: ${MT5_SERVER}

trading:
  symbol: "XAUUSD"
  timeframe: "M15"

risk:
  risk_per_trade_percent: 2.0
  max_daily_loss_percent: 3.0
  max_drawdown_percent: 15.0
```

Set MT5 credentials as environment variables:
```bash
set MT5_LOGIN=12345678
set MT5_PASSWORD=your_password
set MT5_SERVER=YourBroker-Server
```

### 3. Usage

```bash
# Fetch historical data (5 years)
python main.py --mode fetch_data --days 1825

# Train model
python main.py --mode train

# Run backtest
python main.py --mode backtest --start 2023-01-01 --end 2024-01-01

# Paper trading (demo account)
python main.py --mode paper_trade

# Live trading (real account - USE WITH CAUTION)
python main.py --mode live_trade

# Run retraining cycle
python main.py --mode retrain

# Check system status
python main.py --mode status
```

## Architecture

```
TradingBot/
├── config/               # Configuration management
│   ├── config.yaml       # Main configuration file
│   └── config_loader.py  # Config parser and validator
├── data/                 # Data handling
│   ├── data_fetcher.py   # MT5 data fetching
│   ├── data_store.py     # Data persistence (Parquet/CSV)
│   └── data_validator.py # Data quality checks
├── features/             # Feature engineering
│   └── feature_engineering.py  # 50+ technical features
├── models/               # ML models
│   ├── model_baseline.py   # LightGBM classifier
│   ├── model_trainer.py    # Training with TS-CV
│   ├── model_evaluator.py  # Performance metrics
│   └── model_manager.py    # Model versioning
├── strategy/             # Trading strategy
│   ├── signal_generator.py # Signal generation
│   └── position_sizing.py  # Position sizing
├── risk/                 # Risk management
│   └── risk_manager.py   # Circuit breakers
├── execution/            # Trade execution
│   └── broker_mt5.py     # MT5 API wrapper
├── backtesting/          # Backtesting
│   └── backtester.py     # Vectorized backtest
├── retraining/           # Self-learning
│   └── retrain_scheduler.py # Scheduled retraining
├── monitoring/           # Monitoring & logging
│   ├── logger_setup.py   # Logging configuration
│   └── performance_tracker.py # Performance tracking
├── saved_models/         # Model artifacts
├── data_cache/           # Historical data
├── logs/                 # Log files
└── main.py               # CLI entry point
```

## Trading Logic

### Signal Generation
1. Model predicts: UP (+1), NEUTRAL (0), or DOWN (-1)
2. Confidence filter: Only trade if probability > 55%
3. Session filter: Avoid blocked hours (22:00-04:00 UTC)
4. Volatility filter: Skip extreme volatility periods

### Position Sizing
- Risk per trade: 2% of balance
- Stop Loss: 2x ATR from entry
- Take Profit: 3x ATR (1.5:1 R:R)

### Risk Controls
- Max 2 open positions
- Max 3% daily loss → stop trading
- Max 7% weekly loss → stop trading
- Max 15% drawdown → emergency stop
- 5 consecutive losses → 1 hour cooldown

## Self-Learning System

The bot retrains weekly:
1. Fetches latest data
2. Trains on rolling 5-year window
3. Evaluates new model vs current
4. Deploys only if:
   - Sharpe ratio ≥ 0.5
   - Max drawdown ≤ 20%
   - Win rate ≥ 45%
   - Profit factor ≥ 1.1
   - Not significantly worse than current model

Fallback: Auto-reverts if live DD > 15% in 7 days.

## Performance Metrics

The system tracks:
- Total return, CAGR
- Maximum drawdown
- Sharpe ratio
- Win rate, profit factor
- Average win/loss
- Trade frequency

## Configuration Options

Key parameters in `config.yaml`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `trading.timeframe` | M15 | Trading timeframe |
| `risk.risk_per_trade_percent` | 2.0 | Risk per trade |
| `risk.max_daily_loss_percent` | 3.0 | Daily loss limit |
| `risk.max_drawdown_percent` | 15.0 | Max drawdown |
| `model.prediction_horizon_bars` | 4 | Bars to predict (1 hour) |
| `model.min_probability_threshold` | 0.55 | Min confidence to trade |
| `training.training_window_days` | 1825 | Training data window (5y) |

## Disclaimer

⚠️ **IMPORTANT**: This is experimental software for educational purposes.

- Trading involves substantial risk of loss
- Past performance does not guarantee future results
- Use paper trading extensively before live trading
- Never risk money you cannot afford to lose
- This software comes with NO WARRANTY

## License

MIT License - See LICENSE file for details.
