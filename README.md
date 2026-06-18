# Quant AI Research Platform

A production-grade quantitative research platform for systematic investing, feature engineering, machine learning, portfolio optimization, and backtesting on Indian financial markets.

The platform automates the complete quantitative research lifecycle:

- Market data ingestion
- Data validation
- Feature engineering
- Alpha signal generation
- Machine learning modeling
- Portfolio construction
- Backtesting
- Performance analytics
- Experiment tracking

Built using modern Python engineering practices and institutional-grade tooling.

---

## Features

### Market Data Ingestion

Supported Data Sources:

- YFinance
- Alpha Vantage
- FRED
- News API

Capabilities:

- Historical data collection
- Incremental updates
- Retry logic
- Rate limiting
- Data validation
- Schema enforcement
- Automatic failure recovery

---

### Feature Engineering

#### Trend Features

- SMA 5
- SMA 10
- SMA 20
- SMA 50
- SMA 200
- EMA 10
- EMA 20
- EMA 50

#### Momentum Features

- RSI 14
- MACD
- MACD Signal
- ROC 5
- ROC 10
- Momentum 5
- Momentum 10

#### Volatility Features

- Volatility 10
- Volatility 20
- Volatility 50
- ATR 14
- Bollinger Band Width

#### Mean Reversion Features

- Z Score 20
- Distance from SMA
- Distance from EMA

#### Volume Features

- Volume MA 20
- Volume Ratio
- On Balance Volume (OBV)

#### Targets

- Future Return 1 Day
- Future Return 5 Days
- Future Return 10 Days

---

## Architecture

```text
                ┌──────────────────┐
                │ External Sources │
                └─────────┬────────┘
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼

   YFinance      AlphaVantage        FRED API
        │                 │                 │
        └─────────────────┴─────────────────┘
                          │

                Data Connectors Layer
                          │
                          ▼

                Data Validation Layer
                          │
                          ▼

                 Raw Market Dataset
                          │
                          ▼

                Feature Engineering
                          │
                          ▼

                 Feature Dataset
                          │
                          ▼

                 Target Generation
                          │
                          ▼

                 ML Training Layer
                          │
                          ▼

                 Signal Generation
                          │
                          ▼

              Portfolio Construction
                          │
                          ▼

                   Backtesting
                          │
                          ▼

               Performance Metrics
                          │
                          ▼

                      MLflow
```

---

## Project Structure

```text
quant_ai_platform/

├── config/
│   ├── constants.py
│   ├── settings.py
│   └── __init__.py
│
├── data/
│   ├── raw/
│   ├── processed/
│   ├── features/
│   ├── predictions/
│   └── signals/
│
├── alembic/
│
├── docker/
│
├── notebooks/
│
├── src/
│
│   ├── data_pipeline/
│   │   ├── connectors/
│   │   │   ├── yfinance_connector.py
│   │   │   ├── alpha_vantage_connector.py
│   │   │   └── fred_connector.py
│   │   │
│   │   ├── validation/
│   │   └── processing/
│   │
│   ├── feature_engineering/
│   │   ├── trend_features.py
│   │   ├── momentum_features.py
│   │   ├── volatility_features.py
│   │   ├── mean_reversion_features.py
│   │   ├── volume_features.py
│   │   ├── target_generator.py
│   │   └── feature_pipeline.py
│   │
│   ├── models/
│   │   ├── asset.py
│   │   ├── price.py
│   │   ├── feature.py
│   │   ├── signal.py
│   │   ├── trade.py
│   │   └── portfolio_metric.py
│   │
│   ├── database/
│   │   ├── base.py
│   │   ├── session.py
│   │   │
│   │   └── repository/
│   │       ├── asset_repository.py
│   │       ├── price_repository.py
│   │       ├── feature_repository.py
│   │       ├── signal_repository.py
│   │       ├── trade_repository.py
│   │       ├── portfolio_metric_repository.py
│   │       └── __init__.py
│   │
│   ├── utils/
│   │   └── logger.py
│   │
│   ├── training/
│   ├── backtesting/
│   ├── portfolio/
│   └── api/
│
├── tests/
│   ├── database/
│   ├── feature_engineering/
│   ├── data_pipeline/
│   └── api/
│
├── docker-compose.yml
├── pyproject.toml
├── .env.example
└── README.md
```

---

## Database Schema

### Assets

Stores security master data.

| Column | Description |
|----------|-------------|
| symbol | Trading symbol |
| name | Asset name |
| asset_class | Equity, ETF, Index, etc |
| exchange | NSE, BINANCE, FX |
| currency | Trading currency |
| sector | Industry sector |
| is_active | Active flag |

---

### Prices

Stores OHLCV market data.

| Column | Description |
|----------|-------------|
| symbol | Asset symbol |
| date | Trading date |
| open | Open price |
| high | High price |
| low | Low price |
| close | Close price |
| adj_close | Adjusted close |
| volume | Trading volume |
| source | Data source |

---

### Features

Feature metadata registry.

| Column | Description |
|----------|-------------|
| symbol | Asset symbol |
| date | Feature date |
| feature_set_version | Feature version |

---

### Signals

Model predictions.

| Column | Description |
|----------|-------------|
| symbol | Asset symbol |
| date | Signal date |
| signal | Buy/Hold/Sell |
| predicted_return | Expected return |
| model_version | Model version |

---

### Trades

Backtest execution records.

| Column | Description |
|----------|-------------|
| symbol | Asset symbol |
| entry_date | Entry timestamp |
| exit_date | Exit timestamp |
| entry_price | Entry price |
| exit_price | Exit price |
| pnl | Profit/Loss |
| return_pct | Percentage return |

---

### Portfolio Metrics

Strategy performance tracking.

| Column | Description |
|----------|-------------|
| strategy_name | Strategy identifier |
| metric | Metric name |
| value | Metric value |
| calculated_at | Timestamp |

---

## Setup

### Clone Repository

```bash
git clone https://github.com/<username>/quant_ai_platform.git

cd quant_ai_platform
```

---

### Create Virtual Environment

#### Linux / macOS

```bash
python3.12 -m venv .venv

source .venv/bin/activate
```

#### Windows

```powershell
python -m venv .venv

.venv\Scripts\activate
```

---

### Install Dependencies

```bash
pip install -U pip

pip install -e .
```

---

### Configure Environment Variables

Copy:

```bash
cp .env.example .env
```

Update values:

```env
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/quantdb

ALPHA_VANTAGE_KEY=
FRED_API_KEY=
NEWS_API_KEY=

DATA_START_DATE=2015-01-01
DATA_UPDATE_MODE=incremental

MARKET=NSE
STOCK_UNIVERSE=NIFTY500
```

---

## Infrastructure

Start PostgreSQL and MLflow:

```bash
docker compose up -d
```

Verify:

```bash
docker ps
```

---

## Database Migrations

Create migration:

```bash
alembic revision --autogenerate -m "initial schema"
```

Apply migration:

```bash
alembic upgrade head
```

Rollback:

```bash
alembic downgrade -1
```

---

## Running Data Pipeline

Run data ingestion:

```bash
python -m src.data_pipeline.connectors.yfinance_connector
```

Run feature generation:

```bash
python -m src.feature_engineering.feature_pipeline
```

---

## Running Tests

Run all tests:

```bash
pytest
```

Run database tests only:

```bash
pytest tests/database -v
```

Run with coverage:

```bash
pytest --cov=src --cov-report=term-missing
```

---

## MLflow

Launch:

```bash
docker compose up -d mlflow
```

Open:

```text
http://localhost:5000
```

Track:

- Experiments
- Parameters
- Metrics
- Models
- Artifacts
- Feature versions

---

## Development Roadmap

### Phase 1 — Research Infrastructure

- [x] Database Models
- [x] Repository Layer
- [x] Data Connectors
- [x] Feature Engineering Pipeline
- [x] Automated Testing
- [x] MLflow Integration

### Phase 2 — Machine Learning

- [ ] Feature Store
- [ ] XGBoost Models
- [ ] LightGBM Models
- [ ] Model Registry
- [ ] Hyperparameter Optimization

### Phase 3 — Signal Engine

- [ ] Signal Generation
- [ ] Ensemble Models
- [ ] Model Monitoring

### Phase 4 — Portfolio Construction

- [ ] Position Sizing
- [ ] Risk Controls
- [ ] Portfolio Optimization
- [ ] Exposure Constraints

### Phase 5 — Backtesting

- [ ] Transaction Costs
- [ ] Slippage Modeling
- [ ] Walk Forward Testing
- [ ] Performance Analytics

### Phase 6 — Production

- [ ] FastAPI Service
- [ ] Dashboard
- [ ] Scheduled Pipelines
- [ ] Paper Trading
- [ ] Live Deployment

---

## Engineering Principles

- SQLAlchemy 2.0 Typed ORM
- Repository Pattern
- Dependency Injection
- Alembic Migrations
- Transactional Test Isolation
- Type Safety
- Production Logging
- Dockerized Infrastructure
- Reproducible ML Experiments
- No Lookahead Bias
- Institutional Research Standards

---

## Disclaimer

This project is intended for research and educational purposes only.

Nothing contained in this repository constitutes investment advice, financial advice, trading advice, or a recommendation to buy or sell securities.

All investments involve risk, including the possible loss of principal.

---

## License

MIT License

Copyright (c) 2026
