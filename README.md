# Institutional Quantitative AI Research Platform for Indian Equities

[![CI Pipeline](https://github.com/quant-research/quant-research-platform/actions/workflows/ci.yml/badge.svg)](https://github.com/quant-research/quant-research-platform/actions)
[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/downloads/release/python-3120/)
[![Tests](https://img.shields.io/badge/tests-786%20passed-brightgreen.svg)]()
[![Code Style](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

A production-ready quantitative research, simulation, and execution platform tailored for Indian financial markets (NSE / NIFTY 500). Built strictly with institutional standards: capital-constrained execution simulation, portfolio allocation protocols, comprehensive trade analytics, FastAPI headless backend, interactive Streamlit frontend, automated DAG pipeline orchestration, and Docker microservices.

---

## Architecture Overview

```mermaid
flowchart TD
    subgraph Data Layer
        A[Yahoo Finance / NSE] -->|Ingestion| B[Raw Parquet]
        B -->|Cleaning & Splits| C[Processed Parquet]
    end

    subgraph Feature Engineering
        C -->|26 Indicators & Multi-Horizon Targets| D[Feature Store]
    end

    subgraph Predictive Modeling
        D -->|Walk-Forward Embargo Splits| E[Model Training / Evaluation]
        E -->|Artifacts & MLflow| F[Model Store / Registry]
        F -->|OOS Inference| G[Predictions]
    end

    subgraph Signal Generation
        G -->|Threshold / Cross-Sectional Rank| H[Trading Signals]
    end

    subgraph Portfolio Construction
        H --> I[Allocator Layer]
        I -->|Equal Weight / Signal-Weighted / Risk Parity / MV| J[Target Weights]
        J -->|Position Caps, Sector Limits, Cash Reserve| K[Constrained Weights]
    end

    subgraph Capital-Constrained Simulation
        K -->|Execution Lag, Fees, Slippage, Stops| L[Weighted Simulation Engine]
        L -->|Atomic Store| M[Backtest Artifacts / DB]
    end

    subgraph Analytics & Presentation
        M --> N[ReportBuilder & Rolling Metrics]
        N --> O[FastAPI Microservice :8000]
        N --> P[Streamlit UI :8501]
    end

    subgraph Operations & Automation
        Q[PipelineOrchestrator] -->|DAG Execution| A
        R[APScheduler] -->|Daily Cron 08:30 IST| Q
        S[HealthChecker] -->|Deep System Inspection| O
    end
```

---

## Platform Features

### 1. Data & Feature Engineering (Weeks 1–6)
- **Data Layers**: Strict separation across `raw`, `processed`, `features`, `predictions`, `signals`, and `backtests` with columnar Parquet storage.
- **NSE Market Calendars**: Native handling of Indian market trading hours, holidays, and Indian Rupee currency (`₹`).
- **Technical Indicators**: 26 momentum, trend, volatility, and volume indicators generated with zero lookahead bias.

### 2. Machine Learning & Signal Pipeline (Weeks 7–9)
- **Model Registry**: Linear, Lasso, Ridge, Random Forest, Gradient Boosting, XGBoost, and LightGBM models.
- **Chronological Walk-Forward**: Purged and embargoed splits preventing cross-contamination between train and test sets.
- **Signal Types**: Threshold-based and cross-sectional Top-N / Rank strategies.

### 3. Capital-Constrained Backtesting Engine (Week 10)
- **Realistic Execution**: Order execution lag (T+1 decision-to-fill), proportional transaction fees, linear slippage.
- **Risk Management**: Intraday / close price stop-loss and take-profit rules with automatic liquidation.
- **Short Selling Support**: Explicit margin accounting, borrow fees, and short cover mechanics.
- **Sensitivity Analysis**: Multi-dimensional parameter grid sweeps across costs, slippage, and lags.

### 4. Portfolio Construction Layer (Week 11)
- **Standardized Allocators**: Protocol-driven architecture (`Allocator`, `AllocationContext`, `AllocationResult`).
- **Mathematical Optimization**: Equal Weight, Signal-Weighted (confidence / return), Volatility-Weighted, Risk Parity (SLSQP equal risk contribution), and Mean-Variance.
- **Constraint Engine**: Iterative proportional redistribution algorithm enforcing max position caps, sector concentration ceilings, asset class limits, and minimum liquidity reserves.

### 5. Trade & Performance Analytics (Week 12)
- **Institutional Metrics**: Sharpe, Sortino, Calmar, VaR (95%), CVaR (95%), Max Drawdown Duration, Win Rate, Profit Factor, Expectancy, Payoff Ratio.
- **Benchmark Comparisons**: Alpha (Jensen), Beta, Tracking Error, Information Ratio, Up/Down Capture Ratios.
- **Rolling Series**: Dynamic rolling Sharpe, annualized volatility, monthly returns compounding heatmaps, and gross/net exposure.
- **Dual Persistence**: Atomic file-system JSON storage (`data/backtests/`) alongside relational PostgreSQL database repositories.

### 6. Interactive Dashboard & FastAPI Backend (Week 12)
- **FastAPI Backend (`src/dashboard/api/`)**: RESTful API for system health, signal feeds, backtest job management, and run metrics.
- **Streamlit Frontend (`src/dashboard/app.py`)**: Multi-page financial analytics portal featuring Overview, Performance, Trades, Allocations, Benchmarks, Signals, and Interactive Backtest Submission.
- **Dual Client Abstraction**: Transparently switches between in-process (`LocalClient`) and network (`ApiClient`) modes.

### 7. Automation, Operations & Production Infra (Week 13)
- **DAG Pipeline Orchestrator**: Executes the complete quantitative research lifecycle (`Ingestion -> Features -> Training -> Predictions -> Signals -> Backtest -> Analytics`).
- **Failure Resilience**: Automatic downstream step skipping upon upstream failure without crashing the process.
- **Cron Scheduler**: APScheduler daemon triggering recurring daily pipelines at configured market hours (e.g., `08:30 Asia/Kolkata`).
- **Health Checker**: Multi-point diagnostic inspecting Database, Parquet Lake, MLflow, Model Registry, and Disk space.

---

## Quick Start (Windows PowerShell)

### Prerequisites
- Python 3.12+ (64-bit)
- PostgreSQL 15+ (Local or Docker)
- Git

### 1. Environment Setup
```powershell
# Clone the repository
git clone https://github.com/quant-research/quant-research-platform.git
cd quant-research-platform

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install platform and dependencies
pip install --upgrade pip setuptools wheel
pip install -e .
pip install flake8 black isort pytest-cov
```

### 2. Configure Environment Variables
```powershell
# Copy template configuration
Copy-Item .env.example .env

# Verify database connection string in .env:
# DATABASE_URL=postgresql+psycopg2://postgres:postgres@localhost:5432/quant_research
```

### 3. Start Database & Run Migrations
```powershell
# Start PostgreSQL via Docker Compose
docker compose up -d postgres

# Apply database migrations
.\.venv\Scripts\alembic upgrade head
```

### 4. Run Automated Tests
```powershell
# Execute the full 786-test institutional suite
.\.venv\Scripts\python.exe -m pytest tests -q
```

---

## Running the Platform via Docker Compose

Run the complete multi-service microservice stack with a single command:
```powershell
docker compose -f docker-compose.yml -f docker-compose.app.yml up -d --build
```

### Service Map:
| Service | Container Name | Port | Description |
|---|---|---|---|
| `postgres` | `quant_postgres_myversion` | `5432` | Primary relational database |
| `mlflow` | `quant_mlflow_myversion` | `5000` | MLflow experiment tracking server |
| `api` | `quant-research-platform-api-1` | `8000` | FastAPI analytics backend (`http://localhost:8000/docs`) |
| `dashboard` | `quant-research-platform-dashboard-1` | `8501` | Streamlit interactive UI (`http://localhost:8501`) |
| `scheduler` | `quant-research-platform-scheduler-1` | N/A | Background automated daily pipeline daemon |

To inspect system health:
```powershell
docker compose -f docker-compose.yml -f docker-compose.app.yml ps
```

To shut down:
```powershell
docker compose -f docker-compose.yml -f docker-compose.app.yml down
```

---

## CLI Runner Scripts Reference

All CLI runners follow unified standards: `--help` support, exit codes (`0` = Success, `1` = Error, `2` = Usage), and clean logging.

### 1. Execute Backtest Simulation
```powershell
# Run a synthetic zero-dependency validation demo
python scripts/run_backtest.py --demo

# Run a risk-parity backtest with stop-loss and execution lag
python scripts/run_backtest.py --strategy-name risk_parity_strat --allocation-method risk_parity --stop-loss 0.05 --lag 1 --capital 1000000

# Run multi-dimensional sensitivity analysis across costs, slippage, and lag
python scripts/run_backtest.py --demo --sensitivity
```

### 2. Inspect & Rebuild Performance Analytics
```powershell
# Rebuild and display reports for the latest backtest run
python scripts/run_analytics.py --latest

# Rebuild metrics for a specific backtest run id
python scripts/run_analytics.py --run-id <UUID>
```

### 3. System Diagnostic Health Check
```powershell
# Run human-readable terminal health check
python scripts/health_check.py

# Emit JSON report for monitoring / alerting integrations
python scripts/health_check.py --json
```

### 4. Daily Production DAG Pipeline
```powershell
# Execute a dry-run without writing to disk or database
python scripts/run_daily_pipeline.py --dry-run

# Run full end-to-end automated pipeline
python scripts/run_daily_pipeline.py --asset-class equity --timeframe daily
```

### 5. Launch FastAPI Backend
```powershell
python scripts/run_api.py --host 127.0.0.1 --port 8000 --reload
```

### 6. Launch Streamlit UI
```powershell
python scripts/run_dashboard.py --port 8501
```

### 7. Start Background Cron Scheduler
```powershell
python scripts/run_scheduler.py --run-time 08:30 --timezone Asia/Kolkata
```

---

## Project Structure & Modularity Guarantees

Every module enforces strict single-responsibility design and stays **under 300 lines of code**:

```
quant-research-platform/
├── config/                     # Central settings and market configuration
├── data/                       # Partitioned Parquet Data Lake
│   ├── raw/                    # Raw market downloads
│   ├── processed/              # Cleaned OHLCV series
│   ├── features/               # Technical feature datasets
│   ├── predictions/            # Out-of-sample model inferences
│   ├── signals/                # Actionable trading signals
│   ├── backtests/              # Atomic simulation runs (metrics & reports)
│   └── pipeline_runs/          # Orchestrator execution run history
├── docs/                       # Technical platform documentation
│   ├── architecture.md         # Layering, dependency rules, math models
│   ├── runbook.md              # Production operational procedures
│   ├── user_guide.md           # Dashboard operator walkthrough
│   ├── api_reference.md        # REST API specifications
│   ├── production_readiness_checklist.md # Verification checklist
│   └── end_to_end_verification_guide.md  # Detailed execution logs
├── scripts/                    # Thin CLI tools for research and operations
├── src/                        # Core platform source code
│   ├── analytics/              # Performance, trade, benchmark & rolling metrics
│   ├── backtesting/            # Realistic simulation engine, order books, stores
│   ├── dashboard/              # FastAPI API & Streamlit web frontend
│   │   ├── api/                # REST endpoints and security
│   │   └── views/              # Multi-page Streamlit dashboards
│   ├── data_pipeline/          # Ingestion, validation, cleaning, processing
│   ├── data_storage/           # Atomic Parquet writer and layer schemas
│   ├── database/               # SQLAlchemy models and repositories
│   ├── feature_engineering/    # 26 technical indicators & targets
│   ├── infra/                  # DAG orchestrator, steps, scheduler, health
│   ├── models/                 # Walk-forward ML models and training pipelines
│   ├── portfolio/              # Allocators (RP, SW, EW, MV) and constraints
│   ├── signals/                # Threshold and rank signal generators
│   └── utils/                  # Structured logging and helper utilities
└── tests/                      # Automated test suite (786 tests)
```

### Strict Architectural Dependency Direction
$$\text{backtesting} \longrightarrow \text{portfolio} \longrightarrow \text{analytics} \longrightarrow \text{dashboard}$$
- `src/backtesting/` never imports from `portfolio`, `analytics`, or `dashboard`.
- `src/portfolio/` never imports from `analytics` or `dashboard`.
- `src/analytics/` never imports from `dashboard`.
- `src/` modules never import from `scripts/` or `tests/`.

---

## Quality Gates & Verification Status

| Gate | Check | Command | Status |
|---|---|---|---|
| **Gate 1** | Black Code Formatter | `black --check --line-length 88 ...` | PASSED (0 issues) |
| **Gate 1** | isort Import Sorter | `isort --check --profile black ...` | PASSED (0 issues) |
| **Gate 2** | Flake8 Linter | `flake8 ... --max-line-length 88` | PASSED (0 errors) |
| **Gate 3** | File Size (< 300 Lines) | `Get-ChildItem -Recurse ...` | PASSED (Max: 259 lines) |
| **Gate 3** | Static Code Scanner | `gate_check.py` (no prints, no os.getenv) | PASSED (0 violations) |
| **Gate 4** | Dependency Direction | `import_check.py` (strict layer rules) | PASSED (0 violations) |
| **Gate 5** | Test Suite Completeness | `pytest tests/ -q` | PASSED (786 / 786 passed) |
| **Gate 6** | Real End-to-End Run | CLI tools execution & verification | PASSED |
| **Gate 7** | Docker Compose Config | `docker compose ... config` | PASSED |

---

## License

Institutional Quantitative Research Platform — Internal Use Only.
