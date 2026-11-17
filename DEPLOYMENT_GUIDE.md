# Quantitative Research Platform — Setup & Deployment Guide

This guide contains everything needed to set up, run, and deploy the Quantitative Research Platform on any machine (Local PC, Linux VM, AWS EC2, or Docker).

---

## 🏗️ 1. Architecture Overview

The platform consists of:
- **FastAPI Backend (`:8001`)**: High-performance REST API for market data, predictions, backtest results, and system health.
- **Streamlit Dashboard (`:8501`)**: Interactive research and trading UI (Overview, Signals, Trades, Backtesting, Risk Analytics).
- **PostgreSQL (`:5432`)**: Stores asset metadata, benchmark curves, and persistence states.
- **Parquet Data Lake (`data/`)**: High-speed columnar storage for raw OHLCV, computed features, ML predictions, and trade logs.
- **MLflow Tracking**: Local experiment tracking via `sqlite:///mlflow.db`.

---

## 📋 2. Prerequisites

Before starting, ensure the machine has:
1. **Python 3.12+**
2. **Git**
3. **PostgreSQL 15+** (or Docker to run Postgres container)
4. *(Optional)* **Docker & Docker Compose** for containerized setup

---

## 🚀 3. Quickstart: Local / Virtual Machine Setup

### Step 1: Clone the Repository
```bash
git clone git@github.com:onlyarnav/quant-research-platform.git
cd quant-research-platform
```

### Step 2: Create and Activate Virtual Environment
```bash
# On Linux / macOS:
python3 -m venv .venv
source .venv/bin/activate

# On Windows (PowerShell):
python -m venv .venv
.venv\Scripts\Activate.ps1
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

### Step 4: Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```
Ensure the following key settings are in your `.env`:
```env
# Database
DATABASE_URL=postgresql://postgres:postgres@localhost:5432/quant_research

# API & Dashboard Ports
API_HOST=0.0.0.0
API_PORT=8001
DASHBOARD_API_URL=http://localhost:8001

# Signal & Model Defaults
MODEL_NAME=xgboost_model
SIGNAL_STRATEGY=topn
SIGNAL_TOP_N=5
SIGNAL_THRESHOLD=0.0005

# Backtest Defaults
INITIAL_CAPITAL=100000.0
TRANSACTION_COST=0.0010
SLIPPAGE=0.0005
```

### Step 5: Setup PostgreSQL Database
Ensure your local PostgreSQL instance is running, then create the database:
```sql
CREATE DATABASE quant_research;
```

---

## ⚡ 4. Initial Pipeline Warmup (First-Time Data Ingestion)

To fetch live market data from Yahoo Finance, compute technical features, fit the ML model, and generate trading signals, run:

```bash
python scripts/run_daily_pipeline.py
```

This single command automatically executes:
1. **Ingestion**: Fetches daily OHLCV for symbols listed in `data/universe/nifty500.csv`.
2. **Feature Engineering**: Computes 26 indicators (RSI, MACD, Moving Averages, Volatility, Momentum).
3. **ML Inference**: Generates return predictions.
4. **Signal Generation**: Assigns Top-5 BUY (`+1`), Top-5 SELL (`-1`), and HOLD (`0`) signals.
5. **Backtesting & Analytics**: Computes initial equity curves and performance metrics.

---

## 🖥️ 5. Running the Application

To run the application, open two terminal windows (with `.venv` activated):

### Terminal 1: Start the FastAPI Backend
```bash
python scripts/run_api.py --port 8001
```
* Interactive API Docs will be live at: **http://localhost:8001/docs**

### Terminal 2: Start the Streamlit Dashboard
```bash
streamlit run src/dashboard/app.py --server.port 8501
```
* Dashboard will be live at: **http://localhost:8501**

---

## 🐳 6. Docker Deployment (Single-Command Setup)

If you prefer deploying with Docker without manually installing Python or Postgres:

### Production Multi-Container Run:
```bash
docker compose -f docker-compose.prod.yml up -d --build
```
This starts:
- `postgres` on port 5432
- `api` on port 8001
- `dashboard` on port 8501
- `scheduler` running daily automated cron ingestion

---

## ☁️ 7. Cloud / AWS Deployment

For deploying on an AWS EC2 instance:
1. **Automated Setup Script**:
   ```bash
   bash deploy/aws/scripts/bootstrap.sh
   ```
2. **Deploy Code Updates**:
   ```bash
   bash deploy/aws/scripts/deploy.sh
   ```
3. **Terraform Infrastructure**:
   The full AWS infrastructure (VPC, Security Groups, S3 persistence, IAM) is pre-configured in `deploy/aws/terraform/`. Run:
   ```bash
   cd deploy/aws/terraform
   terraform init
   terraform apply
   ```

---

## 🛠️ 8. Useful Operations & CLI Cheatsheet

| Task | Command |
| :--- | :--- |
| **Run System Health Check** | `python scripts/health_check.py` |
| **Run Test Suite (799 tests)** | `pytest` |
| **Change Stock Universe** | Edit `data/universe/nifty500.csv` with desired ticker symbols (e.g. `TCS`, `INFY`) |
| **Retrain ML Model** | `python scripts/run_training.py --model-type xgboost --model-name xgboost_model` |
| **Re-run Signals** | `python scripts/run_signals.py --strategy topn --top-n 5` |
| **Execute Custom Backtest** | `python scripts/run_backtest.py --strategy topn_production --start-date 2026-01-01` |

---

## ❓ 9. Common Troubleshooting

1. **"No signals files found for requested symbols"**:
   - Run `python scripts/run_daily_pipeline.py` to generate the latest predictions and signals.
2. **"Database connection refused"**:
   - Verify PostgreSQL is running (`sudo systemctl status postgresql` or local Postgres service) and `.env` credentials match.
3. **Port already in use (`8001` or `8501`)**:
   - Kill lingering processes or change the port via `--port <new_port>` flag.
