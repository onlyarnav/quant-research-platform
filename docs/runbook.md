# Production Operations Runbook

This guide covers day-to-day operations, service startup, backtesting execution, pipeline monitoring, and troubleshooting procedures for the Quantitative Research Platform.

---

## 1. Quick Start Commands

### 1.1 Running a Backtest
```bash
# Run a quick standalone demo backtest
python scripts/run_backtest.py --demo

# Run backtest with custom allocation and risk parameters
python scripts/run_backtest.py \
    --strategy-name momentum_trend \
    --allocation-method risk_parity \
    --rebalance weekly \
    --stop-loss 0.05 \
    --take-profit 0.15 \
    --execution-lag 1 \
    --initial-capital 1000000 \
    --persist-db
```

### 1.2 Inspecting Backtest Analytics
```bash
# View summary of the most recent backtest run
python scripts/run_analytics.py --latest

# View specific backtest run
python scripts/run_analytics.py --run-id <run_id>

# Export analytics as JSON
python scripts/run_analytics.py --latest --export-json
```

### 1.3 System Health Check
```bash
# Print formatted operational health summary
python scripts/health_check.py

# Output machine-readable JSON health report
python scripts/health_check.py --json
```

### 1.4 Starting the FastAPI Backend
```bash
# Launch FastAPI server on host 0.0.0.0 and port 8000
python scripts/run_api.py --host 0.0.0.0 --port 8000
```

### 1.5 Starting the Streamlit Dashboard
```bash
# Launch interactive research dashboard
python scripts/run_dashboard.py --port 8501
```

### 1.6 Running Daily Pipeline Once (Manual Trigger)
```bash
# Dry-run validation
python scripts/run_daily_pipeline.py --dry-run

# Live sequential execution
python scripts/run_daily_pipeline.py --strategy-name daily_prod
```

### 1.7 Starting the Background Scheduler Daemon
```bash
# Start background cron scheduler (configured via .env or CLI args)
python scripts/run_scheduler.py --time 08:30 --timezone Asia/Kolkata
```

---

## 2. Docker Deployment

### 2.1 Starting Database and App Services
```bash
# Start PostgreSQL database
docker compose up -d postgres

# Start API, Dashboard, and Scheduler services
docker compose -f docker-compose.app.yml up -d
```

### 2.2 Verifying Service Health
```bash
curl http://localhost:8000/health
```

---

## 3. Troubleshooting & Failure Recovery

| Issue / Symptom | Potential Root Cause | Resolution Procedure |
|---|---|---|
| `Database unreachable` in `/health` | PostgreSQL container stopped or port conflict | Verify `docker ps`; run `docker compose up -d postgres`; check `DATABASE_URL` in `.env`. |
| `No common symbols between signals and prices` | Ingestion or signals missing for symbol universe | Run `python scripts/run_ingestion.py` followed by `python scripts/run_signals.py`. |
| `No trained model found ...` in Pipeline | Models have not been trained or artifact folder empty | Run `python scripts/run_training.py` or use synthetic fallback in backtesting. |
| API returns `401 Unauthorized` | `DASHBOARD_API_KEY` configured in `.env` but header omitted | Pass `X-API-Key: <key>` header in HTTP requests, or leave `DASHBOARD_API_KEY` blank for local open access. |
| Streamlit shows `No saved runs found` | Backtests have not been persisted to results directory | Run `python scripts/run_backtest.py --demo` to generate an initial backtest run. |
