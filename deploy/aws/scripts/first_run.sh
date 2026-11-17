#!/usr/bin/env bash
set -euo pipefail

# Initial onboarding and data seeding script for Quant Research Platform
# Prepopulates data lake, models, signals, and demo backtests on a fresh deployment.

APP_DIR="/opt/quant"
COMPOSE_FILE="${APP_DIR}/docker-compose.prod.yml"

echo "============================================================"
echo "QUANT PLATFORM — INITIAL HOST SEEDING & FIRST RUN"
echo "============================================================"

# Helper function to run one-off tasks inside the application container
run_app() {
    echo "-> Running: $*"
    docker compose -f "$COMPOSE_FILE" exec -T api python "$@"
}

echo "[1/6] Running database migrations..."
docker compose -f "$COMPOSE_FILE" run --rm migrate

echo "[2/6] Validating initial data lake contracts..."
run_app scripts/validate_data_contracts.py || echo "[WARN] Data lake initially empty."

echo "[3/6] Running baseline demo backtest to populate dashboard..."
run_app scripts/run_backtest.py --demo

echo "[4/6] Computing strategy analytics..."
run_app scripts/run_analytics.py

echo "[5/6] Verifying system health and readiness..."
run_app scripts/health_check.py

echo "[6/6] Executing smoke test against API..."
run_app scripts/smoke_test.py --url http://127.0.0.1:8000

echo "============================================================"
echo "FIRST RUN COMPLETE: Platform is initialized and dashboard ready."
echo "============================================================"
