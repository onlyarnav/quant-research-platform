#!/usr/bin/env bash
set -euo pipefail

# Database and state restore drill script for Quant Platform
# Usage: ./restore.sh <BACKUP_DATE_TAG> [--confirm-production]

BACKUP_TAG="${1:-}"
if [ -z "$BACKUP_TAG" ]; then
    echo "Usage: $0 <BACKUP_DATE_TAG> [--confirm-production]" >&2
    exit 1
fi

CONFIRM_PROD=false
if [ "${2:-}" = "--confirm-production" ]; then
    CONFIRM_PROD=true
fi

APP_DIR="/opt/quant"
COMPOSE_FILE="${APP_DIR}/docker-compose.prod.yml"
TMP_RESTORE="/tmp/restore_${BACKUP_TAG}"

mkdir -p "$TMP_RESTORE"

echo "============================================================"
echo "RESTORATION DRILL: ${BACKUP_TAG}"
echo "============================================================"

# Source environment
if [ -f "${APP_DIR}/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "${APP_DIR}/.env"
    set +a
fi

S3_BUCKET="${BACKUP_S3_BUCKET:-quant-platform-backup}"

# 1. Download backup from S3
if command -v aws &>/dev/null && [ -n "${S3_BUCKET:-}" ]; then
    echo "[INFO] Downloading backup from s3://${S3_BUCKET}/backups/${BACKUP_TAG}/..."
    aws s3 sync "s3://${S3_BUCKET}/backups/${BACKUP_TAG}/" "$TMP_RESTORE"
fi

DUMP_FILE=$(find "$TMP_RESTORE" -name "quantdb_*.dump" | head -n 1 || true)
if [ -z "$DUMP_FILE" ]; then
    echo "[ERROR] No quantdb dump found in downloaded backup!" >&2
    exit 1
fi

TARGET_DB="quantdb_restore_test"
if [ "$CONFIRM_PROD" = "true" ]; then
    TARGET_DB="quantdb"
    echo "[WARNING] Performing restore directly into PRODUCTION database: quantdb!"
else
    echo "[INFO] Safe drill mode: Restoring into scratch database: ${TARGET_DB}"
fi

# 2. Create scratch target DB if needed
docker compose -f "$COMPOSE_FILE" exec -T postgres \
    psql -U "${POSTGRES_USER:-postgres}" -c "CREATE DATABASE ${TARGET_DB};" 2>/dev/null || true

# 3. Restore dump
echo "[INFO] Restoring dump into ${TARGET_DB}..."
docker compose -f "$COMPOSE_FILE" exec -T postgres \
    pg_restore -U "${POSTGRES_USER:-postgres}" -d "$TARGET_DB" --clean --if-exists < "$DUMP_FILE" || true

# 4. Verify restoration
ROW_COUNT=$(docker compose -f "$COMPOSE_FILE" exec -T postgres \
    psql -U "${POSTGRES_USER:-postgres}" -d "$TARGET_DB" -t -c "SELECT count(*) FROM assets;" | tr -d '[:space:]' || echo "0")

echo "[INFO] Restoration check: ${ROW_COUNT} assets found in ${TARGET_DB}."

# Clean up
rm -rf "$TMP_RESTORE"
echo "============================================================"
echo "RESTORE DRILL COMPLETED (Target DB: ${TARGET_DB}, Assets: ${ROW_COUNT})"
echo "============================================================"
