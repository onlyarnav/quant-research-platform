#!/usr/bin/env bash
set -euo pipefail

# Automated nightly backup script for Quant Platform
# Dumps PostgreSQL databases and archives data/models to S3.

DATE_STR=$(date +%Y%m%d_%H%M%S)
APP_DIR="/opt/quant"
BACKUP_DIR="/tmp/quant_backup_${DATE_STR}"
MARKER_FILE="${APP_DIR}/logs/last_backup_status"
COMPOSE_FILE="${APP_DIR}/docker-compose.prod.yml"

mkdir -p "$BACKUP_DIR" "${APP_DIR}/logs"

echo "[INFO] Starting nightly backup: ${DATE_STR}..."

# Source environment for S3 bucket and credentials
if [ -f "${APP_DIR}/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    source "${APP_DIR}/.env"
    set +a
fi

S3_BUCKET="${BACKUP_S3_BUCKET:-quant-platform-backup}"

# 1. Dump PostgreSQL databases
echo "[INFO] Exporting PostgreSQL databases..."
docker compose -f "$COMPOSE_FILE" exec -T postgres \
    pg_dump -U "${POSTGRES_USER:-postgres}" -Fc quantdb > "${BACKUP_DIR}/quantdb_${DATE_STR}.dump"

docker compose -f "$COMPOSE_FILE" exec -T postgres \
    pg_dump -U "${POSTGRES_USER:-postgres}" -Fc mlflowdb > "${BACKUP_DIR}/mlflowdb_${DATE_STR}.dump" 2>/dev/null || true

# 2. Archive data lake and model artifacts (excluding cache/tmp)
echo "[INFO] Archiving data lake and trained models..."
tar --exclude="*.pyc" --exclude="__pycache__" --exclude=".cache" \
    -czf "${BACKUP_DIR}/data_models_${DATE_STR}.tar.gz" \
    -C "${APP_DIR}" data models

# 3. Upload to S3 if AWS CLI is available
if command -v aws &>/dev/null && [ -n "${S3_BUCKET:-}" ]; then
    echo "[INFO] Uploading backup artifacts to s3://${S3_BUCKET}/backups/${DATE_STR}/..."
    aws s3 sync "$BACKUP_DIR" "s3://${S3_BUCKET}/backups/${DATE_STR}/" --only-show-errors

    # Verify upload
    aws s3api head-object \
        --bucket "$S3_BUCKET" \
        --key "backups/${DATE_STR}/quantdb_${DATE_STR}.dump" >/dev/null
    echo "[INFO] S3 backup upload verified successfully."
fi

# Write success marker
echo "SUCCESS: Backup completed at $(date -u +'%Y-%m-%dT%H:%M:%SZ') (tag: ${DATE_STR})" > "$MARKER_FILE"

# Clean up local temporary files
rm -rf "$BACKUP_DIR"
echo "[INFO] Nightly backup finished cleanly."
