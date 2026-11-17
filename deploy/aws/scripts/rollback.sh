#!/usr/bin/env bash
set -euo pipefail

# Rollback deployment script for Quant Platform on AWS EC2
# Usage: ./rollback.sh [TARGET_IMAGE_TAG]

APP_DIR="/opt/quant"
COMPOSE_FILE="${APP_DIR}/docker-compose.prod.yml"
PREVIOUS_RELEASE_FILE="${APP_DIR}/releases/previous"

TARGET_TAG="${1:-}"
if [ -z "$TARGET_TAG" ]; then
    if [ -f "$PREVIOUS_RELEASE_FILE" ]; then
        TARGET_TAG=$(cat "$PREVIOUS_RELEASE_FILE")
    else
        echo "[ERROR] No previous release tag recorded to roll back to!" >&2
        exit 1
    fi
fi

echo "============================================================"
echo "ROLLING BACK TO TAG: ${TARGET_TAG}"
echo "============================================================"

export IMAGE_TAG="$TARGET_TAG"
export ENV_FILE="${APP_DIR}/.env"

# Re-deploy previous tag
docker compose -f "$COMPOSE_FILE" up -d --remove-orphans

echo "[INFO] Rollback completed to ${TARGET_TAG}."
