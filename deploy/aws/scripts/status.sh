#!/usr/bin/env bash
set -euo pipefail

# Status inspection script for Quant Research Platform
APP_DIR="/opt/quant"
COMPOSE_FILE="${APP_DIR}/docker-compose.prod.yml"

echo "============================================================"
echo "QUANT RESEARCH PLATFORM — SYSTEM STATUS"
echo "Host: $(hostname) | Date: $(date -u +'%Y-%m-%dT%H:%M:%SZ')"
echo "============================================================"

echo "--- 1. Container Status ---"
if command -v docker &>/dev/null && [ -f "$COMPOSE_FILE" ]; then
    docker compose -f "$COMPOSE_FILE" ps
else
    echo "Docker compose not available or compose file missing."
fi

echo ""
echo "--- 2. Storage Utilization (/opt/quant) ---"
df -h /opt/quant 2>/dev/null || df -h /

echo ""
echo "--- 3. System Memory ---"
free -h

echo ""
echo "--- 4. Active Releases ---"
echo "Current Tag:  $(cat ${APP_DIR}/releases/current 2>/dev/null || echo 'None')"
echo "Previous Tag: $(cat ${APP_DIR}/releases/previous 2>/dev/null || echo 'None')"

echo ""
echo "--- 5. Last Backup Status ---"
cat "${APP_DIR}/logs/last_backup_status" 2>/dev/null || echo "No backup status recorded."

echo ""
echo "============================================================"
