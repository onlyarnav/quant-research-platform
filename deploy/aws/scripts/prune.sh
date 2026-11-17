#!/usr/bin/env bash
set -euo pipefail

# Docker and log retention cleanup script
# Run periodically via cron or maintenance window

echo "[INFO] Pruning unused Docker resources..."
docker image prune -af --filter "until=168h" || true
docker container prune -f || true
docker volume prune -f || true

echo "[INFO] Cleaning old application logs older than 30 days..."
find /opt/quant/logs -type f -name "*.log" -mtime +30 -delete 2>/dev/null || true

echo "[INFO] System prune finished."
