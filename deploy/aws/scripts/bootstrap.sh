#!/usr/bin/env bash
set -euo pipefail

# Server bootstrap script for Quant Research Platform
# Idempotent setup of directories, permissions, Docker, and systemd units

echo "[INFO] Running Quant Platform bootstrap..."

# 1. Non-root user setup
if ! id "quant" &>/dev/null; then
    groupadd -g 10001 quant
    useradd -u 10001 -g quant -m -d /home/quant -s /bin/bash quant
fi

# 2. Directory structure under /opt/quant
mkdir -p /opt/quant/{data,models,pgdata,mlflow-artifacts,logs,caddy,releases}
chown -R 10001:10001 /opt/quant
chmod 750 /opt/quant

# 3. Memory optimization (vm.swappiness = 10 for database workloads)
sysctl -w vm.swappiness=10
if ! grep -q "vm.swappiness=10" /etc/sysctl.conf; then
    echo "vm.swappiness=10" >> /etc/sysctl.conf
fi

# 4. Install systemd services
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
if [ -d "$SCRIPT_DIR" ]; then
    cp "$SCRIPT_DIR"/quant*.service /etc/systemd/system/ 2>/dev/null || true
    cp "$SCRIPT_DIR"/quant*.timer /etc/systemd/system/ 2>/dev/null || true
    systemctl daemon-reload
    systemctl enable quant.service 2>/dev/null || true
    systemctl enable quant-backup.timer 2>/dev/null || true
fi

echo "[INFO] Quant Platform bootstrap completed."
