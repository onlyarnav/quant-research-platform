#!/usr/bin/env bash
set -euo pipefail

# Fetch production secrets from AWS SSM Parameter Store into /opt/quant/.env
# Never echoes secrets to stdout or logs.

ENV_NAME="${1:-prod}"
TARGET_FILE="/opt/quant/.env"
TMP_FILE="/opt/quant/.env.tmp"

echo "[INFO] Fetching secrets for environment: ${ENV_NAME} from SSM Parameter Store..."

# Verify AWS CLI is installed
if ! command -v aws &>/dev/null; then
    echo "[ERROR] AWS CLI is required but not installed." >&2
    exit 1
fi

# Fetch all parameters under /quant/${ENV_NAME}/
PARAMS_JSON=$(aws ssm get-parameters-by-path \
    --path "/quant/${ENV_NAME}/" \
    --with-decryption \
    --query "Parameters[*].{Name:Name,Value:Value}" \
    --output json)

COUNT=$(echo "$PARAMS_JSON" | jq '. | length')
if [ "$COUNT" -eq 0 ]; then
    echo "[ERROR] No parameters found under /quant/${ENV_NAME}/ in SSM." >&2
    exit 1
fi

# Write key-value pairs atomically
: > "$TMP_FILE"
chmod 600 "$TMP_FILE"

echo "$PARAMS_JSON" | jq -c '.[]' | while read -r item; do
    FULL_NAME=$(echo "$item" | jq -r '.Name')
    VAL=$(echo "$item" | jq -r '.Value')
    KEY="${FULL_NAME##*/}"
    echo "${KEY}=${VAL}" >> "$TMP_FILE"
done

# Check required keys exist
REQUIRED_KEYS=("POSTGRES_PASSWORD" "DASHBOARD_PASSWORD" "DASHBOARD_API_KEY")
for k in "${REQUIRED_KEYS[@]}"; do
    if ! grep -q "^${k}=" "$TMP_FILE"; then
        echo "[ERROR] Missing required secret parameter: ${k}" >&2
        rm -f "$TMP_FILE"
        exit 1
    fi
done

mv "$TMP_FILE" "$TARGET_FILE"
chmod 600 "$TARGET_FILE"
chown 10001:10001 "$TARGET_FILE"

echo "[INFO] Successfully wrote ${COUNT} secrets to ${TARGET_FILE} (mode 600)."
