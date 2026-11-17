#!/usr/bin/env bash
set -euo pipefail

# Deployment script for Quant Platform on AWS EC2
# Usage: ./deploy.sh <IMAGE_TAG> [BUNDLE_S3_KEY]

IMAGE_TAG="${1:-}"
if [ -z "$IMAGE_TAG" ]; then
    echo "Usage: $0 <IMAGE_TAG> [BUNDLE_S3_KEY]" >&2
    exit 1
fi

BUNDLE_KEY="${2:-}"
APP_DIR="/opt/quant"
COMPOSE_FILE="${APP_DIR}/docker-compose.prod.yml"
RELEASES_DIR="${APP_DIR}/releases"
CURRENT_RELEASE_FILE="${RELEASES_DIR}/current"
PREVIOUS_RELEASE_FILE="${RELEASES_DIR}/previous"
DEPLOY_LOG="${APP_DIR}/logs/deploy_$(date +%Y%m%d_%H%M%S).log"

mkdir -p "$RELEASES_DIR" "${APP_DIR}/logs"
exec > >(tee -a "$DEPLOY_LOG") 2>&1

echo "============================================================"
echo "DEPLOYING RELEASE: ${IMAGE_TAG} ($(date -u +'%Y-%m-%dT%H:%M:%SZ'))"
echo "============================================================"

# Record previous tag for automated rollback
if [ -f "$CURRENT_RELEASE_FILE" ]; then
    cp "$CURRENT_RELEASE_FILE" "$PREVIOUS_RELEASE_FILE"
fi

export IMAGE_TAG
export ENV_FILE="${APP_DIR}/.env"

# Fetch latest secrets if SSM script is present
if [ -f "${APP_DIR}/fetch_secrets.sh" ]; then
    bash "${APP_DIR}/fetch_secrets.sh" prod || true
fi

# 1. ECR Login if AWS CLI is available
if command -v aws &>/dev/null && [ -n "${AWS_DEFAULT_REGION:-}" ]; then
    echo "[INFO] Authenticating Docker with Amazon ECR..."
    aws ecr get-login-password --region "${AWS_DEFAULT_REGION:-ap-south-1}" | \
        docker login --username AWS --password-stdin "${ECR_REPO:-}" || true
fi

# 2. Pull production images
echo "[INFO] Pulling container images..."
docker compose -f "$COMPOSE_FILE" pull

# 3. Run database migrations
echo "[INFO] Executing database migrations..."
if ! docker compose -f "$COMPOSE_FILE" run --rm migrate; then
    echo "[ERROR] Database migration failed!" >&2
    exit 1
fi

# 4. Bring up production stack
echo "[INFO] Starting production containers..."
docker compose -f "$COMPOSE_FILE" up -d

# 5. Wait for health checks
echo "[INFO] Waiting for service health probes..."
MAX_WAIT=90
ELAPSED=0
HEALTHY=false

while [ "$ELAPSED" -lt "$MAX_WAIT" ]; do
    API_STATUS=$(docker inspect --format='{{json .State.Health.Status}}' quant-api 2>/dev/null || echo '"unknown"')
    DASH_STATUS=$(docker inspect --format='{{json .State.Health.Status}}' quant-dashboard 2>/dev/null || echo '"unknown"')

    if [ "$API_STATUS" = '"healthy"' ] && [ "$DASH_STATUS" = '"healthy"' ]; then
        HEALTHY=true
        break
    fi

    sleep 5
    ELAPSED=$((ELAPSED + 5))
    echo "  Waiting for health... (${ELAPSED}s / ${MAX_WAIT}s)"
done

if [ "$HEALTHY" != "true" ]; then
    echo "[ERROR] Containers failed healthcheck after ${MAX_WAIT}s. Triggering rollback..." >&2
    if [ -f "${APP_DIR}/rollback.sh" ]; then
        bash "${APP_DIR}/rollback.sh"
    fi
    exit 1
fi

# 6. Execute container smoke test
echo "[INFO] Running post-deploy smoke test..."
if ! docker compose -f "$COMPOSE_FILE" exec -T api python scripts/smoke_test.py --url http://127.0.0.1:8000; then
    echo "[ERROR] Smoke tests failed! Triggering automatic rollback..." >&2
    if [ -f "${APP_DIR}/rollback.sh" ]; then
        bash "${APP_DIR}/rollback.sh"
    fi
    exit 1
fi

# Record current successful tag
echo "$IMAGE_TAG" > "$CURRENT_RELEASE_FILE"

echo "============================================================"
echo "DEPLOYMENT SUCCESSFUL: ${IMAGE_TAG}"
echo "============================================================"
