#!/usr/bin/env bash
set -euo pipefail

# Build, tag, and push Quant Platform release to Amazon ECR and S3
# Usage: ./push_release.sh [TAG] [REGION]

TAG="${1:-$(git rev-parse --short HEAD)}"
REGION="${2:-ap-south-1}"

echo "============================================================"
echo "QUANT PLATFORM — PUSH RELEASE BUNDLE (${TAG})"
echo "============================================================"

ACCOUNT_ID=$(aws sts get-caller-identity --query "Account" --output text)
ECR_REPO="${ACCOUNT_ID}.dkr.ecr.${REGION}.amazonaws.com/quant-platform"
IMAGE_URI="${ECR_REPO}:${TAG}"

# 1. Build image
echo "[1/4] Building production image (${IMAGE_URI})..."
docker build -t "$IMAGE_URI" -t "${ECR_REPO}:latest" .

# 2. Authenticate and push to ECR
echo "[2/4] Authenticating with Amazon ECR..."
aws ecr get-login-password --region "$REGION" | docker login --username AWS --password-stdin "$ECR_REPO"

echo "[3/4] Pushing image to ECR..."
docker push "$IMAGE_URI"
docker push "${ECR_REPO}:latest"

# 3. Create release bundle zip and upload to S3
echo "[4/4] Packaging and uploading release bundle..."
TMP_BUNDLE="/tmp/quant_release_${TAG}.zip"
zip -r "$TMP_BUNDLE" \
    docker-compose.prod.yml \
    deploy/aws/Caddyfile \
    deploy/docker/init-db.sh \
    deploy/aws/scripts/

BUCKET="quant-platform-backup-${ACCOUNT_ID}"
aws s3 cp "$TMP_BUNDLE" "s3://${BUCKET}/releases/${TAG}.zip"
rm -f "$TMP_BUNDLE"

echo "============================================================"
echo "RELEASE PUSH COMPLETE: ${TAG}"
echo "To deploy on server:"
echo "  aws ssm start-session --target <INSTANCE_ID> --region ${REGION}"
echo "  sudo /opt/quant/deploy.sh ${TAG}"
echo "============================================================"
