<#
.SYNOPSIS
    Builds, tags, and pushes Quant Platform release to Amazon ECR and S3 from Windows.

.PARAMETER Tag
    Release tag or version (defaults to git short SHA).

.PARAMETER Region
    AWS deployment region (defaults to ap-south-1).
#>

[CmdletBinding()]
param(
    [string]$Tag = $(git rev-parse --short HEAD),
    [string]$Region = "ap-south-1"
)

$ErrorActionPreference = "Stop"

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "QUANT PLATFORM — PUSH RELEASE BUNDLE ($Tag)" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

# 1. Resolve Account ID and ECR URI
$AccountId = aws sts get-caller-identity --query "Account" --output text
if (-not $AccountId) {
    Write-Error "Failed to retrieve AWS account ID. Ensure AWS CLI is configured."
}

$EcrRepo = "$AccountId.dkr.ecr.$Region.amazonaws.com/quant-platform"
$ImageUri = "$EcrRepo`:$Tag"

# 2. Build Docker image
Write-Host "[1/5] Building Docker production image ($ImageUri)..." -ForegroundColor Yellow
docker build -t $ImageUri -t "$EcrRepo`:latest" .

# 3. Authenticate with ECR
Write-Host "[2/5] Authenticating Docker with Amazon ECR ($Region)..." -ForegroundColor Yellow
aws ecr get-login-password --region $Region | docker login --username AWS --password-stdin $EcrRepo

# 4. Push to ECR
Write-Host "[3/5] Pushing image to ECR..." -ForegroundColor Yellow
docker push $ImageUri
docker push "$EcrRepo`:latest"

# 5. Create and upload release bundle zip to S3
Write-Host "[4/5] Packaging release bundle..." -ForegroundColor Yellow
$BundleZip = "$env:TEMP\quant_release_$Tag.zip"
if (Test-Path $BundleZip) { Remove-Item $BundleZip -Force }

Compress-Archive -Path `
    docker-compose.prod.yml, `
    deploy\aws\Caddyfile, `
    deploy\docker\init-db.sh, `
    deploy\aws\scripts\* `
    -DestinationPath $BundleZip

$BackupBucket = "quant-platform-backup-$AccountId"
Write-Host "[5/5] Uploading release bundle to s3://$BackupBucket/releases/$Tag.zip..." -ForegroundColor Yellow
aws s3 cp $BundleZip "s3://$BackupBucket/releases/$Tag.zip"

Write-Host "============================================================" -ForegroundColor Green
Write-Host "RELEASE PACKAGED & PUSHED SUCCESSFULLY: $Tag" -ForegroundColor Green
Write-Host ""
Write-Host "To deploy this release on EC2, run:"
Write-Host "  aws ssm start-session --target <INSTANCE_ID> --region $Region" -ForegroundColor Cyan
Write-Host "  sudo /opt/quant/deploy.sh $Tag" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Green
