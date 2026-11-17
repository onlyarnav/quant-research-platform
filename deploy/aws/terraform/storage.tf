# Persistent Encrypted EBS Volume for Data Lake, Models & PostgreSQL
resource "aws_ebs_volume" "quant_data" {
  availability_zone = aws_instance.quant_server.availability_zone
  size              = var.data_volume_gb
  type              = "gp3"
  encrypted         = true

  lifecycle {
    prevent_destroy = true
  }

  tags = {
    Name = "${var.project_name}-${var.environment}-data"
  }
}

resource "aws_volume_attachment" "quant_data_attach" {
  device_name  = "/dev/xvdf"
  volume_id    = aws_ebs_volume.quant_data.id
  instance_id  = aws_instance.quant_server.id
  skip_destroy = true
}

# S3 Bucket for Nightly Backups and Release Bundles
resource "aws_s3_bucket" "backup" {
  bucket        = "${var.project_name}-backup-${data.aws_caller_identity.current.account_id}"
  force_destroy = false

  tags = {
    Name = "${var.project_name}-backup"
  }
}

resource "aws_s3_bucket_versioning" "backup" {
  bucket = aws_s3_bucket.backup.id
  versioning_configuration {
    status = "Enabled"
  }
}

resource "aws_s3_bucket_server_side_encryption_configuration" "backup" {
  bucket = aws_s3_bucket.backup.id
  rule {
    apply_server_side_encryption_by_default {
      sse_algorithm = "AES256"
    }
  }
}

resource "aws_s3_bucket_public_access_block" "backup" {
  bucket                  = aws_s3_bucket.backup.id
  block_public_acls       = true
  block_public_policy     = true
  ignore_public_acls      = true
  restrict_public_buckets = true
}

resource "aws_s3_bucket_policy" "deny_insecure_transport" {
  bucket = aws_s3_bucket.backup.id

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Sid       = "EnforceTLSRequestsOnly"
        Effect    = "Deny"
        Principal = "*"
        Action    = "s3:*"
        Resource = [
          aws_s3_bucket.backup.arn,
          "${aws_s3_bucket.backup.arn}/*"
        ]
        Condition = {
          Bool = {
            "aws:SecureTransport" = "false"
          }
        }
      }
    ]
  })
}

resource "aws_s3_bucket_lifecycle_configuration" "backup_retention" {
  bucket = aws_s3_bucket.backup.id

  rule {
    id     = "expire-nightly-backups"
    status = "Enabled"
    filter {
      prefix = "backups/"
    }
    expiration {
      days = 30
    }
  }
}

# ECR Container Repository
resource "aws_ecr_repository" "app" {
  name                 = var.project_name
  image_tag_mutability = "IMMUTABLE"

  image_scanning_configuration {
    scan_on_push = true
  }
}

resource "aws_ecr_lifecycle_policy" "cleanup" {
  repository = aws_ecr_repository.app.name

  policy = jsonencode({
    rules = [
      {
        rulePriority = 1
        description  = "Retain last 10 container images"
        selection = {
          tagStatus   = "any"
          countType   = "imageCountMoreThan"
          countNumber = 10
        }
        action = {
          type = "expire"
        }
      }
    ]
  })
}

# SSM SecureString Placeholders for Configuration & Secrets
locals {
  secret_names = [
    "POSTGRES_PASSWORD",
    "DASHBOARD_PASSWORD",
    "DASHBOARD_API_KEY",
    "ALPHA_VANTAGE_KEY"
  ]
}

resource "aws_ssm_parameter" "secrets" {
  for_each = toset(locals.secret_names)

  name        = "/quant/${var.environment}/${each.value}"
  type        = "SecureString"
  value       = "CHANGE_ME"
  description = "Managed secret placeholder for ${each.value}. Populate using AWS CLI."

  lifecycle {
    ignore_changes = [value]
  }
}
