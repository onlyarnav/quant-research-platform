data "aws_caller_identity" "current" {}

resource "aws_iam_role" "quant_role" {
  name = "${var.project_name}-${var.environment}-role"

  assume_role_policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      {
        Action = "sts:AssumeRole"
        Effect = "Allow"
        Principal = {
          Service = "ec2.amazonaws.com"
        }
      }
    ]
  })
}

# Attach AWS Systems Manager managed policy for passwordless shell access
resource "aws_iam_role_policy_attachment" "ssm_core" {
  role       = aws_iam_role.quant_role.name
  policy_arn = "arn:aws:iam::aws:policy/AmazonSSMManagedInstanceCore"
}

resource "aws_iam_policy" "quant_least_privilege" {
  name        = "${var.project_name}-${var.environment}-least-privilege"
  description = "Least privilege permissions for Quant Platform EC2 host"

  policy = jsonencode({
    Version = "2012-10-17"
    Statement = [
      # SSM Parameter Store: Read environment secrets
      {
        Sid    = "SSMSecretsRead"
        Effect = "Allow"
        Action = [
          "ssm:GetParameter",
          "ssm:GetParameters",
          "ssm:GetParametersByPath"
        ]
        Resource = "arn:aws:ssm:${var.aws_region}:${data.aws_caller_identity.current.account_id}:parameter/quant/${var.environment}/*"
      },
      # S3: Backups and Deploy Bundles
      {
        Sid    = "S3BackupAndReleasesAccess"
        Effect = "Allow"
        Action = [
          "s3:PutObject",
          "s3:GetObject",
          "s3:ListBucket",
          "s3:DeleteObject"
        ]
        Resource = [
          aws_s3_bucket.backup.arn,
          "${aws_s3_bucket.backup.arn}/*"
        ]
      },
      # ECR: Image Pulling
      {
        Sid      = "ECRAuthToken"
        Effect   = "Allow"
        Action   = "ecr:GetAuthorizationToken"
        Resource = "*"
      },
      {
        Sid    = "ECRImagePull"
        Effect = "Allow"
        Action = [
          "ecr:BatchCheckLayerAvailability",
          "ecr:GetDownloadUrlForLayer",
          "ecr:BatchGetImage"
        ]
        Resource = aws_ecr_repository.app.arn
      },
      # CloudWatch Logs
      {
        Sid    = "CloudWatchLogsWrite"
        Effect = "Allow"
        Action = [
          "logs:CreateLogStream",
          "logs:PutLogEvents",
          "logs:DescribeLogStreams"
        ]
        Resource = "${aws_cloudwatch_log_group.app_logs.arn}:*"
      }
    ]
  })
}

resource "aws_iam_role_policy_attachment" "quant_policy_attach" {
  role       = aws_iam_role.quant_role.name
  policy_arn = aws_iam_policy.quant_least_privilege.arn
}

resource "aws_iam_instance_profile" "quant_profile" {
  name = "${var.project_name}-${var.environment}-profile"
  role = aws_iam_role.quant_role.name
}
