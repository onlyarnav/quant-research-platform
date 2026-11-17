output "public_ip" {
  description = "Public Elastic IP assigned to the Quant Platform instance."
  value       = aws_eip.quant_eip.public_ip
}

output "instance_id" {
  description = "EC2 instance ID."
  value       = aws_instance.quant_server.id
}

output "ecr_repository_url" {
  description = "Amazon ECR repository URI for pushing application images."
  value       = aws_ecr_repository.app.repository_url
}

output "s3_backup_bucket" {
  description = "S3 bucket for nightly encrypted database and data lake backups."
  value       = aws_s3_bucket.backup.id
}

output "ssm_connect_command" {
  description = "Command to open an interactive secure terminal on the EC2 instance via AWS SSM."
  value       = "aws ssm start-session --target ${aws_instance.quant_server.id} --region ${var.aws_region}"
}

output "app_url" {
  description = "Primary URL for the web research platform."
  value       = var.domain != "" ? "https://${var.domain}" : "https://${replace(aws_eip.quant_eip.public_ip, ".", "-")}.sslip.io"
}

output "next_steps" {
  description = "Exact next steps to finalize and deploy to AWS."
  value       = <<-EOT
    1. Populate secrets in AWS SSM Parameter Store:
       aws ssm put-parameter --name "/quant/${var.environment}/POSTGRES_PASSWORD" --value "<pwd>" --type "SecureString" --overwrite
       aws ssm put-parameter --name "/quant/${var.environment}/DASHBOARD_PASSWORD" --value "<pwd>" --type "SecureString" --overwrite
       aws ssm put-parameter --name "/quant/${var.environment}/DASHBOARD_API_KEY" --value "<key>" --type "SecureString" --overwrite

    2. Push production image and release bundle from your machine:
       ./deploy/aws/scripts/push_release.ps1 -Tag v1.0.0

    3. Connect to the host and run deployment:
       aws ssm start-session --target ${aws_instance.quant_server.id}
       sudo /opt/quant/deploy.sh v1.0.0
  EOT
}
