# Terraform Infrastructure for Quant Research Platform on AWS

This module provisions a production-ready, low-cost AWS environment for the Quantitative Research Platform on a single `t3.large` EC2 instance behind an HTTPS reverse proxy (Caddy), backed by encrypted EBS, S3 backups, ECR, SSM Parameter Store, and CloudWatch.

## Architecture

- **Compute:** 1x EC2 (`t3.large` Ubuntu 24.04 LTS), IMDSv2 enforced, encrypted root volume (30GB).
- **Persistent State:** Encrypted gp3 EBS volume (60GB, `prevent_destroy = true`) mounted at `/opt/quant`.
- **Security:** Inbound ports 80/443 only. No open SSH; access via AWS Systems Manager Session Manager.
- **Secrets:** AWS SSM Parameter Store (`SecureString`) under `/quant/<env>/*`.
- **Storage & Backup:** Private S3 bucket with versioning and TLS-only policy for nightly backups.
- **Registry:** Private Amazon ECR repository with vulnerability scanning and lifecycle pruning.
- **Monitoring & Cost:** CloudWatch Logs, metric alarms (CPU, Disk, Status), SNS alerts, and an AWS Budget alarm ($60/mo).

## Deployment Steps

1. Configure variables:
   ```bash
   cp terraform.tfvars.example terraform.tfvars
   ```
2. Initialize and plan:
   ```bash
   terraform init
   terraform plan -out=tfplan
   ```
3. Apply:
   ```bash
   terraform apply tfplan
   ```
4. Follow the commands printed in `terraform output next_steps`.
