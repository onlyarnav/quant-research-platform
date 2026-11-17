variable "project_name" {
  description = "Unique identifier for platform resources and tags."
  type        = string
  default     = "quant-platform"
}

variable "environment" {
  description = "Deployment target environment (dev, staging, prod)."
  type        = string
  default     = "prod"
}

variable "aws_region" {
  description = "AWS deployment region."
  type        = string
  default     = "ap-south-1"
}

variable "instance_type" {
  description = "EC2 instance sizing based on backtest & ML memory benchmarks."
  type        = string
  default     = "t3.large"
}

variable "vpc_id" {
  description = "Custom VPC ID. If null, default VPC is resolved automatically."
  type        = string
  default     = null
}

variable "admin_cidr" {
  description = "Optional CIDR block for direct SSH access. Defaults to null (SSM Session Manager only)."
  type        = string
  default     = null
}

variable "data_volume_gb" {
  description = "Size of persistent EBS data volume for data lake, models, and PostgreSQL in GB."
  type        = number
  default     = 60
}

variable "domain" {
  description = "Public domain name pointing to the Elastic IP (e.g. quant.example.com)."
  type        = string
  default     = ""
}

variable "hosted_zone_id" {
  description = "Route 53 hosted zone ID for creating automated DNS records (optional)."
  type        = string
  default     = ""
}

variable "alert_email" {
  description = "Email address for CloudWatch alarms and AWS Budget notifications."
  type        = string
  default     = "admin@example.com"
}

variable "monthly_budget_usd" {
  description = "Monthly cost budget threshold in USD for spending alerts."
  type        = number
  default     = 60.0
}

variable "log_retention_days" {
  description = "CloudWatch log group retention period in days."
  type        = number
  default     = 30
}
