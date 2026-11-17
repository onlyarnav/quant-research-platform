data "aws_vpc" "selected" {
  default = var.vpc_id == null ? true : false
  id      = var.vpc_id
}

data "aws_subnets" "public" {
  filter {
    name   = "vpc-id"
    values = [data.aws_vpc.selected.id]
  }
}

resource "aws_security_group" "quant_sg" {
  name        = "${var.project_name}-${var.environment}-sg"
  description = "Security group for Quant Research Platform reverse proxy"
  vpc_id      = data.aws_vpc.selected.id

  # Inbound HTTP (Redirected to HTTPS by Caddy)
  ingress {
    description = "HTTP ingress for ACME challenges and redirect"
    from_port   = 80
    to_port     = 80
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Inbound HTTPS (Primary entrypoint)
  ingress {
    description = "HTTPS ingress for web UI and REST API"
    from_port   = 443
    to_port     = 443
    protocol    = "tcp"
    cidr_blocks = ["0.0.0.0/0"]
  }

  # Optional Direct SSH (SSM Session Manager preferred)
  dynamic "ingress" {
    for_each = var.admin_cidr != null ? [var.admin_cidr] : []
    content {
      description = "Admin SSH access"
      from_port   = 22
      to_port     = 22
      protocol    = "tcp"
      cidr_blocks = [ingress.value]
    }
  }

  # All Outbound Traffic Allowed
  egress {
    description = "Outbound access for market data feeds, S3, and ECR"
    from_port   = 0
    to_port     = 0
    protocol    = "-1"
    cidr_blocks = ["0.0.0.0/0"]
  }

  tags = {
    Name = "${var.project_name}-${var.environment}-sg"
  }
}

resource "aws_route53_record" "app" {
  count   = var.domain != "" && var.hosted_zone_id != "" ? 1 : 0
  zone_id = var.hosted_zone_id
  name    = var.domain
  type    = "A"
  ttl     = 300
  records = [aws_eip.quant_eip.public_ip]
}
