data "aws_ami" "ubuntu" {
  most_recent = true
  owners      = ["099720109477"] # Canonical

  filter {
    name   = "name"
    values = ["ubuntu/images/hvm-ssd-gp3/ubuntu-noble-24.04-amd64-server-*"]
  }

  filter {
    name   = "virtualization-type"
    values = ["hvm"]
  }
}

resource "aws_instance" "quant_server" {
  ami                  = data.aws_ami.ubuntu.id
  instance_type        = var.instance_type
  iam_instance_profile = aws_iam_instance_profile.quant_profile.name
  subnet_id            = tolist(data.aws_subnets.public.ids)[0]

  vpc_security_group_ids = [
    aws_security_group.quant_sg.id
  ]

  # Root block volume (Operating system, Docker binaries)
  root_block_device {
    volume_size           = 30
    volume_type           = "gp3"
    encrypted             = true
    delete_on_termination = true
  }

  # Enforce IMDSv2 for security hardening
  metadata_options {
    http_endpoint = "enabled"
    http_tokens   = "required"
  }

  user_data = templatefile("${path.module}/userdata.sh.tftpl", {
    aws_region    = var.aws_region
    project_name  = var.project_name
    environment   = var.environment
    backup_bucket = aws_s3_bucket.backup.id
  })

  tags = {
    Name = "${var.project_name}-${var.environment}-server"
  }
}

resource "aws_eip" "quant_eip" {
  instance = aws_instance.quant_server.id
  domain   = "vpc"

  tags = {
    Name = "${var.project_name}-${var.environment}-eip"
  }
}
