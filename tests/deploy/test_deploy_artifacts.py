"""
Tests for AWS deployment artifacts and configuration validation.

Runs offline without Docker or AWS credentials:
- Validates docker-compose.prod.yml security & architecture rules
- Validates deploy/aws/Caddyfile headers and routes
- Validates shell script hygiene (LF, bash shebang, strict mode)
- Validates Terraform structure, variables, and lack of plaintext secrets
- Validates .env.production.example completeness
- Validates Dockerfile non-root security
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent.parent


def test_dockerfile_security() -> None:
    """Validate Dockerfile adheres to non-root and pinned image standards."""
    df_path = REPO_ROOT / "Dockerfile"
    assert df_path.exists(), "Dockerfile must exist at repository root"
    content = df_path.read_text(encoding="utf-8")

    # Check pinned base image
    assert (
        "python:3.12-slim-bookworm" in content
    ), "Must use pinned Debian bookworm image"
    # Check non-root USER
    assert (
        "USER 10001:10001" in content or "USER quant" in content
    ), "Must switch to non-root user"
    # Ensure no dangerous ADD from URLs
    assert not re.search(
        r"^\s*ADD\s+http", content, re.MULTILINE
    ), "ADD from HTTP is prohibited"
    # Ensure healthcheck is defined
    assert "HEALTHCHECK" in content, "Dockerfile must define a HEALTHCHECK"


def test_compose_prod_security_and_ports() -> None:
    """Validate docker-compose.prod.yml architecture and security rules."""
    compose_path = REPO_ROOT / "docker-compose.prod.yml"
    assert compose_path.exists(), "docker-compose.prod.yml must exist"
    data = yaml.safe_load(compose_path.read_text(encoding="utf-8"))

    services = data.get("services", {})
    assert "postgres" in services
    assert "api" in services
    assert "dashboard" in services
    assert "proxy" in services

    # Ensure PostgreSQL and MLflow do NOT expose host ports
    assert "ports" not in services["postgres"], "Postgres must not expose host ports"
    if "mlflow" in services:
        assert "ports" not in services["mlflow"], "MLflow must not expose host ports"

    # Only proxy publishes public ports 80/443
    proxy_ports = services["proxy"].get("ports", [])
    assert any("80:80" in str(p) for p in proxy_ports)
    assert any("443:443" in str(p) for p in proxy_ports)

    # Ensure security_opt on containers
    for svc_name in ["postgres", "api", "dashboard", "scheduler", "proxy"]:
        if svc_name in services:
            sec_opt = services[svc_name].get("security_opt", [])
            assert (
                "no-new-privileges:true" in sec_opt
            ), f"{svc_name} missing no-new-privileges"


def test_caddyfile_security_headers_and_routes() -> None:
    """Validate Caddyfile routing and production security headers."""
    caddy_path = REPO_ROOT / "deploy" / "aws" / "Caddyfile"
    assert caddy_path.exists(), "Caddyfile must exist"
    content = caddy_path.read_text(encoding="utf-8")

    # Security headers
    assert "Strict-Transport-Security" in content
    assert "X-Content-Type-Options" in content
    assert "X-Frame-Options" in content
    assert "Referrer-Policy" in content

    # Routing targets
    assert "reverse_proxy api:8000" in content
    assert "reverse_proxy dashboard:8501" in content


def test_shell_scripts_hygiene() -> None:
    """Verify all bash scripts have bash shebang, strict mode, and LF line endings."""
    scripts = list((REPO_ROOT / "deploy" / "aws" / "scripts").glob("*.sh"))
    scripts.extend(list((REPO_ROOT / "docker").glob("*.sh")))
    assert len(scripts) >= 5, "Expected deploy scripts"

    for script in scripts:
        raw_bytes = script.read_bytes()
        assert (
            b"\r\n" not in raw_bytes
        ), f"{script.name} must use LF line endings, not CRLF"

        text = raw_bytes.decode("utf-8")
        assert text.startswith("#!/usr/bin/env bash") or text.startswith(
            "#!/bin/bash"
        ), f"{script.name} missing standard bash shebang"
        assert (
            "set -euo pipefail" in text or "set -e" in text
        ), f"{script.name} missing strict mode (set -euo pipefail)"


def test_terraform_variables_and_no_plaintext_secrets() -> None:
    """Verify Terraform configuration variables and ensure no hardcoded secrets."""
    tf_dir = REPO_ROOT / "deploy" / "aws" / "terraform"
    assert tf_dir.exists()

    var_file = tf_dir / "variables.tf"
    assert var_file.exists()
    var_content = var_file.read_text(encoding="utf-8")

    # Ensure all variables have description blocks
    var_blocks = re.findall(
        r'variable\s+"([^"]+)"\s+\{([^}]+)\}', var_content, re.DOTALL
    )
    for var_name, body in var_blocks:
        assert (
            "description" in body
        ), f"Terraform variable '{var_name}' missing description"

    # Scan all .tf files for plaintext password assignments
    for tf_file in tf_dir.glob("*.tf"):
        content = tf_file.read_text(encoding="utf-8")
        assert not re.search(
            r'password\s*=\s*"[A-Za-z0-9_!@#$%^&*]{6,}"', content, re.IGNORECASE
        ), f"Possible hardcoded password in {tf_file.name}"


def test_env_production_example() -> None:
    """Verify .env.production.example contains necessary production variables."""
    env_example = REPO_ROOT / ".env.production.example"
    assert env_example.exists()
    content = env_example.read_text(encoding="utf-8")

    required_vars = [
        "APP_ENV",
        "DOMAIN",
        "POSTGRES_PASSWORD",
        "DATABASE_URL",
        "DASHBOARD_PASSWORD",
        "DASHBOARD_API_KEY",
        "PIPELINE_RUN_TIME",
        "LOG_FORMAT",
    ]
    for var in required_vars:
        assert f"{var}=" in content, f"Missing {var} in .env.production.example"
