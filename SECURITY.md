# Security Policy

## Supported Versions

The following versions of the Quantitative AI Research Platform are currently supported with security updates:

| Version | Supported |
|---|---|
| 1.0.x | :white_check_mark: |
| < 1.0.0 | :x: |

## Reporting a Vulnerability

If you discover a security vulnerability within this repository or associated deployment configurations, please follow responsible disclosure guidelines:

1. **Do not create public GitHub issues** for security vulnerabilities.
2. Email the vulnerability report directly to the project maintainers with details, reproduction steps, and potential impact.
3. You will receive an acknowledgment within 48 hours and regular updates as a fix is developed and released.

## Security Architecture & Best Practices

- **Zero Hardcoded Credentials:** Plaintext secrets, tokens, and database passwords must never be committed to git. Use `.env` locally (untracked) and AWS SSM Parameter Store in production.
- **Least-Privilege Execution:** Containers execute as an unprivileged non-root user (`quant`, UID 10001) with `no-new-privileges:true` and dropped Linux capabilities.
- **Network Isolation:** Internal services (PostgreSQL, MLflow) are never bound to public host ports. All external traffic is routed through the HTTPS reverse proxy (Caddy).
- **Log Credential Redaction:** Sensitive tokens, passwords, and connection strings are masked dynamically via `mask_secrets()` before emission.
- **Automated Security Scanning:** Static analysis (`bandit`), dependency audits (`pip-audit`), and container vulnerability checks are integrated into CI/CD.
