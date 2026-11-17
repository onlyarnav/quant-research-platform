# Changelog

All notable changes to the Quantitative AI Research Platform are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-11-17

### Added
- **AWS Production Deployment:** Complete Terraform infrastructure (`deploy/aws/terraform/`) provisioning EC2 `t3.large`, encrypted EBS, S3 backup buckets, ECR, SSM Parameter Store secrets, and CloudWatch alarms.
- **Production Docker Stack:** Multi-stage `Dockerfile` (Debian 12 Bookworm, non-root user `quant`), `docker-compose.prod.yml`, and `Caddyfile` HTTPS reverse proxy with automated TLS.
- **Operator Automation Scripts:** Automated server bootstrap, secret sync from SSM (`fetch_secrets.sh`), zero-downtime deployment with automatic rollback (`deploy.sh`), nightly S3 backups (`backup.sh`), and restore drills (`restore.sh`).
- **Data Contract Validator:** `src/infra/contract_validator.py` and CLI `scripts/validate_data_contracts.py` ensuring Parquet data lake files conform strictly to canonical specifications.
- **Automated Verification Runner:** `scripts/verify_all.py` providing unified one-command execution for static checks, full tests, Docker specs, and end-to-end dry runs.
- **HTTP Smoke Testing:** `scripts/smoke_test.py` validating liveness, readiness, disclaimer routes, and HTTP security headers post-deployment.
- **Performance Benchmarking:** `scripts/benchmark_scale.py` measuring memory RSS and wall time across large synthetic datasets.
- **Regulatory & Legal Safeguards:** Added `DISCLAIMER.md`, dashboard password gate (`DASHBOARD_PASSWORD`), and read-only mode (`DASHBOARD_READ_ONLY`).

### Changed
- **File Length Optimization:** Refactored all source and script files to strictly $\le 300$ lines across `src/`, `config/`, and `scripts/`.
- **Dependency Modernization:** Pinned all package versions in `requirements/constraints.txt` and separated dev tools into `[project.optional-dependencies] dev`.
- **Packaging:** Bumped package version to `1.0.0` in `pyproject.toml`.
- **Database Engine:** Added `pool_pre_ping=True` and `pool_recycle=3600` for connection resilience.

### Removed
- **Unused Dependencies:** Removed `fredapi`, `newsapi-python`, `tenacity`, `quantstats`, and `pandas-ta`.
- **Dead Code:** Cleaned obsolete legacy modules and imports flagged by `vulture`.

### Fixed
- Fixed unmasked credentials in logs and database URLs via `mask_secrets()`.
- Fixed job persistence across restarts in `BacktestJobManager`.
- Fixed `ModelStore.exists()` keyword argument call in pipeline steps.
- Fixed `IngestionReport` integer counting in orchestration adapter.
