# Quantitative AI Research Platform — Issue & Resolution Log

**Version:** 1.0.0  
**Updated:** November 17, 2026  
**Tracking Protocol:** Severity ratings defined as Critical, High, Medium, Low.

---

## Issue Summary Matrix

| ID | Severity | Area | Symptom | Root Cause | Resolution | Status |
|---|---|---|---|---|---|---|
| ISS-001 | High | Settings | `pydantic` validation crash on optional API keys | Empty strings treated as invalid by Pydantic | Added `validate_optional_str` pre-validator | FIXED |
| ISS-002 | Medium | Dashboard API | Flake8 F821 undefined name `Any` in `main.py` | Missing `from typing import Any` import | Added `from typing import Any` | FIXED |
| ISS-003 | Medium | Backtesting | Mypy error on `abs()` and `None` comparison in `backtest_config.py` | Optional config fields unresolved in type system | Added assertions in `validate()` guaranteeing non-None | FIXED |
| ISS-004 | High | Dashboard | `JobManager` in-memory jobs lost on container restart | In-memory dict storage without persistence | Rewrote with atomic JSON disk persistence & recovery | FIXED |
| ISS-005 | High | Security | Cleartext passwords in logs and URLs | Unmasked log statements | Implemented `mask_secrets()` regex redaction | FIXED |
| ISS-006 | Medium | Ingestion | Incompatible type in `IngestionReport` length check | `report.succeeded` was `int`, called `len(report.succeeded)` | Changed to `report.succeeded` direct integer return | FIXED |
| ISS-007 | High | Models | `ModelStore.exists(name=...)` raised unexpected keyword argument | Method takes `model_name`, not `name` | Fixed caller argument to `model_name` | FIXED |
| ISS-008 | Medium | Data Lake | No contract validation against `Data_Schema.md` | Parquet files could drift without detection | Built `src/infra/contract_validator.py` and CLI | FIXED |
| ISS-009 | High | Streamlit | Dashboard routes and backtests unprotected in production | No password gate or read-only mode | Added `DASHBOARD_PASSWORD` gate and `DASHBOARD_READ_ONLY` | FIXED |
| ISS-010 | Low | Linting | Black formatting drift across 24 files | Formatting divergence during iterative coding | Ran `black --line-length 88` and `isort` repo-wide | FIXED |
| ISS-011 | Medium | Docker | `docker-compose.prod.yml` failed `config` on Windows | Missing host path `/opt/quant/.env` | Updated `env_file` to `${ENV_FILE:-.env}` with `required: false` | FIXED |
| ISS-012 | High | API | Missing security headers on HTTP responses | FastAPI responses lacked HSTS, nosniff, DENY | Added HTTP middleware and reverse proxy headers | FIXED |

---

## Detailed Findings & Fixes

### ISS-004: Job Manager State Persistence Across Restarts
- **Severity:** High
- **Area:** `src/dashboard/job_manager.py`
- **Symptom:** Long-running backtest jobs became orphaned when the API container restarted or crashed; status polling returned 404.
- **Root Cause:** Jobs were stored in an in-memory dictionary without on-disk state synchronization.
- **Fix:** Refactored `BacktestJobManager` to persist job metadata to individual JSON files under `data/backtests/_jobs/` using atomic writes (`tempfile` + `os.replace`). Added startup recovery routine that identifies interrupted `running` jobs and marks them as `failed`.
- **Regression Test:** `tests/test_production_readiness.py::test_job_manager_persistence`.

### ISS-005: Sensitive Credential Masking in System Logs
- **Severity:** High
- **Area:** `src/utils/logger.py`
- **Symptom:** Database connection strings containing passwords (`postgresql://postgres:password@...`) and API keys could leak into CloudWatch logs.
- **Root Cause:** Standard string formatting emitted raw connection URLs.
- **Fix:** Implemented `mask_secrets()` using regex filters targeting database URLs (`://[^:]+:([^@]+)@`) and bearer/API keys (`(key|token|password)[:=]\s*([^\s]+)`), replacing credentials with `***`. Applied filter to custom log formatter.
- **Regression Test:** `tests/test_production_readiness.py::test_logger_mask_secrets`.

### ISS-008: Parquet Data Lake Contract Validation
- **Severity:** Medium
- **Area:** `src/infra/contract_validator.py` & `scripts/validate_data_contracts.py`
- **Symptom:** Schema drift or missing columns in Parquet datasets could propagate downstream into model training and simulator crashes.
- **Fix:** Implemented automated schema validator comparing actual Parquet Arrow schemas against canonical specifications in `docs/Data Schema.md`. Detects missing required columns, zero-row datasets, and corrupt files.
- **Regression Test:** `tests/test_production_readiness.py::test_contract_validator_file`.
