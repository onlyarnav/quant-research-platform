"""
Unit tests for production readiness components (Part E).

Tests:
- Settings validation and constraints
- Log secret masking and JSON formatting
- JobManager disk persistence and concurrency limits
- Data contract validation
- API system routes (/health, /ready, /disclaimer) and security headers
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from config.settings import settings
from config.settings_validators import (
    validate_cors_origins,
    validate_pipeline_run_time,
    validate_workers,
)
from src.dashboard.api.main import app
from src.dashboard.job_manager import BacktestJobManager
from src.infra.contract_validator import (
    validate_dataframe,
    validate_parquet_file,
)
from src.utils.logger import mask_secrets


def test_settings_validators() -> None:
    """Test environment variable validators."""
    assert validate_pipeline_run_time("08:30") == "08:30"
    assert validate_pipeline_run_time("23:59") == "23:59"
    with pytest.raises(ValueError, match="Invalid time format"):
        validate_pipeline_run_time("25:00")
    with pytest.raises(ValueError, match="Invalid time format"):
        validate_pipeline_run_time("invalid")

    assert validate_workers(4) == 4
    with pytest.raises(ValueError, match="API_WORKERS must be between 1 and 32"):
        validate_workers(0)
    with pytest.raises(ValueError, match="API_WORKERS must be between 1 and 32"):
        validate_workers(64)

    assert validate_cors_origins("http://localhost:8501,https://example.com") == [
        "http://localhost:8501",
        "https://example.com",
    ]


def test_logger_mask_secrets() -> None:
    """Test logger secret masking regex logic."""
    raw = "Postgres connection postgresql://admin:SuperSecret123@localhost:5432/quantdb"
    masked = mask_secrets(raw)
    assert "SuperSecret123" not in masked
    assert "admin:***@" in masked

    raw_token = "api_key: secret_token_xyz987"
    masked_token = mask_secrets(raw_token)
    assert "secret_token_xyz987" not in masked_token
    assert "***" in masked_token


def test_job_manager_persistence(tmp_path: Path) -> None:
    """Test BacktestJobManager disk persistence and stale job recovery."""
    mgr = BacktestJobManager(root=tmp_path)
    job_id = "job_test123"
    job_data = {
        "job_id": job_id,
        "status": "running",
        "run_id": None,
        "error": None,
        "created_at": "2026-01-01T00:00:00",
        "updated_at": "2026-01-01T00:00:00",
    }
    mgr._atomic_write_job(job_data)

    # Verify JSON file written to disk
    job_file = tmp_path / f"{job_id}.json"
    assert job_file.exists()
    status = mgr.get_status(job_id)
    assert status["status"] == "running"

    # Create new manager pointing to same storage - running jobs should recover to failed
    mgr2 = BacktestJobManager(root=tmp_path)
    recovered = mgr2.get_status(job_id)
    assert recovered is not None
    assert recovered["status"] == "failed"
    assert "interrupted" in (recovered.get("error") or "").lower()


def test_job_manager_concurrency(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Test BacktestJobManager enforces BACKTEST_MAX_CONCURRENT_JOBS limit."""
    monkeypatch.setattr(settings, "BACKTEST_MAX_CONCURRENT_JOBS", 1)
    mgr = BacktestJobManager(root=tmp_path)

    # Write an active running job
    active_job = {
        "job_id": "job_active",
        "status": "running",
        "run_id": None,
        "error": None,
        "created_at": "2026-01-01T00:00:00",
        "updated_at": "2026-01-01T00:00:00",
    }
    mgr._atomic_write_job(active_job)

    # Submitting another should fail concurrency check
    with pytest.raises(RuntimeError, match="Maximum concurrent backtest jobs"):
        mgr.submit({}, None)


def test_contract_validator_dataframe() -> None:
    """Test DataFrame contract validation."""
    df_valid = pd.DataFrame(
        {
            "symbol": ["AAPL"],
            "asset_class": ["equity"],
            "exchange": ["NASDAQ"],
            "currency": ["USD"],
            "date": [pd.Timestamp("2026-01-01")],
            "open": [150.0],
            "high": [155.0],
            "low": [149.0],
            "close": [154.0],
            "source": ["test"],
        }
    )
    violations = validate_dataframe(df_valid, tier="raw")
    assert len(violations) == 0

    # Missing required column 'source'
    df_invalid = df_valid.drop(columns=["source"])
    violations2 = validate_dataframe(df_invalid, tier="raw")
    assert len(violations2) == 1
    assert violations2[0].violation_type == "MISSING_COLUMNS"


def test_contract_validator_file(tmp_path: Path) -> None:
    """Test Parquet file contract validation."""
    pfile = tmp_path / "test.parquet"
    df = pd.DataFrame({"symbol": ["AAPL"], "date": [pd.Timestamp("2026-01-01")]})
    df.to_parquet(pfile)

    # features tier only requires symbol and date
    v_clean = validate_parquet_file(pfile, tier="features")
    assert len(v_clean) == 0

    # raw tier requires many more columns
    v_missing = validate_parquet_file(pfile, tier="raw")
    assert len(v_missing) > 0


def test_api_system_routes_and_security_headers() -> None:
    """Test /health, /ready, /disclaimer and security headers on FastAPI app."""
    client = TestClient(app)

    # Health check
    resp = client.get("/health")
    assert resp.status_code == 200
    assert "status" in resp.json()
    assert "checks" in resp.json()
    assert resp.headers.get("x-content-type-options") == "nosniff"
    assert resp.headers.get("x-frame-options") == "DENY"

    # Readiness check
    r_ready = client.get("/ready")
    assert r_ready.status_code == 200
    assert r_ready.json().get("status") == "ready"

    # Disclaimer check
    r_disc = client.get("/disclaimer")
    assert r_disc.status_code == 200
    assert "SEBI" in r_disc.json()["disclaimer"]
