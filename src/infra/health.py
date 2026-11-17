"""
System health checking module.

Evaluates database reachability, data pipeline freshness, and model artifact availability.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from sqlalchemy import text

from config.settings import settings
from src.database.sessions import get_session
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class HealthCheck:
    """Individual component health inspection result."""

    name: str
    status: str  # "ok" | "warn" | "fail"
    message: str
    details: dict[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class HealthReport:
    """Comprehensive system-wide health evaluation report."""

    checks: list[HealthCheck]
    overall: str  # "ok" | "degraded" | "down"
    generated_at: str

    def to_dict(self) -> dict[str, Any]:
        """Convert HealthReport to a dictionary."""
        return {
            "checks": [asdict(c) for c in self.checks],
            "overall": self.overall,
            "generated_at": self.generated_at,
        }


class HealthChecker:
    """Inspector verifying operational availability across data, database, and models."""

    def __init__(
        self,
        session_factory: Callable[..., Any] | None = None,
        now: datetime | None = None,
        max_data_age_days: int = 5,
        max_model_age_days: int | None = None,
    ) -> None:
        """Initialize HealthChecker with component thresholds."""
        self.session_factory = session_factory or get_session
        self.now_fn = (lambda: now) if now is not None else (lambda: datetime.now(UTC))
        self.max_data_age_days = max_data_age_days
        self.max_model_age_days = (
            max_model_age_days
            if max_model_age_days is not None
            else int(settings.MODEL_RETRAIN_INTERVAL_DAYS * 2)
        )

    def check_database(self) -> HealthCheck:
        """Verify PostgreSQL connectivity with a lightweight ping query."""
        try:
            with self.session_factory() as session:
                session.execute(text("SELECT 1"))
            return HealthCheck(
                name="database",
                status="ok",
                message="Database connection operational",
            )
        except Exception as exc:
            logger.warning("Database health check failed: %s", exc)
            return HealthCheck(
                name="database",
                status="fail",
                message="Database unreachable",
                details={"error_type": type(exc).__name__},
            )

    def check_data_freshness(self) -> HealthCheck:
        """Check freshness of features and signal output files."""
        candidates: list[Path] = []
        for path_str in [
            settings.DATA_FEATURES_PATH,
            settings.DATA_SIGNALS_PATH,
            settings.DATA_PROCESSED_PATH,
        ]:
            p = Path(path_str)
            if p.exists():
                candidates.extend(list(p.rglob("*.parquet")))

        if not candidates:
            return HealthCheck(
                name="data_freshness",
                status="fail",
                message="No data parquet files found in features or signals paths",
            )

        now = self.now_fn()
        newest_mtime = max(c.stat().st_mtime for c in candidates)
        newest_dt = datetime.fromtimestamp(newest_mtime, tz=UTC)
        age_days = (now - newest_dt).total_seconds() / 86400.0

        if age_days > self.max_data_age_days:
            return HealthCheck(
                name="data_freshness",
                status="warn",
                message=f"Data is {age_days:.1f} days old (exceeds {self.max_data_age_days} days)",
                details={
                    "age_days": round(age_days, 1),
                    "latest_file_date": newest_dt.isoformat(),
                },
            )

        return HealthCheck(
            name="data_freshness",
            status="ok",
            message=f"Data is fresh ({age_days:.1f} days old)",
            details={
                "age_days": round(age_days, 1),
                "latest_file_date": newest_dt.isoformat(),
            },
        )

    def check_models(self) -> HealthCheck:
        """Check trained model artifacts availability and age."""
        reg_path = Path(getattr(settings, "MODEL_ARTIFACT_PATH", Path("models")))
        if not reg_path.exists():
            return HealthCheck(
                name="models",
                status="warn",
                message="Model registry path does not exist",
            )

        model_files = list(reg_path.rglob("*.pkl")) + list(reg_path.rglob("*.joblib"))
        if not model_files:
            return HealthCheck(
                name="models",
                status="warn",
                message="No trained model artifacts found in registry",
            )

        now = self.now_fn()
        newest_mtime = max(m.stat().st_mtime for m in model_files)
        newest_dt = datetime.fromtimestamp(newest_mtime, tz=UTC)
        age_days = (now - newest_dt).total_seconds() / 86400.0

        if age_days > self.max_model_age_days:
            return HealthCheck(
                name="models",
                status="warn",
                message=f"Latest model is {age_days:.1f} days old (exceeds {self.max_model_age_days} days)",
                details={"age_days": round(age_days, 1)},
            )

        return HealthCheck(
            name="models",
            status="ok",
            message="Model artifacts operational",
            details={"n_models": len(model_files), "age_days": round(age_days, 1)},
        )

    def run_all(self) -> HealthReport:
        """Run all health checks and synthesize an overall operational status."""
        checks = [
            self.check_database(),
            self.check_data_freshness(),
            self.check_models(),
        ]

        has_fail = any(c.status == "fail" for c in checks)
        has_warn = any(c.status == "warn" for c in checks)

        if has_fail:
            overall = "down"
        elif has_warn:
            overall = "degraded"
        else:
            overall = "ok"

        return HealthReport(
            checks=checks,
            overall=overall,
            generated_at=self.now_fn().isoformat(),
        )
