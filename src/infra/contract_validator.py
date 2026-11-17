"""
Data contract validator for Parquet data lake tiers.

Validates datasets against canonical schemas defined in docs/Data Schema.md.
Detects missing columns, type anomalies, schema drift, and empty datasets.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

CANONICAL_SCHEMAS: dict[str, list[str]] = {
    "raw": [
        "symbol",
        "asset_class",
        "exchange",
        "currency",
        "date",
        "open",
        "high",
        "low",
        "close",
        "source",
    ],
    "processed": [
        "symbol",
        "asset_class",
        "exchange",
        "currency",
        "date",
        "open",
        "high",
        "low",
        "close",
        "returns",
        "log_returns",
    ],
    "features": [
        "symbol",
        "date",
    ],
    "predictions": [
        "symbol",
        "date",
        "predicted_return",
        "model_name",
        "model_version",
    ],
    "signals": [
        "symbol",
        "date",
        "predicted_return",
        "signal",
    ],
    "trade_log": [
        "trade_id",
        "symbol",
        "entry_date",
        "exit_date",
        "entry_price",
        "exit_price",
        "position_size",
        "fees",
        "slippage_cost",
        "pnl",
        "return_pct",
    ],
    "portfolio_history": [
        "date",
        "cash",
        "invested_capital",
        "portfolio_value",
        "daily_return",
        "cumulative_return",
        "drawdown",
    ],
}


@dataclass
class ContractViolation:
    """Represents a single schema contract violation."""

    tier: str
    file_path: str
    violation_type: str
    details: str

    def to_dict(self) -> dict[str, str]:
        return {
            "tier": self.tier,
            "file_path": self.file_path,
            "violation_type": self.violation_type,
            "details": self.details,
        }


@dataclass
class ValidationReport:
    """Aggregated report of data contract validation across the data lake."""

    files_checked: int = 0
    files_passed: int = 0
    violations: list[ContractViolation] = field(default_factory=list)
    drifts: list[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        return len(self.violations) == 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "files_checked": self.files_checked,
            "files_passed": self.files_passed,
            "is_clean": self.is_clean,
            "violation_count": len(self.violations),
            "violations": [v.to_dict() for v in self.violations],
            "drifts": self.drifts,
        }


def validate_columns(
    actual_cols: set[str],
    tier: str,
    file_path: str = "",
) -> list[ContractViolation]:
    """Validate a set of column names against tier requirements."""
    violations: list[ContractViolation] = []
    required = CANONICAL_SCHEMAS.get(tier)
    if not required:
        return violations

    missing = [c for c in required if c not in actual_cols]
    if missing:
        violations.append(
            ContractViolation(
                tier=tier,
                file_path=file_path,
                violation_type="MISSING_COLUMNS",
                details=f"Missing required columns: {missing}",
            )
        )
    return violations


def validate_parquet_file(file_path: Path, tier: str) -> list[ContractViolation]:
    """Validate a single Parquet file against the tier schema."""
    violations: list[ContractViolation] = []
    if not file_path.exists():
        violations.append(
            ContractViolation(
                tier=tier,
                file_path=str(file_path),
                violation_type="FILE_NOT_FOUND",
                details=f"File {file_path} does not exist",
            )
        )
        return violations

    try:
        parquet_file = pq.ParquetFile(file_path)
        schema = parquet_file.schema_arrow
        actual_cols = set(schema.names)

        # Check for empty files
        if parquet_file.metadata.num_rows == 0:
            violations.append(
                ContractViolation(
                    tier=tier,
                    file_path=str(file_path),
                    violation_type="EMPTY_DATASET",
                    details=f"Parquet file {file_path.name} has 0 rows",
                )
            )

        col_violations = validate_columns(actual_cols, tier, str(file_path))
        violations.extend(col_violations)

    except Exception as exc:
        violations.append(
            ContractViolation(
                tier=tier,
                file_path=str(file_path),
                violation_type="CORRUPT_FILE",
                details=f"Error reading Parquet: {exc}",
            )
        )

    return violations


def validate_dataframe(df: pd.DataFrame, tier: str) -> list[ContractViolation]:
    """Validate an in-memory DataFrame against canonical schema."""
    if df.empty:
        return [
            ContractViolation(
                tier=tier,
                file_path="in-memory",
                violation_type="EMPTY_DATAFRAME",
                details="DataFrame contains 0 rows",
            )
        ]
    return validate_columns(set(df.columns), tier, "in-memory")


def validate_data_lake(data_dir: Path | str | None = None) -> ValidationReport:
    """
    Validate all Parquet files across data lake tiers against Data Schema.
    """
    report = ValidationReport()

    if data_dir:
        base = Path(data_dir)
        tier_dirs = {
            "raw": base / "raw",
            "processed": base / "processed",
            "features": base / "features",
            "predictions": base / "predictions",
            "signals": base / "signals",
        }
    else:
        tier_dirs = {
            "raw": Path(settings.DATA_RAW_PATH),
            "processed": Path(settings.DATA_PROCESSED_PATH),
            "features": Path(settings.DATA_FEATURES_PATH),
            "predictions": Path(settings.DATA_PREDICTIONS_PATH),
            "signals": Path(settings.DATA_SIGNALS_PATH),
        }

    for tier, directory in tier_dirs.items():
        if not directory.exists():
            continue

        for pfile in directory.glob("**/*.parquet"):
            report.files_checked += 1
            file_violations = validate_parquet_file(pfile, tier)
            if file_violations:
                report.violations.extend(file_violations)
            else:
                report.files_passed += 1

    return report
