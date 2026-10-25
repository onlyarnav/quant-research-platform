"""
Training report data structures and metadata compilation helpers.

Provides TrainingRunReport for tracking pipeline outputs and helpers for
filtering finite metrics and constructing ModelMetadata for persistence and logging.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from config.settings import settings
from src.models.base_model import BaseModel
from src.models.dataset_builder import DatasetBuilder, DatasetSplits
from src.models.model_store import ModelMetadata
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class TrainingRunReport:
    """
    Summary report produced at the conclusion of a TrainingPipeline run.

    Attributes:
        model_name: Name of the trained model.
        model_type: Registry model type string.
        horizon: Prediction horizon in days.
        dataset_version: Cryptographic dataset hash.
        feature_columns: Tuple of feature column names.
        val_metrics: Validation metrics including extended metrics.
        test_metrics: Test metrics including extended metrics.
        mlflow_run_id: Tracking ID of the MLflow run.
        model_version: Persisted version string in ModelStore.
        artifact_path: Local filesystem directory path if saved to ModelStore.
    """

    model_name: str
    model_type: str
    horizon: int
    dataset_version: str
    feature_columns: tuple[str, ...]
    val_metrics: dict[str, float]
    test_metrics: dict[str, float]
    mlflow_run_id: str | None
    model_version: str | None
    artifact_path: Path | None


def finite_metrics(metrics: dict[str, float], prefix: str = "") -> dict[str, float]:
    """
    Filter metrics dictionary to contain only finite values with an optional key prefix.

    Logs a warning if any non-finite values (NaN, inf, -inf) are dropped.

    Args:
        metrics: Dictionary of metric name to float value.
        prefix: String prefix to prepend to metric keys (e.g. 'val_', 'test_').

    Returns:
        Dictionary with prefixed keys and strictly finite values.
    """
    filtered: dict[str, float] = {}
    dropped: list[str] = []

    for k, v in metrics.items():
        if v is not None and math.isfinite(v):
            filtered[f"{prefix}{k}"] = float(v)
        else:
            dropped.append(k)

    if dropped:
        logger.warning(
            "Dropping non-finite metrics for prefix '%s': %s",
            prefix,
            dropped,
        )

    return filtered


def build_model_metadata(
    model: BaseModel,
    model_name: str,
    horizon: int,
    splits: DatasetSplits,
    val_metrics: dict[str, float],
    test_metrics: dict[str, float],
    mlflow_run_id: str | None,
    version: str,
) -> ModelMetadata:
    """
    Compile a complete ModelMetadata instance from a fitted model and dataset splits.

    Args:
        model: Fitted BaseModel instance.
        model_name: Identifier name of the model.
        horizon: Prediction horizon in days.
        splits: DatasetSplits used for training and evaluation.
        val_metrics: Validation metrics dictionary.
        test_metrics: Test metrics dictionary.
        mlflow_run_id: Optional MLflow run ID string.
        version: Version timestamp string.

    Returns:
        Populated ModelMetadata ready for persistence.
    """
    now_iso = datetime.now(timezone.utc).isoformat()
    feature_cols = tuple(splits.feature_columns)

    if splits.metadata is not None:
        dataset_ver = splits.metadata.dataset_version
        feat_ver = splits.metadata.feature_set_version
        target_col = splits.metadata.target_column
        train_end = splits.metadata.split("train").end_date
        val_end = splits.metadata.split("val").end_date
        test_end = splits.metadata.split("test").end_date
    else:
        dataset_ver = "unknown"
        feat_ver = settings.FEATURE_SET_VERSION
        target_col = DatasetBuilder.TARGET_COLUMNS.get(
            horizon, f"future_return_{horizon}d"
        )
        train_end = None
        val_end = None
        test_end = None

    return ModelMetadata(
        model_name=model_name,
        model_type=model.model_type,
        version=version,
        created_at=now_iso,
        horizon=horizon,
        target_column=target_col,
        feature_columns=feature_cols,
        feature_set_version=feat_ver,
        dataset_version=dataset_ver,
        hyperparameters=dict(model.hyperparameters),
        val_metrics=dict(val_metrics),
        test_metrics=dict(test_metrics),
        train_end_date=train_end,
        val_end_date=val_end,
        test_end_date=test_end,
        mlflow_run_id=mlflow_run_id,
    )
