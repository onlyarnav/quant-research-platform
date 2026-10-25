"""
Model persistence and version registry on local disk.

Stores trained models, serialized metadata, and atomic LATEST pointer files under
<MODEL_ARTIFACT_PATH>/<model_name>/<version>/.
Supports model promotion, version listing, and deterministic deserialization.
"""

from __future__ import annotations

import json
import math
import os
import re
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.settings import settings
from src.models.base_model import BaseModel
from src.models.model_factory import create_model
from src.utils.logger import get_logger

logger = get_logger(__name__)

_VALID_NAME_PATTERN = re.compile(r"^[A-Za-z0-9_.-]+$")


def _sanitize_metrics(metrics: dict[str, float | None]) -> dict[str, float | None]:
    """Replace NaN and inf values with None for JSON compliance."""
    sanitized: dict[str, float | None] = {}
    for k, v in metrics.items():
        if v is None or math.isnan(v) or math.isinf(v):
            sanitized[k] = None
        else:
            sanitized[k] = float(v)
    return sanitized


@dataclass(slots=True)
class ModelMetadata:
    """Metadata recording model configuration, lineage, split dates, and metrics."""

    model_name: str
    model_type: str
    version: str
    created_at: str
    horizon: int
    target_column: str
    feature_columns: tuple[str, ...]
    feature_set_version: str
    dataset_version: str
    hyperparameters: dict[str, Any]
    val_metrics: dict[str, float | None]
    test_metrics: dict[str, float | None]
    train_end_date: str | None = None
    val_end_date: str | None = None
    test_end_date: str | None = None
    mlflow_run_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Convert metadata to JSON-serializable dictionary with sanitized floats."""
        d = asdict(self)
        d["feature_columns"] = list(self.feature_columns)
        d["val_metrics"] = _sanitize_metrics(self.val_metrics)
        d["test_metrics"] = _sanitize_metrics(self.test_metrics)
        return d

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelMetadata:
        """Construct ModelMetadata from a dictionary."""
        d = dict(data)
        d["feature_columns"] = tuple(d.get("feature_columns", ()))
        return cls(**d)


class ModelStore:
    """
    Filesystem-backed registry for saving, versioning, and loading trained ML models.
    """

    def __init__(self, root: Path | None = None) -> None:
        """
        Initialize the ModelStore.

        Args:
            root: Root directory for models (defaults to settings.MODEL_ARTIFACT_PATH).
        """
        self.root = root or Path(settings.MODEL_ARTIFACT_PATH)

    @staticmethod
    def new_version(now: datetime | None = None) -> str:
        """
        Generate a chronological UTC version string.

        Args:
            now: Optional datetime; defaults to datetime.now(timezone.utc).

        Returns:
            Timestamp string formatted as 'YYYYMMDDTHHMMSSZ'.
        """
        ts = now or datetime.now(timezone.utc)
        if ts.tzinfo is None:
            ts = ts.replace(tzinfo=timezone.utc)
        else:
            ts = ts.astimezone(timezone.utc)
        return ts.strftime("%Y%m%dT%H%M%SZ")

    def _validate_name(self, model_name: str) -> None:
        """Validate model_name against directory-traversal safe characters."""
        if not model_name or not _VALID_NAME_PATTERN.match(model_name):
            raise ValueError(f"Invalid model_name '{model_name}'.")

    def save(
        self,
        model: BaseModel,
        metadata: ModelMetadata,
        *,
        promote: bool = True,
    ) -> Path:
        """
        Persist a fitted model and metadata under <root>/<model_name>/<version>/.

        Args:
            model: Fitted BaseModel instance.
            metadata: Associated ModelMetadata.
            promote: Whether to update the LATEST pointer atomically.

        Returns:
            Path to the saved version directory.

        Raises:
            ValueError: If model_name is invalid.
            FileExistsError: If version directory already exists.
            RuntimeError: If model is not fitted.
        """
        self._validate_name(metadata.model_name)
        model_dir = self.root / metadata.model_name
        version_dir = model_dir / metadata.version

        if version_dir.exists():
            raise FileExistsError(f"Version directory '{version_dir}' already exists.")

        version_dir.mkdir(parents=True, exist_ok=False)

        model_path = version_dir / "model.joblib"
        metadata_path = version_dir / "metadata.json"

        # model.save raises RuntimeError("Cannot save unfitted model...") if unfitted
        model.save(model_path)

        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(metadata.to_dict(), f, indent=2)

        if promote:
            self.promote(metadata.model_name, metadata.version)

        logger.info(
            "Saved model=%s version=%s promoted=%s",
            metadata.model_name,
            metadata.version,
            promote,
        )
        return version_dir

    def load(
        self,
        model_name: str,
        version: str | None = None,
    ) -> tuple[BaseModel, ModelMetadata]:
        """
        Load a model and metadata from disk.

        Args:
            model_name: Model identifier name.
            version: Specific version string. If None, loads LATEST version.

        Returns:
            Tuple of (reconstructed BaseModel, ModelMetadata).

        Raises:
            FileNotFoundError: If model or requested version is not found.
        """
        self._validate_name(model_name)
        model_dir = self.root / model_name

        if version is None:
            version = self.latest_version(model_name)
            if version is None:
                raise FileNotFoundError(
                    f"No trained model found for model_name='{model_name}'"
                )

        version_dir = model_dir / version
        model_path = version_dir / "model.joblib"
        metadata_path = version_dir / "metadata.json"

        if not version_dir.exists() or not model_path.exists():
            raise FileNotFoundError(
                f"No trained model found for model_name='{model_name}' "
                f"version='{version}'"
            )

        with open(metadata_path, "r", encoding="utf-8") as f:
            meta_dict = json.load(f)

        metadata = ModelMetadata.from_dict(meta_dict)
        model = create_model(metadata.model_type, **metadata.hyperparameters)
        model.load(model_path)
        model.model_name = metadata.model_name

        return model, metadata

    def latest_version(self, model_name: str) -> str | None:
        """
        Read the latest promoted version for a model.

        Args:
            model_name: Model identifier name.

        Returns:
            Promoted version string, or None if no LATEST file exists.
        """
        self._validate_name(model_name)
        latest_file = self.root / model_name / "LATEST"
        if not latest_file.exists():
            return None
        return latest_file.read_text(encoding="utf-8").strip()

    def list_versions(self, model_name: str) -> list[str]:
        """
        List all saved versions for a model in chronological ascending order.

        Args:
            model_name: Model identifier name.

        Returns:
            Sorted list of version strings.
        """
        self._validate_name(model_name)
        model_dir = self.root / model_name
        if not model_dir.exists():
            return []
        versions = [
            p.name
            for p in model_dir.iterdir()
            if p.is_dir() and (p / "model.joblib").exists()
        ]
        return sorted(versions)

    def promote(self, model_name: str, version: str) -> None:
        """
        Atomically promote a version to be the LATEST version.

        Args:
            model_name: Model identifier name.
            version: Existing version string to promote.

        Raises:
            FileNotFoundError: If the specified version directory does not exist.
        """
        self._validate_name(model_name)
        model_dir = self.root / model_name
        version_dir = model_dir / version

        if not version_dir.exists():
            raise FileNotFoundError(
                f"Cannot promote nonexistent version='{version}' "
                f"for model='{model_name}'"
            )

        model_dir.mkdir(parents=True, exist_ok=True)
        latest_file = model_dir / "LATEST"
        temp_file = model_dir / f"LATEST.tmp.{os.getpid()}"

        temp_file.write_text(version, encoding="utf-8")
        os.replace(temp_file, latest_file)
        logger.info("Promoted model=%s to version=%s", model_name, version)

    def exists(self, model_name: str) -> bool:
        """
        Check if any version or LATEST exists for the model.

        Args:
            model_name: Model identifier name.

        Returns:
            True if model has a latest version or any valid version directory.
        """
        self._validate_name(model_name)
        return (
            self.latest_version(model_name) is not None
            or len(self.list_versions(model_name)) > 0
        )
