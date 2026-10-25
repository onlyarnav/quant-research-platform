"""Orchestration pipeline for dataset building, model training, evaluation, and logging.

Coordinates the end-to-end training cycle:
- Chronological dataset slicing with embargo and lookback options
- Estimator fitting with optional early stopping on validation fold
- Dual evaluation (base regression metrics and extended financial metrics)
- MLflow experiment tracking (parameters, finite metrics, artifacts, tags)
- Optional disk persistence and model registry promotion via ModelStore
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import mlflow

from config.settings import settings
from src.models.base_model import BaseModel
from src.models.dataset_builder import DatasetBuilder, DatasetSplits
from src.models.model_evaluator import ModelEvaluator
from src.models.model_store import ModelStore
from src.models.training_report import (
    TrainingRunReport,
    build_model_metadata,
    finite_metrics,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class TrainingPipeline:
    """
    Orchestrates the entire machine learning workflow: dataset building,
    model training, evaluation, and experiment tracking via MLflow.
    """

    def __init__(
        self,
        dataset_builder: DatasetBuilder | None = None,
        evaluator: ModelEvaluator | None = None,
        model_store: ModelStore | None = None,
    ) -> None:
        """
        Initialize the training pipeline.

        Args:
            dataset_builder: Optional DatasetBuilder to handle data splitting.
            evaluator: Optional ModelEvaluator for computing metrics.
            model_store: Optional ModelStore instance for artifact persistence.
        """
        self.dataset_builder = dataset_builder
        self.evaluator = evaluator or ModelEvaluator()
        self.model_store = model_store
        self.last_run_report: TrainingRunReport | None = None

    def run(
        self,
        model: BaseModel,
        horizon: int | None = None,
        *,
        splits: DatasetSplits | None = None,
        embargo_dates: int = 0,
        lookback_dates: int | None = None,
        model_name: str | None = None,
        promote: bool = True,
    ) -> dict[str, float]:
        """
        Execute full training, evaluation, logging, and optional persistence.

        Args:
            model: BaseModel wrapper instance to train.
            horizon: Prediction horizon in days for target selection.
            splits: Optional pre-built DatasetSplits (bypasses dataset_builder).
            embargo_dates: Number of purge dates at split boundaries.
            lookback_dates: Optional rolling window in dates.
            model_name: Optional override name for run and storage.
            promote: Whether to promote candidate to LATEST if saved.

        Returns:
            Dictionary containing the six base evaluation metrics on test set.

        Raises:
            ValueError: If dataset_builder and splits are None, or horizon is None.
        """
        if self.dataset_builder is None and splits is None:
            raise ValueError("DatasetBuilder must be provided to the TrainingPipeline.")

        if horizon is None:
            raise ValueError("Horizon must be specified when running the pipeline.")

        effective_name = model_name or model.model_name
        logger.info(
            "Starting training pipeline for model=%s (horizon=%s)",
            effective_name,
            horizon,
        )

        # 1. Dataset splits
        if splits is None:
            assert self.dataset_builder is not None
            if embargo_dates == 0 and lookback_dates is None:
                data_splits = self.dataset_builder.build(horizon=horizon)
            else:
                data_splits = self.dataset_builder.build(
                    horizon=horizon,
                    embargo_dates=embargo_dates,
                    lookback_dates=lookback_dates,
                )
        else:
            data_splits = splits

        # 2. Fit model
        model.fit(
            X_train=data_splits.X_train,
            y_train=data_splits.y_train,
            X_val=data_splits.X_val,
            y_val=data_splits.y_val,
        )

        # 3. Predict exactly twice (val then test)
        val_preds = model.predict(data_splits.X_val)
        test_preds = model.predict(data_splits.X_test)

        # 4. Evaluate base metrics (six keys)
        val_base = self.evaluator.evaluate(data_splits.y_val, val_preds)
        test_base = self.evaluator.evaluate(data_splits.y_test, test_preds)

        # Extended metrics
        val_dates = (
            data_splits.meta_val["date"]
            if data_splits.meta_val is not None
            and "date" in data_splits.meta_val.columns
            else None
        )
        test_dates = (
            data_splits.meta_test["date"]
            if data_splits.meta_test is not None
            and "date" in data_splits.meta_test.columns
            else None
        )
        val_ext = self.evaluator.extended_metrics(
            data_splits.y_val, val_preds, dates=val_dates
        )
        test_ext = self.evaluator.extended_metrics(
            data_splits.y_test, test_preds, dates=test_dates
        )

        val_combined = {**val_base, **val_ext}
        test_combined = {**test_base, **test_ext}

        # 5. MLflow logging (failures must be caught and not crash training)
        mlflow_run_id: str | None = None
        try:
            mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
            mlflow.set_experiment(settings.MLFLOW_EXPERIMENT_NAME)

            with mlflow.start_run(run_name=effective_name) as run:
                if hasattr(run, "info") and hasattr(run.info, "run_id"):
                    mlflow_run_id = str(run.info.run_id)

                dataset_ver = (
                    data_splits.metadata.dataset_version
                    if data_splits.metadata is not None
                    else "unknown"
                )
                mlflow.set_tags(
                    {
                        "model_type": str(model.model_type),
                        "horizon": str(horizon),
                        "dataset_version": str(dataset_ver),
                        "feature_set_version": str(settings.FEATURE_SET_VERSION),
                        "n_features": str(len(data_splits.feature_columns)),
                    }
                )

                mlflow.log_params(model.hyperparameters)
                mlflow.log_metrics(finite_metrics(val_combined, "val_"))
                mlflow.log_metrics(finite_metrics(test_combined, "test_"))

                mlflow.log_dict(
                    {"feature_columns": list(data_splits.feature_columns)},
                    "feature_columns.json",
                )

                try:
                    importance = model.get_feature_importance(
                        list(data_splits.feature_columns)
                    )
                    if importance is not None:
                        mlflow.log_dict(importance, "feature_importance.json")
                except Exception:
                    pass

                self._log_model_artifact(model)
        except Exception as exc:
            logger.warning("MLflow tracking failed (training continues): %s", exc)

        # 6. Model persistence via ModelStore
        persisted_version: str | None = None
        artifact_path = None
        if self.model_store is not None:
            now_ts = datetime.now(timezone.utc)
            persisted_version = self.model_store.new_version(now_ts)
            model_dir = self.model_store.root / effective_name
            offset_sec = 1
            while (model_dir / persisted_version).exists():
                persisted_version = self.model_store.new_version(
                    now_ts + timedelta(seconds=offset_sec)
                )
                offset_sec += 1

            metadata = build_model_metadata(
                model=model,
                model_name=effective_name,
                horizon=horizon,
                splits=data_splits,
                val_metrics=val_combined,
                test_metrics=test_combined,
                mlflow_run_id=mlflow_run_id,
                version=persisted_version,
            )
            artifact_path = self.model_store.save(model, metadata, promote=promote)

        self.last_run_report = TrainingRunReport(
            model_name=effective_name,
            model_type=model.model_type,
            horizon=horizon,
            dataset_version=(
                data_splits.metadata.dataset_version
                if data_splits.metadata is not None
                else "unknown"
            ),
            feature_columns=tuple(data_splits.feature_columns),
            val_metrics=val_combined,
            test_metrics=test_combined,
            mlflow_run_id=mlflow_run_id,
            model_version=persisted_version,
            artifact_path=artifact_path,
        )

        logger.info(
            "Pipeline completed for %s. Test R2: %s",
            effective_name,
            test_base.get("r2", "N/A"),
        )
        return test_base

    def _log_model_artifact(self, model: BaseModel) -> None:
        """Log model artifact to MLflow according to model_type mapping."""
        m_type = model.model_type
        underlying = model.get_underlying_model()

        if m_type in (
            "linear",
            "ols",
            "lasso",
            "random_forest",
            "gradient_boosting",
        ):
            mlflow.sklearn.log_model(underlying, artifact_path="model")
        elif m_type == "xgboost":
            mlflow.xgboost.log_model(underlying, artifact_path="model")
        elif m_type == "lightgbm":
            mlflow.lightgbm.log_model(underlying, artifact_path="model")
        else:
            logger.warning(
                "Unsupported model type %s for MLflow logging. Skipping artifact.",
                m_type,
            )
