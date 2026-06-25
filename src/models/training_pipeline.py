"""
Orchestration pipeline for dataset building, model training, evaluation, and MLflow logging.
"""
from __future__ import annotations

import mlflow
import pandas as pd
import numpy as np
from src.models.base_model import BaseModel
from src.models.dataset_builder import DatasetBuilder
from src.models.model_evaluator import ModelEvaluator
from config.settings import settings
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
    ) -> None:
        """
        Initialize the training pipeline.

        Args:
            dataset_builder: Optional instance of DatasetBuilder to handle data splitting.
            evaluator: Optional instance of ModelEvaluator for computing metrics.
        """
        self.dataset_builder = dataset_builder
        self.evaluator = evaluator or ModelEvaluator()

    def run(self, model: BaseModel, horizon: int | None = None) -> dict[str, float]:
        """
        Execute the full training and evaluation pipeline for a given model.

        The pipeline follows these steps:
        1. Build dataset splits (Train, Val, Test) using the DatasetBuilder.
        2. Fit the model on the training set, using the validation set for early stopping if supported.
        3. Generate predictions on the validation and test sets.
        4. Evaluate predictions using the ModelEvaluator.
        5. Log hyperparameters, metrics, and the model artifact to MLflow.

        Args:
            model: The model wrapper instance to train.
            horizon: The prediction horizon (days) for target selection.

        Returns:
            A dictionary of metrics computed on the test set.

        Raises:
            ValueError: If dataset_builder is not provided.
        """
        if self.dataset_builder is None:
            raise ValueError("DatasetBuilder must be provided to the TrainingPipeline.")

        if horizon is None:
            raise ValueError("Horizon must be specified when running the pipeline.")

        logger.info("Starting training pipeline for model: %s (horizon=%s)", model.model_name, horizon)

        # 1. Build dataset splits
        splits = self.dataset_builder.build(horizon=horizon)

        # 2. Fit model
        model.fit(
            X_train=splits.X_train,
            y_train=splits.y_train,
            X_val=splits.X_val,
            y_val=splits.y_val,
        )

        # 3. Predict on val and test
        val_preds = model.predict(splits.X_val)
        test_preds = model.predict(splits.X_test)

        # 4. Evaluate
        val_metrics = self.evaluator.evaluate(splits.y_val, val_preds)
        test_metrics = self.evaluator.evaluate(splits.y_test, test_preds)

        # 5. MLflow Logging
        mlflow.set_tracking_uri(settings.MLFLOW_TRACKING_URI)
        with mlflow.start_run(run_name=model.model_name):
            # Log hyperparameters
            mlflow.log_params(model.hyperparameters)

            # Log validation metrics
            val_metrics_mlflow = {f"val_{k}": v for k, v in val_metrics.items()}
            mlflow.log_metrics(val_metrics_mlflow)

            # Log test metrics
            test_metrics_mlflow = {f"test_{k}": v for k, v in test_metrics.items()}
            mlflow.log_metrics(test_metrics_mlflow)

            # Log model artifact based on flavor
            self._log_model_artifact(model)

        logger.info("Pipeline completed for %s. Test R2: %s", model.model_name, test_metrics.get("r2", "N/A"))
        return test_metrics

    def _log_model_artifact(self, model: BaseModel) -> None:
        """
        Log the model artifact to MLflow using the appropriate flavor.
        """
        m_type = model.model_type
        underlying_model = model.get_underlying_model()

        if m_type == "linear":
            mlflow.sklearn.log_model(underlying_model, artifact_path="model")
        elif m_type == "xgboost":
            mlflow.xgboost.log_model(underlying_model, artifact_path="model")
        elif m_type == "lightgbm":
            mlflow.lightgbm.log_model(underlying_model, artifact_path="model")
        else:
            logger.warning("Unsupported model type %s for MLflow logging. Skipping artifact.", m_type)
