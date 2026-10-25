"""Machine learning models, dataset building, evaluation, and training pipelines."""

from __future__ import annotations

from src.models.base_model import BaseModel
from src.models.dataset_builder import DatasetBuilder, DatasetSplits
from src.models.drift_detector import DriftDetector, DriftReport
from src.models.model_evaluator import ModelEvaluator
from src.models.model_factory import (
    available_model_types,
    create_model,
    register_model,
)
from src.models.model_store import ModelMetadata, ModelStore
from src.models.prediction_pipeline import PredictionPipeline
from src.models.prediction_service import PredictionService
from src.models.regime import RegimeClassifier, VolatilityRegimeClassifier
from src.models.retraining_pipeline import (
    RetrainDecision,
    RetrainingPipeline,
    RetrainingResult,
)
from src.models.training_pipeline import TrainingPipeline
from src.models.walk_forward import (
    WalkForwardFold,
    WalkForwardResult,
    WalkForwardSplitter,
    WalkForwardValidator,
)

__all__ = [
    "BaseModel",
    "DatasetBuilder",
    "DatasetSplits",
    "DriftDetector",
    "DriftReport",
    "ModelEvaluator",
    "ModelMetadata",
    "ModelStore",
    "PredictionPipeline",
    "PredictionService",
    "RegimeClassifier",
    "RetrainDecision",
    "RetrainingPipeline",
    "RetrainingResult",
    "TrainingPipeline",
    "VolatilityRegimeClassifier",
    "WalkForwardFold",
    "WalkForwardResult",
    "WalkForwardSplitter",
    "WalkForwardValidator",
    "available_model_types",
    "create_model",
    "register_model",
]
