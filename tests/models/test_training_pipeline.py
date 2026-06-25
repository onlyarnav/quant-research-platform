"""
Tests for the TrainingPipeline orchestration.

Exercises the full workflow: dataset building, model fitting, prediction,
evaluation, and MLflow logging. All MLflow calls are mocked to avoid
network dependencies.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from src.models.base_model import BaseModel
from src.models.dataset_builder import DatasetBuilder, DatasetSplits
from src.models.model_evaluator import ModelEvaluator
from src.models.training_pipeline import TrainingPipeline
from config.settings import settings

class DummyTrainingModel(BaseModel):
    """Concrete model for testing the training pipeline without real ML overhead."""
    def __init__(self, model_type_value="linear"):
        super().__init__(model_name="dummy", param1=1)
        self._model_type_value = model_type_value

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, X_val=None, y_val=None) -> None:
        self._model = "fitted_dummy"
        self._is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_is_fitted()
        return np.zeros(len(X))

    @property
    def model_type(self) -> str:
        return self._model_type_value

def test_run_raises_if_dataset_builder_missing():
    """Assert that run() raises ValueError if no DatasetBuilder is provided."""
    pipeline = TrainingPipeline(dataset_builder=None)
    model = DummyTrainingModel()
    with pytest.raises(ValueError, match="DatasetBuilder must be provided"):
        pipeline.run(model, horizon=1)

def test_run_raises_if_horizon_missing(mocker):
    """Assert that run() raises ValueError if horizon is not specified."""
    mock_builder = mocker.MagicMock(spec=DatasetBuilder)
    pipeline = TrainingPipeline(dataset_builder=mock_builder)
    model = DummyTrainingModel()
    # We bypass the default horizon in settings by passing None explicitly if allowed,
    # but according to the spec: 'if horizon is None: raise ValueError'
    with pytest.raises(ValueError, match="Horizon must be specified"):
        pipeline.run(model, horizon=None)

def test_run_calls_dataset_builder_build_with_horizon(mocker, regression_data):
    """Assert that TrainingPipeline requests the correct horizon from DatasetBuilder."""
    X_train, y_train, X_val, y_val, X_test, y_test = regression_data
    splits = DatasetSplits(X_train, y_train, X_val, y_val, X_test, y_test, ["f1"])

    mock_builder = mocker.MagicMock(spec=DatasetBuilder)
    mock_builder.build.return_value = splits

    mocker.patch("src.models.training_pipeline.mlflow")

    pipeline = TrainingPipeline(dataset_builder=mock_builder)
    model = DummyTrainingModel()
    pipeline.run(model, horizon=1)

    mock_builder.build.assert_called_once_with(horizon=1)

def test_run_fits_model_with_correct_splits(mocker, regression_data):
    """Assert that the model is fitted using the exact splits provided by the builder."""
    X_train, y_train, X_val, y_val, X_test, y_test = regression_data
    splits = DatasetSplits(X_train, y_train, X_val, y_val, X_test, y_test, ["f1"])

    mock_builder = mocker.MagicMock(spec=DatasetBuilder)
    mock_builder.build.return_value = splits

    mocker.patch("src.models.training_pipeline.mlflow")

    pipeline = TrainingPipeline(dataset_builder=mock_builder)
    model = DummyTrainingModel()

    # Use a spy to check arguments
    spy_fit = mocker.spy(model, "fit")
    pipeline.run(model, horizon=1)

    spy_fit.assert_called_once_with(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
    )
    assert model._is_fitted is True

def test_run_calls_predict_on_val_and_test(mocker, regression_data):
    """Assert that the pipeline generates predictions for both validation and test sets."""
    X_train, y_train, X_val, y_val, X_test, y_test = regression_data
    splits = DatasetSplits(X_train, y_train, X_val, y_val, X_test, y_test, ["f1"])

    mock_builder = mocker.MagicMock(spec=DatasetBuilder)
    mock_builder.build.return_value = splits

    mocker.patch("src.models.training_pipeline.mlflow")

    pipeline = TrainingPipeline(dataset_builder=mock_builder)
    model = DummyTrainingModel()

    spy_predict = mocker.spy(model, "predict")
    pipeline.run(model, horizon=1)

    assert spy_predict.call_count == 2

def test_run_returns_test_metrics_dict(mocker, regression_data):
    """Assert that run() returns a dictionary with the 6 standard metrics."""
    X_train, y_train, X_val, y_val, X_test, y_test = regression_data
    splits = DatasetSplits(X_train, y_train, X_val, y_val, X_test, y_test, ["f1"])

    mock_builder = mocker.MagicMock(spec=DatasetBuilder)
    mock_builder.build.return_value = splits

    mocker.patch("src.models.training_pipeline.mlflow")

    pipeline = TrainingPipeline(dataset_builder=mock_builder)
    model = DummyTrainingModel()

    metrics = pipeline.run(model, horizon=1)

    expected_keys = {"mse", "rmse", "mae", "r2", "directional_accuracy", "ic"}
    assert set(metrics.keys()) == expected_keys

def test_run_sets_mlflow_tracking_uri(mocker, regression_data):
    """Assert that MLflow tracking URI is configured from settings."""
    X_train, y_train, X_val, y_val, X_test, y_test = regression_data
    splits = DatasetSplits(X_train, y_train, X_val, y_val, X_test, y_test, ["f1"])

    mock_builder = mocker.MagicMock(spec=DatasetBuilder)
    mock_builder.build.return_value = splits

    mock_mlflow = mocker.patch("src.models.training_pipeline.mlflow")

    pipeline = TrainingPipeline(dataset_builder=mock_builder)
    model = DummyTrainingModel()
    pipeline.run(model, horizon=1)

    mock_mlflow.set_tracking_uri.assert_called_once_with(settings.MLFLOW_TRACKING_URI)

def test_run_logs_params_to_mlflow(mocker, regression_data):
    """Assert that model hyperparameters are logged to MLflow."""
    X_train, y_train, X_val, y_val, X_test, y_test = regression_data
    splits = DatasetSplits(X_train, y_train, X_val, y_val, X_test, y_test, ["f1"])

    mock_builder = mocker.MagicMock(spec=DatasetBuilder)
    mock_builder.build.return_value = splits

    mock_mlflow = mocker.patch("src.models.training_pipeline.mlflow")

    pipeline = TrainingPipeline(dataset_builder=mock_builder)
    model = DummyTrainingModel()
    pipeline.run(model, horizon=1)

    mock_mlflow.log_params.assert_called_once_with(model.hyperparameters)

def test_run_logs_val_and_test_metrics_with_prefixes(mocker, regression_data):
    """Assert that validation and test metrics are logged with correct prefixes."""
    X_train, y_train, X_val, y_val, X_test, y_test = regression_data
    splits = DatasetSplits(X_train, y_train, X_val, y_val, X_test, y_test, ["f1"])

    mock_builder = mocker.MagicMock(spec=DatasetBuilder)
    mock_builder.build.return_value = splits

    mock_mlflow = mocker.patch("src.models.training_pipeline.mlflow")

    pipeline = TrainingPipeline(dataset_builder=mock_builder)
    model = DummyTrainingModel()
    pipeline.run(model, horizon=1)

    # Check calls to log_metrics
    calls = mock_mlflow.log_metrics.call_args_list
    assert len(calls) == 2

    val_metrics = calls[0][0][0] # first arg of first call
    test_metrics = calls[1][0][0] # first arg of second call

    assert all(k.startswith("val_") for k in val_metrics.keys())
    assert all(k.startswith("test_") for k in test_metrics.keys())

def test_log_model_artifact_dispatches_linear(mocker, regression_data):
    """Assert that 'linear' models are logged using the sklearn flavor."""
    X_train, y_train, X_val, y_val, X_test, y_test = regression_data
    splits = DatasetSplits(X_train, y_train, X_val, y_val, X_test, y_test, ["f1"])

    mock_builder = mocker.MagicMock(spec=DatasetBuilder)
    mock_builder.build.return_value = splits

    mock_mlflow = mocker.patch("src.models.training_pipeline.mlflow")

    pipeline = TrainingPipeline(dataset_builder=mock_builder)
    model = DummyTrainingModel(model_type_value="linear")
    pipeline.run(model, horizon=1)

    mock_mlflow.sklearn.log_model.assert_called_once()

def test_log_model_artifact_dispatches_xgboost(mocker, regression_data):
    """Assert that 'xgboost' models are logged using the xgboost flavor."""
    X_train, y_train, X_val, y_val, X_test, y_test = regression_data
    splits = DatasetSplits(X_train, y_train, X_val, y_val, X_test, y_test, ["f1"])

    mock_builder = mocker.MagicMock(spec=DatasetBuilder)
    mock_builder.build.return_value = splits

    mock_mlflow = mocker.patch("src.models.training_pipeline.mlflow")

    pipeline = TrainingPipeline(dataset_builder=mock_builder)
    model = DummyTrainingModel(model_type_value="xgboost")
    pipeline.run(model, horizon=1)

    mock_mlflow.xgboost.log_model.assert_called_once()

def test_log_model_artifact_dispatches_lightgbm(mocker, regression_data):
    """Assert that 'lightgbm' models are logged using the lightgbm flavor."""
    X_train, y_train, X_val, y_val, X_test, y_test = regression_data
    splits = DatasetSplits(X_train, y_train, X_val, y_val, X_test, y_test, ["f1"])

    mock_builder = mocker.MagicMock(spec=DatasetBuilder)
    mock_builder.build.return_value = splits

    mock_mlflow = mocker.patch("src.models.training_pipeline.mlflow")

    pipeline = TrainingPipeline(dataset_builder=mock_builder)
    model = DummyTrainingModel(model_type_value="lightgbm")
    pipeline.run(model, horizon=1)

    mock_mlflow.lightgbm.log_model.assert_called_once()

def test_log_model_artifact_unknown_type_logs_warning_no_raise(mocker, regression_data, caplog):
    """Assert that unknown model types log a warning and do not crash the pipeline."""
    X_train, y_train, X_val, y_val, X_test, y_test = regression_data
    splits = DatasetSplits(X_train, y_train, X_val, y_val, X_test, y_test, ["f1"])

    mock_builder = mocker.MagicMock(spec=DatasetBuilder)
    mock_builder.build.return_value = splits

    mock_mlflow = mocker.patch("src.models.training_pipeline.mlflow")

    pipeline = TrainingPipeline(dataset_builder=mock_builder)
    model = DummyTrainingModel(model_type_value="unknown_type")

    # Should not raise
    pipeline.run(model, horizon=1)

    assert any("Unsupported model type unknown_type" in rec.message for rec in caplog.records)
