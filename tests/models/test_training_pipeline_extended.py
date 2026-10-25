"""
Extended tests for TrainingPipeline: tags, MLflow logging, ModelStore, and end-to-end workflow.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config.settings import settings
from src.models.base_model import BaseModel
from src.models.dataset_builder import DatasetBuilder, DatasetSplits
from src.models.lightgbm_model import LightGBMModel
from src.models.linear_model import LinearModel
from src.models.model_evaluator import ModelEvaluator
from src.models.model_factory import create_model
from src.models.model_store import ModelStore
from src.models.training_pipeline import TrainingPipeline


def _make_dummy_splits(n_rows: int = 20) -> DatasetSplits:
    """Helper to create minimal valid DatasetSplits."""
    X = pd.DataFrame({"f1": np.arange(n_rows, dtype=float), "f2": np.ones(n_rows)})
    y = pd.Series(np.arange(n_rows, dtype=float))
    meta = pd.DataFrame(
        {
            "symbol": ["AAPL"] * n_rows,
            "date": pd.date_range("2023-01-01", periods=n_rows),
        }
    )
    return DatasetSplits(
        X_train=X.iloc[:10],
        y_train=y.iloc[:10],
        X_val=X.iloc[10:15],
        y_val=y.iloc[10:15],
        X_test=X.iloc[15:],
        y_test=y.iloc[15:],
        feature_columns=["f1", "f2"],
        meta_train=meta.iloc[:10],
        meta_val=meta.iloc[10:15],
        meta_test=meta.iloc[15:],
    )


def test_splits_argument_bypasses_dataset_builder(mocker) -> None:
    """Assert passing splits= to run() bypasses dataset_builder.build()."""
    mock_builder = mocker.MagicMock(spec=DatasetBuilder)
    splits = _make_dummy_splits()
    model = LinearModel()

    pipeline = TrainingPipeline(dataset_builder=mock_builder)
    pipeline.run(model, horizon=1, splits=splits)

    mock_builder.build.assert_not_called()


def test_builder_build_called_with_exact_args(mocker) -> None:
    """Assert build is called with horizon=1 for defaults, and keywords when non-default."""
    mock_builder = mocker.MagicMock(spec=DatasetBuilder)
    mock_builder.build.return_value = _make_dummy_splits()
    model = LinearModel()
    pipeline = TrainingPipeline(dataset_builder=mock_builder)

    # Default call: exactly build(horizon=1)
    pipeline.run(model, horizon=1)
    mock_builder.build.assert_called_once_with(horizon=1)

    # Non-default call: with keywords
    mock_builder.reset_mock()
    pipeline.run(model, horizon=1, embargo_dates=2, lookback_dates=50)
    mock_builder.build.assert_called_once_with(
        horizon=1, embargo_dates=2, lookback_dates=50
    )


def test_mlflow_set_experiment_and_tags(mocker) -> None:
    """Assert set_experiment and set_tags are called with string tags."""
    mock_mlflow = mocker.patch("src.models.training_pipeline.mlflow")
    mock_run = mocker.MagicMock()
    mock_run.info.run_id = "test-run-id-99"
    mock_mlflow.start_run.return_value.__enter__.return_value = mock_run

    splits = _make_dummy_splits()
    model = LinearModel()
    pipeline = TrainingPipeline(evaluator=ModelEvaluator())
    pipeline.run(model, horizon=5, splits=splits)

    mock_mlflow.set_experiment.assert_called_once_with(settings.MLFLOW_EXPERIMENT_NAME)
    mock_mlflow.set_tags.assert_called_once()
    tags = mock_mlflow.set_tags.call_args[0][0]
    assert tags["model_type"] == "linear"
    assert tags["horizon"] == "5"
    assert "dataset_version" in tags
    assert "feature_set_version" in tags
    assert tags["n_features"] == "2"


def test_mlflow_drops_non_finite_metrics(mocker) -> None:
    """Assert non-finite metric values (e.g. NaN IC from constant preds) are dropped."""
    mock_mlflow = mocker.patch("src.models.training_pipeline.mlflow")
    mock_run = mocker.MagicMock()
    mock_mlflow.start_run.return_value.__enter__.return_value = mock_run

    class ConstantModel(BaseModel):
        def fit(self, X_train, y_train, X_val=None, y_val=None):
            self._is_fitted = True

        def predict(self, X):
            return np.ones(len(X))  # Constant -> IC is NaN

        @property
        def model_type(self):
            return "linear"

    splits = _make_dummy_splits()
    pipeline = TrainingPipeline()
    pipeline.run(ConstantModel(model_name="c_mod"), horizon=1, splits=splits)

    # Assert log_metrics called twice
    assert mock_mlflow.log_metrics.call_count == 2
    val_metrics = mock_mlflow.log_metrics.call_args_list[0][0][0]
    assert "val_ic" not in val_metrics
    assert "val_mse" in val_metrics
    assert "val_rank_ic" not in val_metrics  # Also NaN


def test_mlflow_log_dict_feature_columns_and_importance(mocker) -> None:
    """Assert feature_columns.json and feature_importance.json are logged."""
    mock_mlflow = mocker.patch("src.models.training_pipeline.mlflow")
    mock_run = mocker.MagicMock()
    mock_mlflow.start_run.return_value.__enter__.return_value = mock_run

    splits = _make_dummy_splits()
    model = LinearModel()
    pipeline = TrainingPipeline()
    pipeline.run(model, horizon=1, splits=splits)

    logged_dicts = {
        call[0][1]: call[0][0] for call in mock_mlflow.log_dict.call_args_list
    }
    assert "feature_columns.json" in logged_dicts
    assert logged_dicts["feature_columns.json"] == {"feature_columns": ["f1", "f2"]}
    assert "feature_importance.json" in logged_dicts


@pytest.mark.parametrize(
    "m_type", ["ols", "lasso", "random_forest", "gradient_boosting"]
)
def test_log_model_artifact_dispatches_sklearn(mocker, m_type: str) -> None:
    """Assert new models dispatch to mlflow.sklearn.log_model."""
    mock_mlflow = mocker.patch("src.models.training_pipeline.mlflow")
    mock_run = mocker.MagicMock()
    mock_mlflow.start_run.return_value.__enter__.return_value = mock_run

    kwargs = {}
    if m_type in ("random_forest", "gradient_boosting"):
        kwargs = {"n_estimators": 2}
    model = create_model(m_type, **kwargs)
    splits = _make_dummy_splits()

    pipeline = TrainingPipeline()
    pipeline.run(model, horizon=1, splits=splits)

    mock_mlflow.sklearn.log_model.assert_called_once()


def test_training_pipeline_with_model_store_saves_and_promotes(
    tmp_path, mocker
) -> None:
    """Assert TrainingPipeline with ModelStore saves artifact and sets last_run_report."""
    mocker.patch("src.models.training_pipeline.mlflow")
    store = ModelStore(root=tmp_path / "models")
    pipeline = TrainingPipeline(model_store=store)
    splits = _make_dummy_splits()
    model = LinearModel()

    res = pipeline.run(
        model, horizon=1, splits=splits, model_name="my_model", promote=True
    )

    # Returned dict has exactly six base keys
    assert set(res.keys()) == {"mse", "rmse", "mae", "r2", "directional_accuracy", "ic"}

    # Last run report is set
    report = pipeline.last_run_report
    assert report is not None
    assert report.model_name == "my_model"
    assert report.model_version is not None
    assert report.artifact_path is not None
    assert report.artifact_path.exists()
    assert store.latest_version("my_model") == report.model_version


def test_training_pipeline_promote_false(tmp_path, mocker) -> None:
    """Assert promote=False saves the model version without creating LATEST."""
    mocker.patch("src.models.training_pipeline.mlflow")
    store = ModelStore(root=tmp_path / "models")
    pipeline = TrainingPipeline(model_store=store)
    splits = _make_dummy_splits()
    model = LinearModel()

    pipeline.run(model, horizon=1, splits=splits, model_name="no_promo", promote=False)
    assert store.latest_version("no_promo") is None
    assert len(store.list_versions("no_promo")) == 1


def test_training_pipeline_end_to_end(
    feature_frame, feature_dir_factory, tmp_path, mocker
) -> None:
    """End-to-end integration test with real DatasetBuilder and real LightGBMModel."""
    mocker.patch("src.models.training_pipeline.mlflow")
    feat_dir = feature_dir_factory(feature_frame)
    builder = DatasetBuilder(features_path=feat_dir)
    store = ModelStore(root=tmp_path / "models")
    pipeline = TrainingPipeline(dataset_builder=builder, model_store=store)

    lgb_model = LightGBMModel(n_estimators=5, verbose=-1)
    metrics = pipeline.run(lgb_model, horizon=1, model_name="lgb_test")

    assert "ic" in metrics
    assert "r2" in metrics
    assert pipeline.last_run_report is not None
    assert store.latest_version("lgb_test") is not None
