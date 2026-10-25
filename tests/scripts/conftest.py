"""Pytest fixtures for scripts testing."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pandas as pd
import pytest

from config.settings import settings
from src.data_storage.parquet_writer import LAYER_PATH_MAP, DataLayer
from src.models.dataset_builder import DatasetBuilder
from src.models.linear_model import LinearModel
from src.models.model_store import ModelStore
from src.models.prediction_pipeline import PredictionPipeline
from src.models.prediction_service import PredictionService
from src.models.training_pipeline import TrainingPipeline


@pytest.fixture(autouse=True)
def mock_mlflow_globally(monkeypatch):
    """Automatically mock mlflow to prevent tests from hanging on network connections."""
    fake_mlflow = MagicMock()
    fake_mlflow.start_run.return_value.__enter__.return_value = fake_mlflow
    monkeypatch.setattr("src.models.training_pipeline.mlflow", fake_mlflow)
    return fake_mlflow


@pytest.fixture
def feature_frame() -> pd.DataFrame:
    """Fixture providing a 4-symbol x 150-day synthetic feature dataframe."""
    rng = np.random.default_rng(42)
    symbols = ["AAPL", "MSFT", "INFY", "TCS"]
    n_days = 150
    dates = pd.date_range("2023-01-01", periods=n_days, freq="B")
    frames = []

    for sym in symbols:
        f1 = rng.standard_normal(n_days)
        f2 = rng.standard_normal(n_days)
        f3 = rng.standard_normal(n_days)
        f4 = rng.standard_normal(n_days)

        ret_1d = 0.5 * f1 + rng.normal(0, 0.2, n_days)
        ret_5d = 0.5 * f1 + rng.normal(0, 0.3, n_days)
        ret_10d = 0.5 * f1 + rng.normal(0, 0.4, n_days)

        ret_1d[-1] = np.nan
        ret_5d[-5:] = np.nan
        ret_10d[-10:] = np.nan

        df = pd.DataFrame(
            {
                "symbol": sym,
                "date": dates,
                "f1": f1,
                "f2": f2,
                "f3": f3,
                "f4": f4,
                "future_return_1d": ret_1d,
                "future_return_5d": ret_5d,
                "future_return_10d": ret_10d,
            }
        )
        frames.append(df)

    return pd.concat(frames, ignore_index=True)


@pytest.fixture
def feature_dir_factory(tmp_path):
    """Factory fixture to write parquet features to tmp directory."""
    counter = 0

    def _create(df: pd.DataFrame):
        nonlocal counter
        counter += 1
        feat_dir = tmp_path / f"features_dir_{counter}"
        feat_dir.mkdir(parents=True, exist_ok=True)
        for sym, group in df.groupby("symbol"):
            group.to_parquet(feat_dir / f"{sym}.parquet", index=False)
        return feat_dir

    return _create


@pytest.fixture
def trained_model_setup(tmp_path, feature_frame, feature_dir_factory, monkeypatch):
    """Setup a real store with one trained model and patched predictions folder."""
    pred_dir = tmp_path / "data" / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(settings, "DATA_PREDICTIONS_PATH", pred_dir)
    monkeypatch.setitem(LAYER_PATH_MAP, DataLayer.PREDICTIONS, pred_dir)

    models_dir = tmp_path / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.setattr(settings, "MODEL_ARTIFACT_PATH", models_dir)

    feat_dir = feature_dir_factory(feature_frame)
    monkeypatch.setattr(settings, "DATA_FEATURES_PATH", feat_dir)
    monkeypatch.setitem(LAYER_PATH_MAP, DataLayer.FEATURES, feat_dir)

    builder = DatasetBuilder(features_path=feat_dir)
    store = ModelStore(root=models_dir)
    pipeline = TrainingPipeline(dataset_builder=builder, model_store=store)

    model = LinearModel(alpha=1.0)
    pipeline.run(model, horizon=1, model_name="test_linear", promote=True)

    service = PredictionService(
        model_store=store,
        prediction_pipeline=PredictionPipeline(),
        dataset_builder=builder,
    )
    return service, store, builder, pred_dir
