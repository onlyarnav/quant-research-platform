"""
Pytest fixtures for ML dataset builder testing.

Provides a temporary feature directory containing synthetic parquet
files used to exercise DatasetBuilder without touching real data.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def tmp_features_dir(tmp_path):
    """
    Build a temporary feature directory with synthetic parquet files.

    Creates 2 symbols, each with 20 unique dates, including:
    - NaN feature rows (warm-up simulation, rows 0-2)
    - NaN target rows (end-of-series simulation, rows 17-19), applied
      partially across all three target columns so no horizon ends up
      with a fully-NaN target column.
    """
    features_dir = tmp_path / "features"
    features_dir.mkdir()

    rng = np.random.default_rng(42)
    symbols = ["AAPL", "MSFT"]
    dates = pd.date_range(start="2023-01-01", periods=20, freq="D")

    for symbol in symbols:
        df = pd.DataFrame(
            {
                "symbol": [symbol] * len(dates),
                "date": dates,
                "sma_5": rng.standard_normal(len(dates)),
                "rsi_14": rng.uniform(0, 100, len(dates)),
                "future_return_1d": rng.standard_normal(len(dates)),
                "future_return_5d": rng.standard_normal(len(dates)),
                "future_return_10d": rng.standard_normal(len(dates)),
            }
        )

        # Warm-up rows (feature NaNs)
        df.loc[0:2, ["sma_5", "rsi_14"]] = np.nan

        # End-of-series rows (target NaNs) — partial, never wipe a full column
        df.loc[17:19, ["future_return_1d", "future_return_5d", "future_return_10d"]] = (
            np.nan
        )

        file_path = features_dir / f"{symbol}.parquet"
        df.to_parquet(file_path, index=False)

    return features_dir


@pytest.fixture
def regression_data():
    """
    Shared synthetic train/val/test data for ML model wrapper tests.
    """
    rng = np.random.default_rng(7)
    X_train = pd.DataFrame(
        rng.standard_normal((50, 4)), columns=["f1", "f2", "f3", "f4"]
    )
    y_train = pd.Series(rng.standard_normal(50))
    X_val = pd.DataFrame(rng.standard_normal((10, 4)), columns=["f1", "f2", "f3", "f4"])
    y_val = pd.Series(rng.standard_normal(10))
    X_test = pd.DataFrame(
        rng.standard_normal((10, 4)), columns=["f1", "f2", "f3", "f4"]
    )
    y_test = pd.Series(rng.standard_normal(10))
    return X_train, y_train, X_val, y_val, X_test, y_test


@pytest.fixture(autouse=True)
def mock_mlflow_globally(monkeypatch):
    """
    Autouse fixture that mocks mlflow in training_pipeline so tests never make network calls.
    Tests that assert on MLflow calls may still patch src.models.training_pipeline.mlflow themselves.
    """
    from unittest.mock import MagicMock

    mock_mlflow = MagicMock()
    # Configure mock run context manager
    mock_run = MagicMock()
    mock_run.info.run_id = "mock-run-id-12345"
    mock_mlflow.start_run.return_value.__enter__.return_value = mock_run
    mock_mlflow.start_run.return_value.__exit__.return_value = None
    monkeypatch.setattr("src.models.training_pipeline.mlflow", mock_mlflow)
    return mock_mlflow


@pytest.fixture
def fitted_dummy_model():
    """A pre-fitted dummy model for prediction pipeline tests."""
    from src.models.base_model import BaseModel

    class DummyPredictModel(BaseModel):
        def fit(self, X_train, y_train, X_val=None, y_val=None) -> None:
            self._model = "fitted"
            self._is_fitted = True

        def predict(self, X):
            self._check_is_fitted()
            # Return predictable, deterministic values for assertions
            return np.arange(len(X), dtype=float) * 0.01

        @property
        def model_type(self) -> str:
            return "dummy"

    model = DummyPredictModel(model_name="dummy_predict_model")
    model.fit(pd.DataFrame({"f1": [1, 2, 3]}), pd.Series([0.1, 0.2, 0.3]))
    return model


@pytest.fixture
def feature_frame() -> pd.DataFrame:
    """
    Shared DataFrame of 4 symbols x 150 consecutive business days.
    Contains f1..f4 features, with future_return_1d = 0.5*f1 + noise so models learn.
    """
    rng = np.random.default_rng(42)
    symbols = ["INFY", "RELIANCE", "TCS", "HDFCBANK"]
    n_days = 150
    dates = pd.date_range("2023-01-01", periods=n_days, freq="B")
    frames = []

    for sym in symbols:
        f1 = rng.standard_normal(n_days)
        f2 = rng.standard_normal(n_days)
        f3 = rng.standard_normal(n_days)
        f4 = rng.standard_normal(n_days)

        # Signal: 0.5 * f1 + noise
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
def feature_frame_tz(feature_frame) -> pd.DataFrame:
    """Timezone-aware variant of feature_frame localized to Asia/Kolkata."""
    df = feature_frame.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize("Asia/Kolkata")
    return df


@pytest.fixture
def feature_dir_factory(tmp_path):
    """
    Factory fixture: writes a given feature DataFrame to per-symbol parquet files
    in a unique subfolder under tmp_path and returns the folder Path.
    """
    counter = 0

    def _create(df: pd.DataFrame) -> Path:
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
    from config.settings import settings
    from src.data_storage.parquet_writer import LAYER_PATH_MAP, DataLayer
    from src.models.dataset_builder import DatasetBuilder
    from src.models.linear_model import LinearModel
    from src.models.model_store import ModelStore
    from src.models.prediction_pipeline import PredictionPipeline
    from src.models.prediction_service import PredictionService
    from src.models.training_pipeline import TrainingPipeline

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
