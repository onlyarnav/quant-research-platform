"""
Pytest fixtures for ML dataset builder testing.

Provides a temporary feature directory containing synthetic parquet
files used to exercise DatasetBuilder without touching real data.
"""

from __future__ import annotations

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
        df.loc[17:19, ["future_return_1d", "future_return_5d", "future_return_10d"]] = np.nan

        file_path = features_dir / f"{symbol}.parquet"
        df.to_parquet(file_path, index=False)

    return features_dir

@pytest.fixture
def regression_data():
    """
    Shared synthetic train/val/test data for ML model wrapper tests.
    """
    rng = np.random.default_rng(7)
    X_train = pd.DataFrame(rng.standard_normal((50, 4)), columns=["f1", "f2", "f3", "f4"])
    y_train = pd.Series(rng.standard_normal(50))
    X_val = pd.DataFrame(rng.standard_normal((10, 4)), columns=["f1", "f2", "f3", "f4"])
    y_val = pd.Series(rng.standard_normal(10))
    X_test = pd.DataFrame(rng.standard_normal((10, 4)), columns=["f1", "f2", "f3", "f4"])
    y_test = pd.Series(rng.standard_normal(10))

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
