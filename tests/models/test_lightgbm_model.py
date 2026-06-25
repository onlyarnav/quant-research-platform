"""
Tests for the LightGBMModel wrapper.

Exercises LGBMRegressor initialization, fitting, prediction, and persistence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from src.models.lightgbm_model import LightGBMModel

def test_init_default_hyperparameters():
    """Assert that LightGBMModel uses the specified defaults."""
    model = LightGBMModel()
    assert model.n_estimators == 100
    assert model.max_depth == -1
    assert model.learning_rate == 0.1

def test_model_type_returns_lightgbm():
    """Assert that model_type returns 'lightgbm'."""
    model = LightGBMModel()
    assert model.model_type == "lightgbm"

def test_fit_without_validation_data(regression_data):
    """Assert that fit() works correctly when only training data is provided."""
    X_train, y_train, _, _, _, _ = regression_data
    # verbose=-1 suppresses noisy training logs in pytest output
    model = LightGBMModel(verbose=-1)
    model.fit(X_train, y_train)
    assert model._is_fitted is True

def test_fit_with_validation_data(regression_data):
    """Assert that fit() works correctly when validation data is provided."""
    X_train, y_train, X_val, y_val, _, _ = regression_data
    model = LightGBMModel(verbose=-1)
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    assert model._is_fitted is True

def test_predict_before_fit_raises():
    """Assert that predict() raises RuntimeError if model is not fitted."""
    model = LightGBMModel()
    X = pd.DataFrame(np.random.randn(5, 4))
    with pytest.raises(RuntimeError, match="must be fitted before calling predict"):
        model.predict(X)

def test_predict_after_fit_returns_correct_shape(regression_data):
    """Assert that predict() returns an array of the correct shape."""
    X_train, y_train, _, _, X_test, _ = regression_data
    model = LightGBMModel(verbose=-1)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    assert preds.shape == (len(X_test),)

def test_save_and_load_round_trip(regression_data, tmp_path: Path):
    """Assert that saving and loading a LightGBM model preserves its output."""
    X_train, y_train, _, _, X_test, _ = regression_data

    model_orig = LightGBMModel(verbose=-1)
    model_orig.fit(X_train, y_train)
    save_path = tmp_path / "lgbm.joblib"
    model_orig.save(save_path)
    preds_orig = model_orig.predict(X_test)

    model_loaded = LightGBMModel(verbose=-1)
    model_loaded.load(save_path)
    preds_loaded = model_loaded.predict(X_test)

    assert np.allclose(preds_orig, preds_loaded)
