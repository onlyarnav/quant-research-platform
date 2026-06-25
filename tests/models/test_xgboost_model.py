"""
Tests for the XGBoostModel wrapper.

Exercises XGBRegressor initialization, fitting with/without early stopping (XGBoost 2.0+),
and persistence.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from src.models.xgboost_model import XGBoostModel

def test_init_default_hyperparameters():
    """Assert that XGBoostModel uses the specified defaults."""
    model = XGBoostModel()
    assert model.n_estimators == 100
    assert model.max_depth == 6
    assert model.learning_rate == 0.1

def test_model_type_returns_xgboost():
    """Assert that model_type returns 'xgboost'."""
    model = XGBoostModel()
    assert model.model_type == "xgboost"

def test_fit_without_validation_data(regression_data):
    """Assert that fit() works correctly when only training data is provided."""
    X_train, y_train, _, _, _, _ = regression_data
    model = XGBoostModel()
    model.fit(X_train, y_train)
    assert model._is_fitted is True

def test_fit_with_validation_data_sets_early_stopping_rounds(regression_data):
    """Assert that providing validation data sets early_stopping_rounds in the XGBoost model."""
    X_train, y_train, X_val, y_val, _, _ = regression_data
    model = XGBoostModel()
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)

    params = model.get_underlying_model().get_params()
    assert params["early_stopping_rounds"] == 20

def test_fit_without_validation_data_disables_early_stopping(regression_data):
    """Assert that omitting validation data sets early_stopping_rounds to None (XGBoost 2.0 fix)."""
    X_train, y_train, _, _, _, _ = regression_data
    model = XGBoostModel()
    model.fit(X_train, y_train)

    params = model.get_underlying_model().get_params()
    assert params["early_stopping_rounds"] is None

def test_predict_before_fit_raises():
    """Assert that predict() raises RuntimeError if model is not fitted."""
    model = XGBoostModel()
    X = pd.DataFrame(np.random.randn(5, 4))
    with pytest.raises(RuntimeError, match="must be fitted before calling predict"):
        model.predict(X)

def test_predict_after_fit_returns_correct_shape(regression_data):
    """Assert that predict() returns an array of the correct shape."""
    X_train, y_train, _, _, X_test, _ = regression_data
    model = XGBoostModel()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    assert preds.shape == (len(X_test),)

def test_save_and_load_round_trip(regression_data, tmp_path: Path):
    """Assert that saving and loading an XGBoost model preserves its output."""
    X_train, y_train, _, _, X_test, _ = regression_data

    model_orig = XGBoostModel()
    model_orig.fit(X_train, y_train)
    save_path = tmp_path / "xgb.joblib"
    model_orig.save(save_path)
    preds_orig = model_orig.predict(X_test)

    model_loaded = XGBoostModel()
    model_loaded.load(save_path)
    preds_loaded = model_loaded.predict(X_test)

    assert np.allclose(preds_orig, preds_loaded)
