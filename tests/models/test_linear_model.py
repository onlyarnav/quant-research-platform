"""
Tests for the LinearModel wrapper.

Exercises Ridge regression initialization, fitting, prediction, and persistence
using synthetic regression data.
"""
from __future__ import annotations

import logging
import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from src.models.linear_model import LinearModel

def test_init_default_alpha():
    """Assert that LinearModel uses alpha=1.0 by default."""
    model = LinearModel()
    assert model.alpha == 1.0
    assert model.hyperparameters == {"alpha": 1.0}

def test_init_custom_alpha_stored():
    """Assert that custom alpha is correctly stored in hyperparameters."""
    model = LinearModel(alpha=0.5)
    assert model.hyperparameters["alpha"] == 0.5

def test_model_type_returns_linear():
    """Assert that model_type returns 'linear'."""
    model = LinearModel()
    assert model.model_type == "linear"

def test_fit_sets_is_fitted_true(regression_data):
    """Assert that calling fit() marks the model as fitted."""
    X_train, y_train, _, _, _, _ = regression_data
    model = LinearModel()
    model.fit(X_train, y_train)
    assert model._is_fitted is True

def test_fit_ignores_validation_data_without_error(regression_data):
    """Assert that fit() accepts X_val/y_val without raising, despite Ridge not using them."""
    X_train, y_train, X_val, y_val, _, _ = regression_data
    model = LinearModel()
    # Should complete without exception
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    assert model._is_fitted is True

def test_fit_logs_debug_when_validation_provided(regression_data, caplog):
    """Assert that fit() logs a debug message when validation data is provided."""
    caplog.set_level(logging.DEBUG)
    X_train, y_train, X_val, y_val, _, _ = regression_data
    model = LinearModel()
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    assert any("validation data provided but ignored" in rec.message for rec in caplog.records)

def test_predict_before_fit_raises_runtime_error():
    """Assert that predict() raises RuntimeError if model is not fitted."""
    model = LinearModel()
    X = pd.DataFrame(np.random.randn(5, 4))
    with pytest.raises(RuntimeError, match="must be fitted before calling predict"):
        model.predict(X)

def test_predict_after_fit_returns_correct_length(regression_data):
    """Assert that predict() returns an array of the correct length."""
    X_train, y_train, _, _, X_test, _ = regression_data
    model = LinearModel()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    assert len(preds) == len(X_test)

def test_predict_returns_ndarray(regression_data):
    """Assert that predict() returns a numpy ndarray."""
    X_train, y_train, _, _, X_test, _ = regression_data
    model = LinearModel()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    assert isinstance(preds, np.ndarray)

def test_save_and_load_round_trip(regression_data, tmp_path: Path):
    """Assert that saving and loading a model preserves its predictive output."""
    X_train, y_train, _, _, X_test, _ = regression_data

    # 1. Fit and save
    model_orig = LinearModel()
    model_orig.fit(X_train, y_train)
    save_path = tmp_path / "linear.joblib"
    model_orig.save(save_path)
    preds_orig = model_orig.predict(X_test)

    # 2. Load into fresh model and predict
    model_loaded = LinearModel()
    model_loaded.load(save_path)
    preds_loaded = model_loaded.predict(X_test)

    assert np.allclose(preds_orig, preds_loaded)
