"""
Tests for OLSModel and LassoModel wrappers.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.baseline_models import LassoModel, OLSModel

# =====================================================================
# OLSModel Tests
# =====================================================================


def test_ols_recovers_noiseless_coefficients() -> None:
    """Assert OLS recovers exact coefficients on noiseless linear data (y = 2*f1 - 3*f2)."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        {
            "f1": rng.standard_normal(100),
            "f2": rng.standard_normal(100),
        }
    )
    # y = 2*f1 - 3*f2
    y = 2.0 * X["f1"] - 3.0 * X["f2"]

    model = OLSModel(scale_features=False)
    model.fit(X, y)
    imp = model.get_feature_importance(["f1", "f2"])

    # Coefficients: f1 -> 2.0, f2 -> 3.0 (absolute)
    assert imp["f1"] == pytest.approx(2.0, rel=1e-4)
    assert imp["f2"] == pytest.approx(3.0, rel=1e-4)


def test_ols_predictions_identical_with_and_without_scaling_noiseless() -> None:
    """Assert predictions are identical on noiseless data whether scaling is enabled or not."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame({"f1": rng.standard_normal(50), "f2": rng.standard_normal(50)})
    y = 1.5 * X["f1"] + 0.5 * X["f2"]

    model_unscaled = OLSModel(scale_features=False)
    model_scaled = OLSModel(scale_features=True)

    model_unscaled.fit(X, y)
    model_scaled.fit(X, y)

    preds_unscaled = model_unscaled.predict(X)
    preds_scaled = model_scaled.predict(X)

    np.testing.assert_allclose(preds_unscaled, preds_scaled, rtol=1e-5, atol=1e-5)


def test_ols_scaler_fitted_on_training_data_only() -> None:
    """Assert StandardScaler in Pipeline reflects training data statistics only."""
    X_train = pd.DataFrame({"f1": [10.0, 20.0, 30.0]})
    y_train = pd.Series([1.0, 2.0, 3.0])

    model = OLSModel(scale_features=True)
    model.fit(X_train, y_train)

    pipeline = model.get_underlying_model()
    assert isinstance(pipeline, Pipeline)
    scaler: StandardScaler = pipeline.named_steps["scaler"]
    # Mean of [10, 20, 30] is 20.0
    assert scaler.mean_[0] == pytest.approx(20.0)


def test_ols_logs_debug_on_validation_data(caplog) -> None:
    """Assert OLS logs debug line when validation data is provided."""
    X_train = pd.DataFrame({"f1": [1.0, 2.0]})
    y_train = pd.Series([1.0, 2.0])
    model = OLSModel()

    with caplog.at_level(logging.DEBUG):
        model.fit(X_train, y_train, X_val=X_train, y_val=y_train)

    assert "validation data provided but ignored" in caplog.text


# =====================================================================
# LassoModel Tests
# =====================================================================


def test_lasso_large_alpha_shrinks_coefficients() -> None:
    """Assert Lasso with large alpha shrinks feature importances to near zero."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame({"f1": rng.standard_normal(100), "f2": rng.standard_normal(100)})
    y = 0.05 * X["f1"] + 0.01 * rng.standard_normal(100)

    # Very large alpha for daily returns (e.g. 1.0) shrinks everything
    model = LassoModel(alpha=1.0)
    model.fit(X, y)
    imp = model.get_feature_importance(["f1", "f2"])
    assert imp["f1"] == pytest.approx(0.0, abs=1e-6)
    assert imp["f2"] == pytest.approx(0.0, abs=1e-6)


def test_lasso_default_alpha_selects_true_feature() -> None:
    """Assert Lasso with small default alpha (1e-4) identifies informative feature."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        {
            "informative": rng.standard_normal(200),
            "noise": rng.standard_normal(200),
        }
    )
    # Target scale ~ 0.01 typical for daily returns
    y = 0.02 * X["informative"] + 0.001 * rng.standard_normal(200)

    model = LassoModel()
    model.fit(X, y)
    imp = model.get_feature_importance(["informative", "noise"])
    assert imp["informative"] > imp["noise"]


def test_lasso_logs_debug_on_validation_data(caplog) -> None:
    """Assert Lasso logs debug line when validation data is provided."""
    X_train = pd.DataFrame({"f1": [1.0, 2.0]})
    y_train = pd.Series([1.0, 2.0])
    model = LassoModel()

    with caplog.at_level(logging.DEBUG):
        model.fit(X_train, y_train, X_val=X_train, y_val=y_train)

    assert "validation data provided but ignored" in caplog.text
