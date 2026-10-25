"""
Extended tests for LinearModel: feature scaling option and feature importance.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline

from src.models.linear_model import LinearModel


def test_linear_model_scale_features_default() -> None:
    """Assert scale_features default (False) leaves hyperparameters == {'alpha': 1.0}."""
    model = LinearModel()
    assert model.hyperparameters == {"alpha": 1.0}
    assert model.scale_features is False


def test_linear_model_scale_features_true() -> None:
    """Assert scale_features=True adds key to hyperparameters and builds Pipeline."""
    model = LinearModel(alpha=2.0, scale_features=True)
    assert model.hyperparameters["scale_features"] is True
    assert model.hyperparameters["alpha"] == 2.0

    X = pd.DataFrame({"f1": [1.0, 2.0, 3.0], "f2": [4.0, 5.0, 6.0]})
    y = pd.Series([1.0, 2.0, 3.0])
    model.fit(X, y)

    underlying = model.get_underlying_model()
    assert isinstance(underlying, Pipeline)
    assert "scaler" in underlying.named_steps
    assert "regressor" in underlying.named_steps


def test_linear_model_feature_importance_equals_abs_coef() -> None:
    """Assert get_feature_importance returns absolute coefficients."""
    X = pd.DataFrame(
        {
            "f1": [1.0, 2.0, 3.0, 4.0, 5.0],
            "f2": [-2.0, -4.0, -6.0, -8.0, -10.0],
        }
    )
    y = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])

    model = LinearModel(alpha=0.01)
    model.fit(X, y)

    imp = model.get_feature_importance(["f1", "f2"])
    assert imp["f1"] >= 0.0
    assert imp["f2"] >= 0.0

    underlying = model.get_underlying_model()
    np.testing.assert_allclose(
        [imp["f1"], imp["f2"]],
        np.abs(underlying.coef_),
    )
