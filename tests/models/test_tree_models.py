"""
Tests for RandomForestModel and GradientBoostingModel wrappers.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

from src.models.tree_models import GradientBoostingModel, RandomForestModel

# =====================================================================
# Tree Models Tests
# =====================================================================


def test_tree_models_learn_nonlinear_rule() -> None:
    """Assert RandomForest and GradientBoosting achieve rank correlation > 0.8 on test set."""
    rng = np.random.default_rng(42)
    n = 300
    X = pd.DataFrame(
        {
            "f1": rng.standard_normal(n),
            "noise": rng.standard_normal(n),
        }
    )
    # Simple nonlinear rule: 1 if f1 > 0 else -1
    y = pd.Series(np.where(X["f1"] > 0, 1.0, -1.0))

    X_train, X_test = X.iloc[:150], X.iloc[150:]
    y_train, y_test = y.iloc[:150], y.iloc[150:]

    rf = RandomForestModel(
        n_estimators=30, max_depth=4, min_samples_leaf=5, random_state=42
    )
    rf.fit(X_train, y_train)
    rf_preds = rf.predict(X_test)
    rf_corr, _ = spearmanr(y_test, rf_preds)
    assert rf_corr > 0.80

    gb = GradientBoostingModel(
        n_estimators=30, max_depth=3, learning_rate=0.1, random_state=42
    )
    gb.fit(X_train, y_train)
    gb_preds = gb.predict(X_test)
    gb_corr, _ = spearmanr(y_test, gb_preds)
    assert gb_corr > 0.80


def test_tree_models_determinism() -> None:
    """Assert random_state makes two independent fits completely identical."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame({"f1": rng.standard_normal(50), "f2": rng.standard_normal(50)})
    y = pd.Series(rng.standard_normal(50))

    rf1 = RandomForestModel(n_estimators=10, random_state=42)
    rf2 = RandomForestModel(n_estimators=10, random_state=42)
    rf1.fit(X, y)
    rf2.fit(X, y)
    np.testing.assert_allclose(rf1.predict(X), rf2.predict(X))

    gb1 = GradientBoostingModel(n_estimators=10, random_state=42)
    gb2 = GradientBoostingModel(n_estimators=10, random_state=42)
    gb1.fit(X, y)
    gb2.fit(X, y)
    np.testing.assert_allclose(gb1.predict(X), gb2.predict(X))


def test_tree_models_informative_feature_highest_importance() -> None:
    """Assert feature_importances is highest for the informative signal feature."""
    rng = np.random.default_rng(42)
    X = pd.DataFrame(
        {
            "signal": rng.standard_normal(200),
            "noise": rng.standard_normal(200),
        }
    )
    y = pd.Series(2.0 * X["signal"] + 0.1 * rng.standard_normal(200))

    rf = RandomForestModel(n_estimators=30, min_samples_leaf=5, random_state=42)
    rf.fit(X, y)
    rf_imp = rf.get_feature_importance(["signal", "noise"])
    assert rf_imp["signal"] > rf_imp["noise"]

    gb = GradientBoostingModel(n_estimators=30, random_state=42)
    gb.fit(X, y)
    gb_imp = gb.get_feature_importance(["signal", "noise"])
    assert gb_imp["signal"] > gb_imp["noise"]


def test_gradient_boosting_no_internal_early_stopping_and_ignores_val(caplog) -> None:
    """Assert GradientBoosting does not use internal early stopping and logs debug on val."""
    X = pd.DataFrame({"f1": [1.0, 2.0, 3.0]})
    y = pd.Series([1.0, 2.0, 3.0])

    gb = GradientBoostingModel(n_estimators=5, random_state=42)
    with caplog.at_level(logging.DEBUG):
        gb.fit(X, y, X_val=X, y_val=y)

    # Check underlying regressor has n_iter_no_change as None
    underlying = gb.get_underlying_model()
    assert underlying.n_iter_no_change is None
    assert "validation data provided but ignored" in caplog.text
