"""
Parametrized common interface tests across all registered model wrappers.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.models.model_factory import available_model_types, create_model


@pytest.mark.parametrize("model_type", available_model_types())
def test_wrapper_model_type_property(model_type: str) -> None:
    """Assert create_model(key).model_type == key."""
    model = create_model(model_type)
    assert model.model_type == model_type


@pytest.mark.parametrize("model_type", available_model_types())
def test_wrapper_fit_predict_contract(model_type: str, regression_data) -> None:
    """Assert model fits and predict returns a 1-D array of correct length."""
    X_train, y_train, _, _, X_test, _ = regression_data
    # Use fast hyperparameters for trees
    kwargs = {}
    if model_type in ("random_forest", "gradient_boosting", "xgboost", "lightgbm"):
        kwargs = {"n_estimators": 5, "max_depth": 2}
        if model_type == "random_forest":
            kwargs["min_samples_leaf"] = 2
        if model_type == "lightgbm":
            kwargs["verbose"] = -1
        if model_type == "xgboost":
            kwargs["n_jobs"] = 1

    model = create_model(model_type, **kwargs)
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    assert isinstance(preds, np.ndarray)
    assert preds.ndim == 1
    assert len(preds) == len(X_test)


@pytest.mark.parametrize("model_type", available_model_types())
def test_wrapper_predict_before_fit_raises(model_type: str, regression_data) -> None:
    """Assert predict before fit raises RuntimeError with expected substring."""
    _, _, _, _, X_test, _ = regression_data
    model = create_model(model_type)
    with pytest.raises(RuntimeError, match="must be fitted before calling predict"):
        model.predict(X_test)


@pytest.mark.parametrize("model_type", available_model_types())
def test_wrapper_save_load_round_trip(
    model_type: str, regression_data, tmp_path
) -> None:
    """Assert saving and loading produces identical predictions."""
    X_train, y_train, _, _, X_test, _ = regression_data
    kwargs = {}
    if model_type in ("random_forest", "gradient_boosting", "xgboost", "lightgbm"):
        kwargs = {"n_estimators": 5, "max_depth": 2}
        if model_type == "random_forest":
            kwargs["min_samples_leaf"] = 2
        if model_type == "lightgbm":
            kwargs["verbose"] = -1
        if model_type == "xgboost":
            kwargs["n_jobs"] = 1

    model = create_model(model_type, **kwargs)
    model.fit(X_train, y_train)
    orig_preds = model.predict(X_test)

    save_path = tmp_path / f"{model_type}.joblib"
    model.save(save_path)

    reloaded = create_model(model_type, **kwargs)
    reloaded.load(save_path)
    loaded_preds = reloaded.predict(X_test)

    np.testing.assert_allclose(orig_preds, loaded_preds)


@pytest.mark.parametrize("model_type", available_model_types())
def test_wrapper_accepts_validation_data(model_type: str, regression_data) -> None:
    """Assert fitting accepts validation data without raising error."""
    X_train, y_train, X_val, y_val, _, _ = regression_data
    kwargs = {}
    if model_type in ("random_forest", "gradient_boosting", "xgboost", "lightgbm"):
        kwargs = {"n_estimators": 5, "max_depth": 2}
        if model_type == "random_forest":
            kwargs["min_samples_leaf"] = 2
        if model_type == "lightgbm":
            kwargs["verbose"] = -1
        if model_type == "xgboost":
            kwargs["n_jobs"] = 1

    model = create_model(model_type, **kwargs)
    model.fit(X_train, y_train, X_val=X_val, y_val=y_val)
    assert model._is_fitted


@pytest.mark.parametrize("model_type", available_model_types())
def test_wrapper_feature_importance_contract(model_type: str, regression_data) -> None:
    """Assert get_feature_importance returns non-negative dict and validates length."""
    X_train, y_train, _, _, _, _ = regression_data
    kwargs = {}
    if model_type in ("random_forest", "gradient_boosting", "xgboost", "lightgbm"):
        kwargs = {"n_estimators": 5, "max_depth": 2}
        if model_type == "random_forest":
            kwargs["min_samples_leaf"] = 2
        if model_type == "lightgbm":
            kwargs["verbose"] = -1
        if model_type == "xgboost":
            kwargs["n_jobs"] = 1

    model = create_model(model_type, **kwargs)
    model.fit(X_train, y_train)

    feature_names = list(X_train.columns)
    importance = model.get_feature_importance(feature_names)

    assert isinstance(importance, dict)
    assert len(importance) == len(feature_names)
    for name in feature_names:
        assert name in importance
        assert importance[name] >= 0.0
        assert np.isfinite(importance[name])

    # Wrong length raises ValueError
    with pytest.raises(ValueError, match="does not match"):
        model.get_feature_importance(["f1", "f2"])


@pytest.mark.parametrize("model_type", available_model_types())
def test_wrapper_hyperparameters_dict_populated(model_type: str) -> None:
    """Assert hyperparameters dict contains constructor arguments."""
    model = create_model(model_type)
    assert isinstance(model.hyperparameters, dict)
    assert len(model.hyperparameters) > 0
