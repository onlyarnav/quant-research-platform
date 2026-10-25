"""
Tests for ModelFactory and registry operations.
"""

from __future__ import annotations

import pytest

from src.models.base_model import BaseModel
from src.models.model_factory import (
    MODEL_REGISTRY,
    available_model_types,
    create_model,
    default_model_name,
    register_model,
)


def test_registry_contains_seven_keys() -> None:
    """Assert registry contains exactly the 7 documented keys."""
    expected = {
        "ols",
        "linear",
        "lasso",
        "random_forest",
        "gradient_boosting",
        "xgboost",
        "lightgbm",
    }
    assert set(MODEL_REGISTRY.keys()) == expected


def test_available_model_types_is_sorted() -> None:
    """Assert available_model_types returns an alphabetically sorted list."""
    types = available_model_types()
    assert types == sorted(types)


def test_create_model_unknown_type_raises() -> None:
    """Assert create_model with unknown type raises ValueError containing available types."""
    with pytest.raises(ValueError, match="Unknown model_type='deep_nn'"):
        create_model("deep_nn")


def test_register_model_and_rejection(monkeypatch) -> None:
    """Assert register_model adds a new class, rejects duplicates and non-BaseModel classes."""

    class DummyCustomModel(BaseModel):
        def fit(self, X_train, y_train, X_val=None, y_val=None):
            pass

        def predict(self, X):
            pass

        @property
        def model_type(self) -> str:
            return "dummy_custom"

    # Use monkeypatch to ensure isolation
    registry_copy = dict(MODEL_REGISTRY)
    monkeypatch.setattr("src.models.model_factory.MODEL_REGISTRY", registry_copy)

    # Register successfully
    register_model("dummy_custom", DummyCustomModel)
    assert "dummy_custom" in available_model_types()
    m = create_model("dummy_custom", model_name="custom")
    assert m.model_type == "dummy_custom"

    # Duplicate raises ValueError
    with pytest.raises(ValueError, match="already registered"):
        register_model("dummy_custom", DummyCustomModel)

    # Non-BaseModel raises TypeError
    class NotAModel:
        pass

    with pytest.raises(TypeError, match="subclass of BaseModel"):
        register_model("not_a_model", NotAModel)  # type: ignore


def test_default_model_name() -> None:
    """Assert default_model_name formats correctly."""
    assert default_model_name("xgboost", 5) == "xgboost_h5d"
    assert default_model_name("ols", 1) == "ols_h1d"
