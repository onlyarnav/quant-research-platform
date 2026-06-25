"""
Tests for the BaseModel abstract base class.

Exercises the common interface for fit, predict, save, and load,
ensuring that abstract enforcement and fitting guards are correctly implemented.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from pathlib import Path
from src.models.base_model import BaseModel

class DummyModel(BaseModel):
    """Concrete subclass of BaseModel for testing shared base-class behavior."""
    def fit(self, X_train: pd.DataFrame, y_train: pd.Series, X_val=None, y_val=None) -> None:
        self._model = {"trained_on_rows": len(X_train)}
        self._is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        self._check_is_fitted()
        return np.zeros(len(X))

    @property
    def model_type(self) -> str:
        return "dummy"

def test_cannot_instantiate_base_model_directly():
    """Confirm that BaseModel cannot be instantiated because it is an ABC."""
    with pytest.raises(TypeError, match="BaseModel"):
        BaseModel(model_name="abstract")

def test_init_sets_attributes():
    """Assert that __init__ correctly sets the base attributes."""
    model = DummyModel(model_name="d1", alpha=0.5)
    assert model.model_name == "d1"
    assert model.hyperparameters == {"alpha": 0.5}
    assert model._model is None
    assert model._is_fitted is False

def test_check_is_fitted_raises_before_fit():
    """Assert that predict() raises RuntimeError if model is not fitted."""
    model = DummyModel(model_name="d1")
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.standard_normal((5, 4)))
    with pytest.raises(RuntimeError, match="d1 must be fitted"):
        model.predict(X)

def test_check_is_fitted_passes_after_fit():
    """Assert that predict() does not raise after fit() is called."""
    model = DummyModel(model_name="d1")
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.standard_normal((5, 4)))
    y = pd.Series(rng.standard_normal(5))
    model.fit(X, y)
    # Should not raise
    model.predict(X)

def test_get_underlying_model_raises_before_fit():
    """Assert that get_underlying_model() raises before fit."""
    model = DummyModel(model_name="d1")
    with pytest.raises(RuntimeError, match="d1 must be fitted"):
        model.get_underlying_model()

def test_get_underlying_model_returns_model_after_fit():
    """Assert that get_underlying_model() returns the internal estimator after fit."""
    model = DummyModel(model_name="d1")
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.standard_normal((5, 4)))
    y = pd.Series(rng.standard_normal(5))
    model.fit(X, y)
    assert model.get_underlying_model() == model._model

def test_save_raises_if_not_fitted(tmp_path: Path):
    """Assert that save() raises RuntimeError if model is unfitted."""
    model = DummyModel(model_name="d1")
    with pytest.raises(RuntimeError, match="Cannot save unfitted model"):
        model.save(tmp_path / "model.joblib")

def test_save_writes_file(tmp_path: Path):
    """Assert that save() actually writes the model to disk."""
    model = DummyModel(model_name="d1")
    rng = np.random.default_rng(42)
    X = pd.DataFrame(rng.standard_normal((5, 4)))
    y = pd.Series(rng.standard_normal(5))
    model.fit(X, y)

    save_path = tmp_path / "model.joblib"
    model.save(save_path)
    assert save_path.exists()

def test_load_sets_is_fitted_true(tmp_path: Path):
    """Assert that loading a model sets _is_fitted to True."""
    # 1. Save a fitted model
    rng = np.random.default_rng(42)
    orig_model = DummyModel(model_name="d1")
    orig_model.fit(pd.DataFrame(rng.standard_normal((5, 4))), pd.Series(rng.standard_normal(5)))
    save_path = tmp_path / "model.joblib"
    orig_model.save(save_path)

    # 2. Load into a fresh unfitted model
    new_model = DummyModel(model_name="d1")
    assert new_model._is_fitted is False
    new_model.load(save_path)
    assert new_model._is_fitted is True

def test_load_restores_model_content(tmp_path: Path):
    """Assert that load() restores the exact content of the underlying model."""
    rng = np.random.default_rng(42)
    orig_model = DummyModel(model_name="d1")
    X = pd.DataFrame(rng.standard_normal((5, 4)))
    y = pd.Series(rng.standard_normal(5))
    orig_model.fit(X, y)

    save_path = tmp_path / "model.joblib"
    orig_model.save(save_path)

    new_model = DummyModel(model_name="d1")
    new_model.load(save_path)

    assert new_model._model == orig_model._model

def test_model_type_is_abstract():
    """
    Confirm that the ABC machinery blocks instantiation of subclasses
    that do not implement the required abstract properties.
    """
    class IncompleteModel(BaseModel):
        def fit(self, X_train, y_train, X_val=None, y_val=None) -> None:
            pass
        def predict(self, X):
            return np.zeros(len(X))
        # model_type is missing

    with pytest.raises(TypeError, match="IncompleteModel"):
        IncompleteModel(model_name="incomplete")
