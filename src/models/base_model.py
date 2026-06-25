"""
Abstract base class for ML model wrappers.

Defines a consistent interface for training, prediction, and persistence
so the training pipeline can handle XGBoost, LightGBM, and linear models
interchangeably.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class BaseModel(ABC):
    """
    Abstract base class for all model wrappers.

    Defines a consistent interface for training, prediction, and persistence
    to allow the training pipeline to handle different ML algorithms generically.
    """

    def __init__(self, model_name: str, **hyperparameters) -> None:
        """
        Initialize the base model wrapper.

        Args:
            model_name: Unique identifier for the model instance.
            **hyperparameters: Model-specific hyperparameters.
        """
        self.model_name = model_name
        self.hyperparameters = hyperparameters
        self._model = None
        self._is_fitted: bool = False

    @abstractmethod
    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> None:
        """
        Train the model on the provided training data.

        Args:
            X_train: Training feature set.
            y_train: Training target values.
            X_val: Optional validation feature set for early stopping.
            y_val: Optional validation target values for early stopping.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions for the given feature set.

        Implementations MUST call self._check_is_fitted() as their
        first line before delegating to the underlying model.

        Args:
            X: Feature set to predict on.

        Returns:
            Predicted values as a numpy array.

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        raise NotImplementedError

    def _check_is_fitted(self) -> None:
        """
        Guard clause for predict()/save()/get_underlying_model().

        Subclasses must call this at the start of their predict()
        implementation, since abstract method bodies are never
        executed by Python — each subclass replaces this body entirely.
        """
        if not self._is_fitted:
            raise RuntimeError(
                f"Model {self.model_name} must be fitted before calling predict()."
            )

    def get_underlying_model(self):
        """
        Return the underlying fitted estimator.

        Used by the training pipeline for MLflow artifact logging
        without reaching into a "private" attribute directly.

        Raises:
            RuntimeError: If the model has not been fitted yet.
        """
        self._check_is_fitted()
        return self._model

    def save(self, path: Path) -> None:
        """
        Persist the fitted model to disk using joblib.

        Args:
            path: Path to save the model artifact.

        Raises:
            RuntimeError: If the model is not fitted and cannot be saved.
        """
        if not self._is_fitted:
            raise RuntimeError(f"Cannot save unfitted model {self.model_name}.")

        logger.info("Saving model %s to %s", self.model_name, path)
        joblib.dump(self._model, path)

    def load(self, path: Path) -> None:
        """
        Load a fitted model from disk.

        Args:
            path: Path to the model artifact.
        """
        logger.info("Loading model %s from %s", self.model_name, path)
        self._model = joblib.load(path)
        self._is_fitted = True

    @property
    @abstractmethod
    def model_type(self) -> str:
        """
        Return the type of the underlying model (e.g., 'xgboost', 'linear').

        Used for MLflow logging and model identification.
        """
        raise NotImplementedError