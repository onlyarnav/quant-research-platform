"""
Linear baseline model wrappers (OLS and Lasso) for financial return forecasting.

Provides Ordinary Least Squares (LinearRegression) and L1-penalized regression (Lasso)
wrappers conforming to the platform BaseModel interface. Both models optionally apply
StandardScaler inside an sklearn Pipeline to prevent lookahead data leakage (scaling
statistics are learned strictly on training data).
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.base_model import BaseModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


class OLSModel(BaseModel):
    """
    Ordinary Least Squares (OLS) regression wrapper.

    Wraps sklearn.linear_model.LinearRegression. Standardizes features via
    StandardScaler by default to ensure uniform numerical conditioning across
    features with disparate scales (e.g. volume vs price ratios).
    """

    def __init__(self, scale_features: bool = True, **kwargs: Any) -> None:
        """
        Initialize the OLSModel wrapper.

        Args:
            scale_features: Whether to scale features using StandardScaler.
            **kwargs: Additional parameters passed to LinearRegression.
        """
        hp = {"scale_features": scale_features, **kwargs}
        super().__init__(model_name="OLSModel", **hp)
        self.scale_features = scale_features
        self.kwargs = kwargs

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> None:
        """
        Fit the OLS regression model.

        Note: X_val and y_val are ignored as LinearRegression has no early stopping.

        Args:
            X_train: Training features.
            y_train: Training targets.
            X_val: Ignored validation features.
            y_val: Ignored validation targets.
        """
        if X_val is not None or y_val is not None:
            logger.debug(
                "OLSModel.fit: validation data provided but ignored "
                "(LinearRegression does not support early stopping)."
            )

        regressor = LinearRegression(**self.kwargs)
        if self.scale_features:
            self._model = Pipeline(
                [("scaler", StandardScaler()), ("regressor", regressor)]
            )
        else:
            self._model = regressor

        self._model.fit(X_train, y_train)
        self._is_fitted = True
        logger.info("Successfully fitted OLSModel %s", self.model_name)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions using the fitted OLS model.

        Args:
            X: Feature set to predict on.

        Returns:
            1-D numpy array of predicted values.
        """
        self._check_is_fitted()
        return np.asarray(self._model.predict(X)).ravel()

    def get_feature_importance(self, feature_names: Sequence[str]) -> dict[str, float]:
        """
        Return absolute regression coefficients as feature importance.

        Args:
            feature_names: Sequence of feature column names.

        Returns:
            Dictionary mapping feature name to absolute coefficient value.

        Raises:
            RuntimeError: If model is not fitted.
            ValueError: If feature_names length does not match coefficient count.
        """
        self._check_is_fitted()
        if self.scale_features:
            regressor = self._model.named_steps["regressor"]
        else:
            regressor = self._model
        coefs = np.abs(regressor.coef_).ravel()
        if len(feature_names) != len(coefs):
            raise ValueError(
                f"feature_names length {len(feature_names)} does not "
                f"match number of features {len(coefs)}."
            )
        return {name: float(imp) for name, imp in zip(feature_names, coefs)}

    @property
    def model_type(self) -> str:
        """Return the model type identifier."""
        return "ols"


class LassoModel(BaseModel):
    """
    L1-regularized Lasso regression wrapper for sparse feature selection.

    Wraps sklearn.linear_model.Lasso. Financial asset returns have high noise
    and low signal-to-noise ratio, with typical daily returns of order 0.01.
    Consequently, default alpha is set to 0.0001 (1e-4) rather than 1.0; an alpha
    of 1.0 would excessively penalize coefficients and shrink all predictions to zero.
    """

    def __init__(
        self,
        alpha: float = 0.0001,
        scale_features: bool = True,
        max_iter: int = 10000,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the LassoModel wrapper.

        Args:
            alpha: L1 regularization multiplier (default 1e-4 for return orders).
            scale_features: Whether to scale features with StandardScaler.
            max_iter: Maximum optimization iterations.
            **kwargs: Additional parameters passed to sklearn.linear_model.Lasso.
        """
        hp = {
            "alpha": alpha,
            "scale_features": scale_features,
            "max_iter": max_iter,
            **kwargs,
        }
        super().__init__(model_name="LassoModel", **hp)
        self.alpha = alpha
        self.scale_features = scale_features
        self.max_iter = max_iter
        self.kwargs = kwargs

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> None:
        """
        Fit the Lasso regression model.

        Note: X_val and y_val are ignored as Lasso has no early stopping.

        Args:
            X_train: Training features.
            y_train: Training targets.
            X_val: Ignored validation features.
            y_val: Ignored validation targets.
        """
        if X_val is not None or y_val is not None:
            logger.debug(
                "LassoModel.fit: validation data provided but ignored "
                "(Lasso does not support early stopping)."
            )

        regressor = Lasso(alpha=self.alpha, max_iter=self.max_iter, **self.kwargs)
        if self.scale_features:
            self._model = Pipeline(
                [("scaler", StandardScaler()), ("regressor", regressor)]
            )
        else:
            self._model = regressor

        self._model.fit(X_train, y_train)
        self._is_fitted = True
        logger.info("Successfully fitted LassoModel %s", self.model_name)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions using the fitted Lasso model.

        Args:
            X: Feature set to predict on.

        Returns:
            1-D numpy array of predicted values.
        """
        self._check_is_fitted()
        return np.asarray(self._model.predict(X)).ravel()

    def get_feature_importance(self, feature_names: Sequence[str]) -> dict[str, float]:
        """
        Return absolute Lasso coefficients as feature importance scores.

        Args:
            feature_names: Sequence of feature column names.

        Returns:
            Dictionary mapping feature name to absolute coefficient value.

        Raises:
            RuntimeError: If model is not fitted.
            ValueError: If feature_names length does not match coefficient count.
        """
        self._check_is_fitted()
        if self.scale_features:
            regressor = self._model.named_steps["regressor"]
        else:
            regressor = self._model
        coefs = np.abs(regressor.coef_).ravel()
        if len(feature_names) != len(coefs):
            raise ValueError(
                f"feature_names length {len(feature_names)} does not "
                f"match number of features {len(coefs)}."
            )
        return {name: float(imp) for name, imp in zip(feature_names, coefs)}

    @property
    def model_type(self) -> str:
        """Return the model type identifier."""
        return "lasso"
