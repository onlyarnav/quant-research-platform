"""
Wrapper for sklearn.linear_model.Ridge for financial feature prediction.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.models.base_model import BaseModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LinearModel(BaseModel):
    """
    Wrapper for sklearn.linear_model.Ridge.
    Ridge regression is preferred over plain LinearRegression in financial contexts
    because it handles multicollinearity among features through L2 regularization.
    """

    def __init__(
        self,
        alpha: float = 1.0,
        *,
        scale_features: bool = False,
        **kwargs,
    ) -> None:
        """
        Initialize the LinearModel wrapper.

        Args:
            alpha: Regularization strength. Larger values specify stronger penalty.
            scale_features: Whether to scale features with StandardScaler.
            **kwargs: Additional arguments passed to Ridge regressor.
        """
        hp = {"alpha": alpha, **kwargs}
        if scale_features:
            hp["scale_features"] = True
        super().__init__(model_name="LinearModel", **hp)
        self.alpha = alpha
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
        Fit the Ridge regression model.

        Note: X_val and y_val are ignored as Ridge does not support early stopping.
        """
        if X_val is not None or y_val is not None:
            logger.debug(
                "LinearModel.fit: validation data provided but ignored "
                "(Ridge does not support early stopping)."
            )

        if self.scale_features:
            self._model = Pipeline(
                [
                    ("scaler", StandardScaler()),
                    ("regressor", Ridge(alpha=self.alpha, **self.kwargs)),
                ]
            )
        else:
            self._model = Ridge(alpha=self.alpha, **self.kwargs)

        self._model.fit(X_train, y_train)
        self._is_fitted = True
        logger.info("Successfully fitted LinearModel %s", self.model_name)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Generate predictions using the fitted Ridge model."""
        self._check_is_fitted()
        return np.asarray(self._model.predict(X))

    def get_feature_importance(self, feature_names: Sequence[str]) -> dict[str, float]:
        """Return absolute Ridge regression coefficients keyed by feature name."""
        self._check_is_fitted()
        if self.scale_features:
            regressor = self._model.named_steps["regressor"]
        else:
            regressor = self._model
        coefs = np.abs(regressor.coef_)
        if len(feature_names) != len(coefs):
            raise ValueError(
                f"feature_names length {len(feature_names)} does not "
                f"match number of features {len(coefs)}."
            )
        return {name: float(imp) for name, imp in zip(feature_names, coefs)}

    @property
    def model_type(self) -> str:
        """Return the model type identifier."""
        return "linear"
