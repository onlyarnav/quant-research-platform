"""
Wrapper for sklearn.linear_model.Ridge for financial feature prediction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from src.models.base_model import BaseModel
from src.utils.logger import get_logger

logger = get_logger(__name__)

class LinearModel(BaseModel):
    """
    Wrapper for sklearn.linear_model.Ridge.
    Ridge regression is preferred over plain LinearRegression in financial contexts
    because it handles multicollinearity among features through L2 regularization.
    """

    def __init__(self, alpha: float = 1.0, **kwargs) -> None:
        """
        Initialize the LinearModel wrapper.

        Args:
            alpha: Regularization strength. Larger values specify stronger regularization.
            **kwargs: Additional arguments passed to sklearn.linear_model.Ridge.
        """
        super().__init__(model_name="LinearModel", **{"alpha": alpha, **kwargs})
        self.alpha = alpha
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

        self._model = Ridge(alpha=self.alpha, **self.kwargs)
        self._model.fit(X_train, y_train)
        self._is_fitted = True
        logger.info("Successfully fitted LinearModel %s", self.model_name)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions using the fitted Ridge model.
        """
        self._check_is_fitted()
        return self._model.predict(X)

    @property
    def model_type(self) -> str:
        """Return the model type identifier."""
        return "linear"
