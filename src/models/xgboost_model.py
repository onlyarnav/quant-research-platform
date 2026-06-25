"""
Wrapper for xgboost.XGBRegressor with early stopping support.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import xgboost as xgb
from src.models.base_model import BaseModel
from src.utils.logger import get_logger

logger = get_logger(__name__)

class XGBoostModel(BaseModel):
    """
    Wrapper for xgboost.XGBRegressor.
    XGBoost provides gradient boosted decision trees that often capture non-linear
    relationships in financial data better than linear models.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 6,
        learning_rate: float = 0.1,
        **kwargs,
    ) -> None:
        """
        Initialize the XGBoostModel wrapper.

        Args:
            n_estimators: Number of boosting rounds.
            max_depth: Maximum depth of a tree.
            learning_rate: Step size shrinkage used in update to prevent overfitting.
            **kwargs: Additional arguments passed to xgboost.XGBRegressor.
        """
        super().__init__(
            model_name="XGBoostModel",
            **{
                "n_estimators": n_estimators,
                "max_depth": max_depth,
                "learning_rate": learning_rate,
                **kwargs,
            },
        )
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.kwargs = kwargs

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> None:
        """
        Fit the XGBoost model with optional early stopping.

        Args:
            X_train: Training features.
            y_train: Training targets.
            X_val: Validation features for early stopping.
            y_val: Validation targets for early stopping.
        """
        has_validation = X_val is not None and y_val is not None

        self._model = xgb.XGBRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            early_stopping_rounds=20 if has_validation else None,
            **self.kwargs,
        )

        if has_validation:
            logger.info("Fitting XGBoostModel %s with early stopping.", self.model_name)
            self._model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        else:
            logger.info("Fitting XGBoostModel %s without early stopping.", self.model_name)
            self._model.fit(X_train, y_train)

        self._is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions using the fitted XGBoost model.
        """
        self._check_is_fitted()
        return self._model.predict(X)

    @property
    def model_type(self) -> str:
        """Return the model type identifier."""
        return "xgboost"
