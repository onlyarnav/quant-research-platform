"""
Wrapper for lightgbm.LGBMRegressor with early stopping support.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import lightgbm as lgb
from src.models.base_model import BaseModel
from src.utils.logger import get_logger

logger = get_logger(__name__)

class LightGBMModel(BaseModel):
    """
    Wrapper for lightgbm.LGBMRegressor.
    LightGBM is often faster and more memory-efficient than XGBoost while maintaining
    similar performance on large financial datasets.
    """

    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = -1,
        learning_rate: float = 0.1,
        **kwargs,
    ) -> None:
        """
        Initialize the LightGBMModel wrapper.

        Args:
            n_estimators: Number of boosting rounds.
            max_depth: Maximum depth of a tree. -1 means no limit.
            learning_rate: Step size shrinkage used in update to prevent overfitting.
            **kwargs: Additional arguments passed to lightgbm.LGBMRegressor.
        """
        super().__init__(
            model_name="LightGBMModel",
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
        Fit the LightGBM model with optional early stopping.

        Args:
            X_train: Training features.
            y_train: Training targets.
            X_val: Validation features for early stopping.
            y_val: Validation targets for early stopping.
        """
        self._model = lgb.LGBMRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            **self.kwargs,
        )

        if X_val is not None and y_val is not None:
            logger.info("Fitting LightGBMModel %s with early stopping.", self.model_name)
            self._model.fit(
                X_train,
                y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[lgb.early_stopping(stopping_rounds=20)],
            )
        else:
            logger.info("Fitting LightGBMModel %s without early stopping.", self.model_name)
            self._model.fit(X_train, y_train)

        self._is_fitted = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions using the fitted LightGBM model.
        """
        self._check_is_fitted()
        return self._model.predict(X)

    @property
    def model_type(self) -> str:
        """Return the model type identifier."""
        return "lightgbm"
