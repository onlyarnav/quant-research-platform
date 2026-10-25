"""
Tree-based ensemble regression model wrappers (Random Forest and Gradient Boosting).

Implements financial market regularizations:
- Large min_samples_leaf in RandomForest to avoid overfitting noisy asset returns.
- Prohibition of scikit-learn internal early stopping in GradientBoosting, because
  sklearn's internal validation_fraction performs random shuffling which leaks
  future observations across chronological sequences.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor

from src.models.base_model import BaseModel
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RandomForestModel(BaseModel):
    """
    Random Forest ensemble regressor for financial return prediction.

    Financial market targets exhibit low signal-to-noise ratios. Deep trees with
    small leaf sizes readily memorize noise. A relatively large min_samples_leaf
    (default 50) regularizes individual trees, ensuring that predictions reflect
    broad statistical tendencies rather than idiosyncratic sample noise.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 8,
        min_samples_leaf: int = 50,
        n_jobs: int = -1,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the RandomForestModel wrapper.

        Args:
            n_estimators: Number of decision trees in the forest.
            max_depth: Maximum tree depth.
            min_samples_leaf: Minimum samples required at each terminal leaf.
            n_jobs: Parallel worker threads (-1 for all cores).
            random_state: Random seed for deterministic reproducibility.
            **kwargs: Additional parameters passed to sklearn RandomForestRegressor.
        """
        hp = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "min_samples_leaf": min_samples_leaf,
            "n_jobs": n_jobs,
            "random_state": random_state,
            **kwargs,
        }
        super().__init__(model_name="RandomForestModel", **hp)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.kwargs = kwargs

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> None:
        """
        Fit the Random Forest regressor.

        Note: X_val and y_val are ignored as Random Forest fits via bagging
        without early stopping.

        Args:
            X_train: Training features.
            y_train: Training targets.
            X_val: Ignored validation features.
            y_val: Ignored validation targets.
        """
        if X_val is not None or y_val is not None:
            logger.debug(
                "RandomForestModel.fit: validation data provided but ignored "
                "(Random Forest does not support early stopping)."
            )

        self._model = RandomForestRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            min_samples_leaf=self.min_samples_leaf,
            n_jobs=self.n_jobs,
            random_state=self.random_state,
            **self.kwargs,
        )
        self._model.fit(X_train, y_train)
        self._is_fitted = True
        logger.info("Successfully fitted RandomForestModel %s", self.model_name)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions using the fitted Random Forest.

        Args:
            X: Feature set to predict on.

        Returns:
            1-D numpy array of predicted return forecasts.
        """
        self._check_is_fitted()
        return np.asarray(self._model.predict(X)).ravel()

    def get_feature_importance(self, feature_names: Sequence[str]) -> dict[str, float]:
        """
        Return Gini / MDI feature importances.

        Args:
            feature_names: Sequence of feature column names.

        Returns:
            Dictionary mapping feature names to non-negative importance scores.

        Raises:
            RuntimeError: If model is not fitted.
            ValueError: If feature_names length does not match model features.
        """
        self._check_is_fitted()
        importances = self._model.feature_importances_
        if len(feature_names) != len(importances):
            raise ValueError(
                f"feature_names length {len(feature_names)} does not "
                f"match number of features {len(importances)}."
            )
        return {name: float(imp) for name, imp in zip(feature_names, importances)}

    @property
    def model_type(self) -> str:
        """Return the model type identifier."""
        return "random_forest"


class GradientBoostingModel(BaseModel):
    """
    Gradient Boosting regressor wrapper.

    Wraps sklearn.ensemble.GradientBoostingRegressor. Does NOT use sklearn's
    built-in early stopping (n_iter_no_change / validation_fraction) because
    sklearn shuffles data randomly to carve its validation fold, violating
    time-series ordering and causing future data leakage.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 3,
        learning_rate: float = 0.05,
        subsample: float = 0.8,
        random_state: int = 42,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the GradientBoostingModel wrapper.

        Args:
            n_estimators: Number of boosting stages.
            max_depth: Maximum depth of individual estimators.
            learning_rate: Step size shrinkage to shrink individual tree contribution.
            subsample: Fraction of samples used for fitting individual base learners.
            random_state: Random state for deterministic fitting.
            **kwargs: Additional parameters passed to GradientBoostingRegressor.
        """
        hp = {
            "n_estimators": n_estimators,
            "max_depth": max_depth,
            "learning_rate": learning_rate,
            "subsample": subsample,
            "random_state": random_state,
            **kwargs,
        }
        super().__init__(model_name="GradientBoostingModel", **hp)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.random_state = random_state
        self.kwargs = kwargs

    def fit(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> None:
        """
        Fit the Gradient Boosting regressor.

        Note: X_val and y_val are ignored to prevent time-series shuffling leakage.

        Args:
            X_train: Training features.
            y_train: Training targets.
            X_val: Ignored validation features.
            y_val: Ignored validation targets.
        """
        if X_val is not None or y_val is not None:
            logger.debug(
                "GradientBoostingModel.fit: validation data provided but ignored "
                "(internal early stopping disabled to prevent shuffling leakage)."
            )

        self._model = GradientBoostingRegressor(
            n_estimators=self.n_estimators,
            max_depth=self.max_depth,
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            random_state=self.random_state,
            **self.kwargs,
        )
        self._model.fit(X_train, y_train)
        self._is_fitted = True
        logger.info("Successfully fitted GradientBoostingModel %s", self.model_name)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """
        Generate predictions using the fitted Gradient Boosting model.

        Args:
            X: Feature set to predict on.

        Returns:
            1-D numpy array of predicted return values.
        """
        self._check_is_fitted()
        return np.asarray(self._model.predict(X)).ravel()

    def get_feature_importance(self, feature_names: Sequence[str]) -> dict[str, float]:
        """
        Return impurity-based feature importances.

        Args:
            feature_names: Sequence of feature column names.

        Returns:
            Dictionary mapping feature name to importance score.

        Raises:
            RuntimeError: If model is not fitted.
            ValueError: If feature_names length does not match model features.
        """
        self._check_is_fitted()
        importances = self._model.feature_importances_
        if len(feature_names) != len(importances):
            raise ValueError(
                f"feature_names length {len(feature_names)} does not "
                f"match number of features {len(importances)}."
            )
        return {name: float(imp) for name, imp in zip(feature_names, importances)}

    @property
    def model_type(self) -> str:
        """Return the model type identifier."""
        return "gradient_boosting"
