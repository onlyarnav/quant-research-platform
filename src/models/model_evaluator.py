"""
Stateless utility for computing regression and financial prediction metrics.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from src.utils.logger import get_logger

logger = get_logger(__name__)

class ModelEvaluator:
    """
    Stateless utility class for computing regression and financial prediction metrics.
    Provides a consistent way to evaluate the predictive power and quality of models.
    """

    def evaluate(self, y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
        """
        Compute all supported metrics for a set of predictions.

        Args:
            y_true: True target values.
            y_pred: Predicted target values.

        Returns:
            A dictionary containing the calculated metrics.
        """
        return {
            "mse": self.mse(y_true, y_pred),
            "rmse": self.rmse(y_true, y_pred),
            "mae": self.mae(y_true, y_pred),
            "r2": self.r2(y_true, y_pred),
            "directional_accuracy": self.directional_accuracy(y_true, y_pred),
            "ic": self.information_coefficient(y_true, y_pred),
        }

    @staticmethod
    def _validate_inputs(y_true: pd.Series, y_pred: np.ndarray) -> None:
        """
        Ensure inputs are non-empty and of matching length.
        """
        if len(y_true) == 0:
            raise ValueError("y_true cannot be empty.")
        if len(y_true) != len(y_pred):
            raise ValueError(
                "Length mismatch: y_true (%s) and y_pred (%s) must have the same length."
                % (len(y_true), len(y_pred))
            )

    @staticmethod
    def mse(y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Compute Mean Squared Error."""
        ModelEvaluator._validate_inputs(y_true, y_pred)
        return float(mean_squared_error(y_true, y_pred))

    @staticmethod
    def rmse(y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Compute Root Mean Squared Error."""
        ModelEvaluator._validate_inputs(y_true, y_pred)
        return float(np.sqrt(mean_squared_error(y_true, y_pred)))

    @staticmethod
    def mae(y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Compute Mean Absolute Error."""
        ModelEvaluator._validate_inputs(y_true, y_pred)
        return float(mean_absolute_error(y_true, y_pred))

    @staticmethod
    def r2(y_true: pd.Series, y_pred: np.ndarray) -> float:
        """Compute R-squared score."""
        ModelEvaluator._validate_inputs(y_true, y_pred)
        return float(r2_score(y_true, y_pred))

    @staticmethod
    def directional_accuracy(y_true: pd.Series, y_pred: np.ndarray) -> float:
        """
        Compute the percentage of predictions where the sign of the prediction
        matches the sign of the actual target. This is a critical metric in
        financial trading to determine if the model predicts the correct direction.
        """
        ModelEvaluator._validate_inputs(y_true, y_pred)

        # np.sign returns -1, 0, or 1.
        correct_direction = np.sign(y_true) == np.sign(y_pred)
        return float(np.mean(correct_direction))

    @staticmethod
    def information_coefficient(y_true: pd.Series, y_pred: np.ndarray) -> float:
        """
        Compute the Information Coefficient (IC), which is the Spearman rank
        correlation between true values and predictions. IC is a standard
        metric in quant research to assess the quality of a signal.
        """
        ModelEvaluator._validate_inputs(y_true, y_pred)

        if np.std(y_true) == 0 or np.std(y_pred) == 0:
            logger.warning(
                "Cannot compute information coefficient: "
                "y_true or y_pred is constant (zero variance)."
            )
            return float("nan")

        corr, _ = spearmanr(y_true, y_pred)
        return float(corr)