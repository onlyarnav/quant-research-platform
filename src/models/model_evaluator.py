"""
Stateless utility for computing regression and financial prediction metrics.

Provides statistical and quantitative finance evaluation metrics:
- Base regression metrics: MSE, RMSE, MAE, R-squared
- Financial accuracy metrics: Directional Accuracy, Information Coefficient (IC)
- Extended metrics: Pearson IC, Daily Rank IC, Information Ratio (IC IR)
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from src.utils.logger import get_logger

logger = get_logger(__name__)

HIGHER_IS_BETTER: dict[str, bool] = {
    "mse": False,
    "rmse": False,
    "mae": False,
    "r2": True,
    "directional_accuracy": True,
    "ic": True,
    "pearson_ic": True,
    "rank_ic": True,
    "mean_daily_rank_ic": True,
    "daily_rank_ic_std": False,
    "ic_ir": True,
    "daily_rank_ic_positive_rate": True,
}


class ModelEvaluator:
    """
    Stateless utility class for computing regression and financial prediction metrics.
    Provides standard and cross-sectional metric calculations across model forecasts.
    """

    def evaluate(self, y_true: pd.Series, y_pred: np.ndarray) -> dict[str, float]:
        """
        Compute the six canonical evaluation metrics for a set of predictions.

        Args:
            y_true: True target values.
            y_pred: Predicted target values.

        Returns:
            Dictionary with exactly: mse, rmse, mae, r2, directional_accuracy, ic.
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
    def _validate_inputs(
        y_true: pd.Series | np.ndarray, y_pred: np.ndarray | pd.Series
    ) -> None:
        """Ensure inputs are non-empty and of matching length."""
        if len(y_true) == 0:
            raise ValueError("y_true cannot be empty.")
        if len(y_true) != len(y_pred):
            raise ValueError(
                "Length mismatch: y_true (%s) and y_pred (%s) "
                "must have the same length." % (len(y_true), len(y_pred))
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
        Compute percentage of predictions matching the sign of actual target.
        """
        ModelEvaluator._validate_inputs(y_true, y_pred)
        correct_direction = np.sign(y_true) == np.sign(y_pred)
        return float(np.mean(correct_direction))

    @staticmethod
    def information_coefficient(
        y_true: pd.Series | np.ndarray, y_pred: np.ndarray | pd.Series
    ) -> float:
        """
        Compute Spearman rank correlation (Information Coefficient).

        Returns NaN if either input has zero standard deviation (constant array).
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

    rank_information_coefficient = information_coefficient

    @staticmethod
    def pearson_information_coefficient(
        y_true: pd.Series | np.ndarray, y_pred: np.ndarray | pd.Series
    ) -> float:
        """
        Compute Pearson correlation coefficient between true values and predictions.

        Returns NaN if either input is constant.
        """
        ModelEvaluator._validate_inputs(y_true, y_pred)

        if np.std(y_true) == 0 or np.std(y_pred) == 0:
            logger.warning(
                "Cannot compute pearson information coefficient: "
                "y_true or y_pred is constant (zero variance)."
            )
            return float("nan")

        corr_matrix = np.corrcoef(
            np.asarray(y_true).ravel(), np.asarray(y_pred).ravel()
        )
        return float(corr_matrix[0, 1])

    @staticmethod
    def daily_rank_ic(
        y_true: pd.Series | np.ndarray,
        y_pred: np.ndarray | pd.Series,
        dates: pd.Series | Sequence[Any],
        min_group_size: int = 5,
    ) -> pd.Series:
        """
        Compute daily cross-sectional rank Information Coefficient (Spearman).

        Groups predictions by observation date, computes Spearman IC for dates with
        at least min_group_size rows and non-constant predictions, and drops NaNs.

        Args:
            y_true: Actual returns.
            y_pred: Predicted returns.
            dates: Sequence of date labels matching y_true and y_pred.
            min_group_size: Minimum observations on a date required to compute IC.

        Returns:
            pd.Series indexed by date containing daily rank IC values.
        """
        ModelEvaluator._validate_inputs(y_true, y_pred)
        if len(y_true) != len(dates):
            raise ValueError(
                f"Length mismatch: y_true ({len(y_true)}) and "
                f"dates ({len(dates)}) must have same length."
            )

        df = pd.DataFrame(
            {
                "y_true": np.asarray(y_true).ravel(),
                "y_pred": np.asarray(y_pred).ravel(),
                "date": pd.Series(dates).values,
            }
        )

        daily_ics: dict[Any, float] = {}
        for d, group in df.groupby("date", sort=True):
            if len(group) < min_group_size:
                continue
            std_true = float(np.std(group["y_true"]))
            std_pred = float(np.std(group["y_pred"]))
            if std_true == 0.0 or std_pred == 0.0:
                continue
            corr, _ = spearmanr(group["y_true"], group["y_pred"])
            if not np.isnan(corr):
                daily_ics[d] = float(corr)

        return pd.Series(daily_ics, dtype=float)

    def extended_metrics(
        self,
        y_true: pd.Series | np.ndarray,
        y_pred: np.ndarray | pd.Series,
        dates: pd.Series | None = None,
    ) -> dict[str, float]:
        """
        Compute extended financial metrics: Pearson IC, Rank IC, and daily IC stats.

        Args:
            y_true: True returns.
            y_pred: Predicted returns.
            dates: Optional date Series for cross-sectional daily IC calculation.

        Returns:
            Dictionary with pearson_ic, rank_ic, and daily IC statistics.
        """
        metrics: dict[str, float] = {
            "pearson_ic": self.pearson_information_coefficient(y_true, y_pred),
            "rank_ic": self.rank_information_coefficient(y_true, y_pred),
        }

        if dates is not None:
            daily_ic = self.daily_rank_ic(y_true, y_pred, dates)
            if len(daily_ic) > 0:
                mean_ic = float(daily_ic.mean())
                std_ic = float(daily_ic.std(ddof=0))
                ic_ir = float(mean_ic / std_ic) if std_ic != 0.0 else float("nan")
                pos_rate = float((daily_ic > 0).mean())

                metrics["mean_daily_rank_ic"] = mean_ic
                metrics["daily_rank_ic_std"] = std_ic
                metrics["ic_ir"] = ic_ir
                metrics["daily_rank_ic_positive_rate"] = pos_rate

        return metrics
