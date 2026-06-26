"""
Threshold-based signal generation.

Converts predicted returns into BUY/HOLD/SELL signals by comparing
each prediction against a fixed threshold, independently per row.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class ThresholdSignalGenerator:
    """
    Generates trading signals by comparing predicted returns against
    a fixed threshold.

    Signal encoding:
        1  = BUY  (predicted_return > threshold)
        -1 = SELL (predicted_return < -threshold)
        0  = HOLD (otherwise)
    """

    def __init__(self, threshold: float = 0.01) -> None:
        """
        Args:
            threshold: Minimum absolute predicted return required to
                trigger a BUY or SELL signal. Predictions within
                [-threshold, threshold] are classified HOLD.

        Raises:
            ValueError: If threshold is negative.
        """
        if threshold < 0:
            raise ValueError(f"threshold must be non-negative, got {threshold}")

        self.threshold = threshold

    def generate(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply threshold logic to a predictions DataFrame.

        Args:
            predictions_df: DataFrame with at least 'symbol', 'date',
                'predicted_return' columns.

        Returns:
            DataFrame with columns: symbol, date, predicted_return,
            signal, confidence, rank. 'rank' is always None for
            threshold-based signals (not applicable). 'confidence'
            is the absolute predicted_return value.

        Raises:
            ValueError: If predictions_df is empty or missing required columns.
        """
        if predictions_df.empty:
            raise ValueError("predictions_df cannot be empty.")

        required_cols = {"symbol", "date", "predicted_return"}
        missing = required_cols - set(predictions_df.columns)
        if missing:
            raise ValueError(f"predictions_df missing required columns: {missing}")

        conditions = [
            predictions_df["predicted_return"] > self.threshold,
            predictions_df["predicted_return"] < -self.threshold,
        ]
        choices = [1, -1]
        signal = np.select(conditions, choices, default=0)

        output_df = predictions_df[["symbol", "date", "predicted_return"]].copy()
        output_df["signal"] = signal
        output_df["confidence"] = predictions_df["predicted_return"].abs()
        output_df["rank"] = None

        logger.info(
            "Generated threshold signals for %s rows (threshold=%s).",
            len(output_df),
            self.threshold,
        )

        return output_df