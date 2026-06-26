"""
Rank-based signal generation.

Converts predicted returns into BUY/HOLD/SELL signals by ranking
assets cross-sectionally within each date, going long the top-N
predicted performers and optionally short the bottom-N.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class RankSignalGenerator:
    """
    Generates trading signals by ranking predicted returns within
    each date across the entire symbol universe.

    Signal encoding:
        1  = BUY  (rank <= top_n within that date)
        -1 = SELL (rank in the bottom_n within that date)
        0  = HOLD (otherwise)

    Ranking is performed independently per date — this is a
    cross-sectional signal, unlike ThresholdSignalGenerator which
    evaluates each row in isolation.
    """

    def __init__(self, top_n: int = 10, bottom_n: int = 0) -> None:
        """
        Args:
            top_n: Number of highest-ranked symbols per date to mark BUY.
            bottom_n: Number of lowest-ranked symbols per date to mark
                SELL. Defaults to 0 (long-only, no short signals).

        Raises:
            ValueError: If top_n < 1 or bottom_n < 0.
        """
        if top_n < 1:
            raise ValueError(f"top_n must be at least 1, got {top_n}")
        if bottom_n < 0:
            raise ValueError(f"bottom_n must be non-negative, got {bottom_n}")

        self.top_n = top_n
        self.bottom_n = bottom_n

    def generate(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply cross-sectional rank logic to a predictions DataFrame.

        Args:
            predictions_df: DataFrame with at least 'symbol', 'date',
                'predicted_return' columns. May contain multiple dates;
                ranking is performed independently within each date.

        Returns:
            DataFrame with columns: symbol, date, predicted_return,
            signal, confidence, rank. 'rank' is the integer rank
            (1 = highest predicted_return) within that date.
            'confidence' is the absolute predicted_return value.

        Raises:
            ValueError: If predictions_df is empty or missing required columns.
        """
        if predictions_df.empty:
            raise ValueError("predictions_df cannot be empty.")

        required_cols = {"symbol", "date", "predicted_return"}
        missing = required_cols - set(predictions_df.columns)
        if missing:
            raise ValueError(f"predictions_df missing required columns: {missing}")

        df = predictions_df.copy()

        # Rank 1 = highest predicted_return, within each date independently.
        # method="first" breaks ties deterministically by row order.
        df["rank"] = (
            df.groupby("date")["predicted_return"]
            .rank(ascending=False, method="first")
            .astype(int)
        )

        group_sizes = df.groupby("date")["predicted_return"].transform("count")

        conditions = [
            df["rank"] <= self.top_n,
            df["rank"] > (group_sizes - self.bottom_n),
        ]
        choices = [1, -1]
        df["signal"] = np.select(conditions, choices, default=0)
        df["confidence"] = df["predicted_return"].abs()

        logger.info(
            "Generated rank signals for %s rows across %s dates (top_n=%s, bottom_n=%s).",
            len(df),
            df["date"].nunique(),
            self.top_n,
            self.bottom_n,
        )

        return df[["symbol", "date", "predicted_return", "signal", "confidence", "rank"]]