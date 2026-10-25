"""Top-N Long/Short cross-sectional signal generator.

Converts predicted asset returns into simultaneous long (+1) and short (-1)
positions per date, capping positions to half the daily universe size to
strictly prevent long/short portfolio overlap.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class TopNLongShortSignalGenerator:
    """Generates cross-sectional top-N long and bottom-N short trading signals.

    For each date independently:
    - Symbols are ranked descending by predicted return (rank 1 = highest return).
    - Long allocation is capped at min(top_n, size // 2).
    - Short allocation is capped at min(bottom_n, size // 2).
    - Any symbol ranked between the long cutoff and short cutoff is assigned HOLD (0).
    - Single-symbol dates receive HOLD (0).
    """

    def __init__(self, top_n: int = 10, bottom_n: int | None = None) -> None:
        """Initialize TopNLongShortSignalGenerator.

        Args:
            top_n: Number of highest-ranked symbols per date to mark long (+1).
            bottom_n: Number of lowest-ranked symbols per date to mark short (-1).
                Defaults to top_n if None.

        Raises:
            ValueError: If top_n < 1 or bottom_n < 1.
        """
        resolved_bottom_n = top_n if bottom_n is None else bottom_n

        if top_n < 1:
            raise ValueError(f"top_n must be at least 1, got {top_n}")
        if resolved_bottom_n < 1:
            raise ValueError(f"bottom_n must be at least 1, got {resolved_bottom_n}")

        self.top_n = top_n
        self.bottom_n = resolved_bottom_n

    def generate(self, predictions_df: pd.DataFrame) -> pd.DataFrame:
        """Generate long/short signals from cross-sectional predictions.

        Args:
            predictions_df: DataFrame with symbol, date, and predicted_return.

        Returns:
            DataFrame with symbol, date, predicted_return, signal, confidence, rank.

        Raises:
            ValueError: If predictions_df is empty or missing required columns.
        """
        if predictions_df.empty:
            raise ValueError("predictions_df cannot be empty.")

        required_cols = {"symbol", "date", "predicted_return"}
        missing = required_cols - set(predictions_df.columns)
        if missing:
            raise ValueError(
                f"predictions_df missing required columns: {sorted(missing)}"
            )

        df = predictions_df.copy()
        date_key = (
            pd.to_datetime(df["date"], utc=True)
            if df["date"].dtype == "object"
            else df["date"]
        )

        df["rank"] = (
            df.groupby(date_key)["predicted_return"]
            .rank(ascending=False, method="first")
            .astype(int)
        )

        group_sizes = df.groupby(date_key)["predicted_return"].transform("count")

        k_long = np.minimum(self.top_n, group_sizes // 2)
        k_short = np.minimum(self.bottom_n, group_sizes // 2)

        conditions = [
            (df["rank"] <= k_long) & (k_long > 0),
            (df["rank"] > (group_sizes - k_short)) & (k_short > 0),
        ]
        choices = [1, -1]
        df["signal"] = np.select(conditions, choices, default=0)
        df["confidence"] = df["predicted_return"].abs()

        logger.info(
            "Generated top-N signals for %s rows (%s dates, top_n=%s, bottom_n=%s).",
            len(df),
            df["date"].nunique(),
            self.top_n,
            self.bottom_n,
        )

        return df[
            ["symbol", "date", "predicted_return", "signal", "confidence", "rank"]
        ]
