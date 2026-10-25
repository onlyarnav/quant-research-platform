"""Market regime classification protocol and volatility-based classifier.

Provides regime classification interfaces and a leakage-safe volatility
regime classifier using expanding point-in-time medians.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


@runtime_checkable
class RegimeClassifier(Protocol):
    """Protocol defining the interface for market regime classification models."""

    def classify(self, features_df: pd.DataFrame) -> pd.Series:
        """Classify dates into discrete market regimes.

        Args:
            features_df: Feature DataFrame containing dates and indicator columns.

        Returns:
            Pandas Series with date index and regime label strings.
        """
        ...


class VolatilityRegimeClassifier:
    """Classifies market dates into high/low volatility regimes without lookahead.

    Calculates daily cross-sectional average volatility, and compares it to the
    expanding median of strictly previous dates (shifted by 1 period).
    Dates with fewer previous observations than min_periods are labeled 'unknown'.
    """

    def __init__(
        self,
        volatility_column: str = "volatility_20",
        min_periods: int = 60,
    ) -> None:
        """Initialize VolatilityRegimeClassifier.

        Args:
            volatility_column: Column name in features_df to measure volatility.
                Default 'volatility_20'.
            min_periods: Minimum previous dates required before classifying.
                Default 60.

        Raises:
            ValueError: If min_periods < 1.
        """
        if min_periods < 1:
            raise ValueError(f"min_periods must be at least 1, got {min_periods}.")

        self.volatility_column = volatility_column
        self.min_periods = min_periods

    def classify(self, features_df: pd.DataFrame) -> pd.Series:
        """Classify each date into 'unknown', 'high_vol', or 'low_vol'.

        Args:
            features_df: DataFrame containing 'date' and volatility column.

        Returns:
            Series indexed by date with regime label string values.

        Raises:
            ValueError: If features_df is empty or missing required columns.
        """
        if features_df.empty:
            raise ValueError("features_df cannot be empty.")

        required = {"date", self.volatility_column}
        missing = required - set(features_df.columns)
        if missing:
            raise ValueError(f"features_df missing required columns: {sorted(missing)}")

        daily_vol = (
            features_df.groupby("date")[self.volatility_column].mean().sort_index()
        )

        # Expanding median of strictly past dates (shifted by 1 to prevent lookahead)
        exp_median = daily_vol.shift(1).expanding(min_periods=self.min_periods).median()

        regimes: list[str] = []
        for vol_val, med_val in zip(daily_vol, exp_median, strict=False):
            if pd.isna(med_val):
                regimes.append("unknown")
            elif vol_val > med_val:
                regimes.append("high_vol")
            else:
                regimes.append("low_vol")

        result = pd.Series(regimes, index=daily_vol.index, name="regime")
        logger.info(
            "Classified %s dates: high_vol=%s, low_vol=%s, unknown=%s",
            len(result),
            (result == "high_vol").sum(),
            (result == "low_vol").sum(),
            (result == "unknown").sum(),
        )
        return result
