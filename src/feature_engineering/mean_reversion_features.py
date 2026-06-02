"""
Mean reversion feature generation.

Generates mean-reversion based technical indicators used by downstream
machine learning models and quantitative research pipelines.

Features:
    - Z-Score (20)
    - Distance from SMA(20)
    - Distance from EMA(20)

Rules:
    - Uses adj_close if available, otherwise falls back to close
    - Computed independently per symbol
    - No lookahead bias
    - Returns only symbol, date, and generated feature columns
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class MeanReversionFeatureGenerator:
    """
    Generate mean-reversion based technical indicators.

    Input DataFrame requirements:
        symbol
        date
        close

    Optional:
        adj_close

    Output:
        symbol
        date
        zscore_20
        distance_from_sma
        distance_from_ema
    """

    REQUIRED_COLUMNS: tuple[str, ...] = (
        "symbol",
        "date",
        "close",
    )

    FEATURE_COLUMNS: tuple[str, ...] = (
        "zscore_20",
        "distance_from_sma",
        "distance_from_ema",
    )

    LOOKBACK_WINDOW: int = 20

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all mean-reversion features.

        Parameters
        ----------
        df : pd.DataFrame
            Processed market data.

        Returns
        -------
        pd.DataFrame
            DataFrame containing symbol, date, and mean-reversion features.
        """

        self._validate_input(df)

        logger.info(
            "Generating mean-reversion features for %s rows across %s symbols.",
            len(df),
            df["symbol"].nunique(),
        )

        result = df.copy()

        result = result.sort_values(
            ["symbol", "date"]
        ).reset_index(drop=True)

        price_column = self._get_price_column(result)

        logger.info(
            "Using '%s' as source price column.",
            price_column,
        )

        result = self._generate_zscore(
            result,
            price_column,
        )

        result = self._generate_distance_from_sma(
            result,
            price_column,
        )

        result = self._generate_distance_from_ema(
            result,
            price_column,
        )

        logger.info(
            "Successfully generated %s mean-reversion features.",
            len(self.FEATURE_COLUMNS),
        )

        return result[
            [
                "symbol",
                "date",
                *self.FEATURE_COLUMNS,
            ]
        ]

    def _generate_zscore(
        self,
        df: pd.DataFrame,
        price_column: str,
    ) -> pd.DataFrame:
        """
        Generate Z-Score(20).

        Formula:
            zscore_20 =
                (price - SMA(20))
                / rolling_std(20)
        """

        rolling_mean = (
            df.groupby("symbol")[price_column]
            .transform(
                lambda x, w=self.LOOKBACK_WINDOW: x.rolling(
                    window=w,
                    min_periods=w,
                ).mean()
            )
        )

        rolling_std = (
            df.groupby("symbol")[price_column]
            .transform(
                lambda x, w=self.LOOKBACK_WINDOW: x.rolling(
                    window=w,
                    min_periods=w,
                ).std()
            )
        )

        rolling_std = rolling_std.replace(
            0,
            np.nan,
        )

        df["zscore_20"] = (
            df[price_column] - rolling_mean
        ) / rolling_std

        return df

    def _generate_distance_from_sma(
        self,
        df: pd.DataFrame,
        price_column: str,
    ) -> pd.DataFrame:
        """
        Generate percentage distance from SMA(20).

        Formula:
            (price - SMA(20))
            / SMA(20)
        """

        sma_20 = (
            df.groupby("symbol")[price_column]
            .transform(
                lambda x, w=self.LOOKBACK_WINDOW: x.rolling(
                    window=w,
                    min_periods=w,
                ).mean()
            )
        )

        sma_20 = sma_20.replace(
            0,
            np.nan,
        )

        df["distance_from_sma"] = (
            df[price_column] - sma_20
        ) / sma_20

        return df

    def _generate_distance_from_ema(
        self,
        df: pd.DataFrame,
        price_column: str,
    ) -> pd.DataFrame:
        """
        Generate percentage distance from EMA(20).

        Formula:
            (price - EMA(20))
            / EMA(20)
        """

        ema_20 = (
            df.groupby("symbol")[price_column]
            .transform(
                lambda x, w=self.LOOKBACK_WINDOW: x.ewm(
                    span=w,
                    adjust=False,
                ).mean()
            )
        )

        ema_20 = ema_20.replace(
            0,
            np.nan,
        )

        df["distance_from_ema"] = (
            df[price_column] - ema_20
        ) / ema_20

        return df

    def _get_price_column(
        self,
        df: pd.DataFrame,
    ) -> str:
        """
        Select source price column.

        Preference:
            1. adj_close
            2. close
        """

        if (
            "adj_close" in df.columns
            and not df["adj_close"].isna().all()
        ):
            return "adj_close"

        return "close"

    def _validate_input(
        self,
        df: pd.DataFrame,
    ) -> None:
        """
        Validate input DataFrame.
        """

        if df.empty:
            raise ValueError(
                "Input DataFrame is empty."
            )

        missing_columns = [
            column
            for column in self.REQUIRED_COLUMNS
            if column not in df.columns
        ]

        if missing_columns:
            raise ValueError(
                f"Missing required columns: {missing_columns}"
            )

        if df["symbol"].isna().any():
            raise ValueError(
                "Column 'symbol' contains null values."
            )

        if df["date"].isna().any():
            raise ValueError(
                "Column 'date' contains null values."
            )

        if df["close"].isna().any():
            raise ValueError(
                "Column 'close' contains null values."
            )