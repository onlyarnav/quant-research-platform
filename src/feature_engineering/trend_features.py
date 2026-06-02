"""
Trend feature generation.

Generates trend-following technical indicators used by downstream
machine learning models and quantitative research pipelines.

Features:
    - SMA: 5, 10, 20, 50, 200
    - EMA: 10, 20, 50

Rules:
    - Uses adj_close if available, otherwise falls back to close
    - Computed independently per symbol
    - No lookahead bias (only historical observations are used)
    - Returns only symbol, date, and generated feature columns
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class TrendFeatureGenerator:
    """
    Generate trend-based technical indicators.

    Input DataFrame requirements:
        symbol
        date
        close

    Optional:
        adj_close

    Output:
        symbol
        date
        sma_5
        sma_10
        sma_20
        sma_50
        sma_200
        ema_10
        ema_20
        ema_50
    """

    SMA_WINDOWS: tuple[int, ...] = (
        5,
        10,
        20,
        50,
        200,
    )

    EMA_WINDOWS: tuple[int, ...] = (
        10,
        20,
        50,
    )

    REQUIRED_COLUMNS: tuple[str, ...] = (
        "symbol",
        "date",
        "close",
    )

    FEATURE_COLUMNS: tuple[str, ...] = (
        "sma_5",
        "sma_10",
        "sma_20",
        "sma_50",
        "sma_200",
        "ema_10",
        "ema_20",
        "ema_50",
    )

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all trend features.

        Parameters
        ----------
        df : pd.DataFrame
            Processed market data.

        Returns
        -------
        pd.DataFrame
            DataFrame containing symbol, date, and trend features.
        """

        self._validate_input(df)

        logger.info(
            "Generating trend features for %s rows across %s symbols.",
            len(df),
            df["symbol"].nunique(),
        )

        result = df.copy()

        result = result.sort_values(
            ["symbol", "date"]
        ).reset_index(drop=True)

        price_column = self._get_price_column(result)

        logger.info(
            "Using '%s' as source price column for trend features.",
            price_column,
        )

        result = self._generate_sma_features(
            df=result,
            price_column=price_column,
        )

        result = self._generate_ema_features(
            df=result,
            price_column=price_column,
        )

        output_columns = [
            "symbol",
            "date",
            *self.FEATURE_COLUMNS,
        ]

        logger.info(
            "Successfully generated %s trend features.",
            len(self.FEATURE_COLUMNS),
        )

        return result[output_columns]

    def _generate_sma_features(
        self,
        df: pd.DataFrame,
        price_column: str,
    ) -> pd.DataFrame:
        """
        Generate Simple Moving Average features.
        """

        for window in self.SMA_WINDOWS:
            feature_name = f"sma_{window}"

            df[feature_name] = (
                df.groupby("symbol")[price_column]
                .transform(
                    lambda x, w=window: x.rolling(
                        window=w,
                        min_periods=w,
                    ).mean()
                )
            )

        return df

    def _generate_ema_features(
        self,
        df: pd.DataFrame,
        price_column: str,
    ) -> pd.DataFrame:
        """
        Generate Exponential Moving Average features.
        """

        for window in self.EMA_WINDOWS:
            feature_name = f"ema_{window}"

            df[feature_name] = (
                df.groupby("symbol")[price_column]
                .transform(
                    lambda x, w=window: x.ewm(
                        span=w,
                        adjust=False,
                    ).mean()
                )
            )

        return df

    def _get_price_column(
        self,
        df: pd.DataFrame,
    ) -> str:
        """
        Select price column.

        Preference order:
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