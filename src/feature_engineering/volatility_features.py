"""
Volatility feature generation.

Generates volatility-based technical indicators used by downstream
machine learning models and quantitative research pipelines.

Features:
    - Volatility (10, 20, 50)
    - ATR (14)
    - Bollinger Width (20)

Rules:
    - Uses adj_close if available, otherwise falls back to close
    - ATR requires high, low, close
    - Computed independently per symbol
    - No lookahead bias
    - Returns only symbol, date, and generated feature columns
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class VolatilityFeatureGenerator:
    """
    Generate volatility-based technical indicators.

    Input DataFrame requirements:
        symbol
        date
        high
        low
        close

    Optional:
        adj_close

    Output:
        symbol
        date
        volatility_10
        volatility_20
        volatility_50
        atr_14
        bollinger_width
    """

    REQUIRED_COLUMNS: tuple[str, ...] = (
        "symbol",
        "date",
        "high",
        "low",
        "close",
    )

    FEATURE_COLUMNS: tuple[str, ...] = (
        "volatility_10",
        "volatility_20",
        "volatility_50",
        "atr_14",
        "bollinger_width",
    )

    TRADING_DAYS_PER_YEAR: int = 252

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all volatility features.

        Parameters
        ----------
        df : pd.DataFrame
            Processed market data.

        Returns
        -------
        pd.DataFrame
            DataFrame containing symbol, date, and volatility features.
        """

        self._validate_input(df)

        logger.info(
            "Generating volatility features for %s rows across %s symbols.",
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

        result = self._generate_rolling_volatility(
            result,
            price_column,
        )

        result = self._generate_atr(result)

        result = self._generate_bollinger_width(
            result,
            price_column,
        )

        logger.info(
            "Successfully generated %s volatility features.",
            len(self.FEATURE_COLUMNS),
        )

        return result[
            [
                "symbol",
                "date",
                *self.FEATURE_COLUMNS,
            ]
        ]

    def _generate_rolling_volatility(
        self,
        df: pd.DataFrame,
        price_column: str,
    ) -> pd.DataFrame:
        """
        Generate annualized rolling volatility features.

        Formula:
            volatility = rolling_std(log_returns) * sqrt(252)
        """

        log_returns = (
            df.groupby("symbol")[price_column]
            .transform(
                lambda x: np.log(
                    x / x.shift(1)
                )
            )
        )

        for window in (10, 20, 50):
            df[f"volatility_{window}"] = (
                log_returns.groupby(df["symbol"])
                .transform(
                    lambda x, w=window: x.rolling(
                        window=w,
                        min_periods=w,
                    ).std()
                )
                * np.sqrt(self.TRADING_DAYS_PER_YEAR)
            )

        return df

    def _generate_atr(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate ATR(14) using Wilder smoothing.

        True Range:
            max(
                high - low,
                abs(high - previous_close),
                abs(low - previous_close)
            )

        ATR:
            Wilder EMA of True Range
        """

        def calculate_atr(
            group: pd.DataFrame,
        ) -> pd.Series:
            previous_close = group["close"].shift(1)

            tr_1 = group["high"] - group["low"]

            tr_2 = (
                group["high"] - previous_close
            ).abs()

            tr_3 = (
                group["low"] - previous_close
            ).abs()

            true_range = pd.concat(
                [tr_1, tr_2, tr_3],
                axis=1,
            ).max(axis=1)

            atr = true_range.ewm(
                alpha=1 / 14,
                adjust=False,
                min_periods=14,
            ).mean()

            return atr

        df["atr_14"] = (
            df.groupby(
                "symbol",
                group_keys=False,
            )
            .apply(calculate_atr)
            .reset_index(drop=True)
        )

        return df

    def _generate_bollinger_width(
        self,
        df: pd.DataFrame,
        price_column: str,
    ) -> pd.DataFrame:
        """
        Generate Bollinger Band Width.

        Formula:
            upper_band = SMA(20) + 2 * std(20)
            lower_band = SMA(20) - 2 * std(20)

            width =
                (upper_band - lower_band)
                / middle_band
        """

        def calculate_width(
            prices: pd.Series,
        ) -> pd.Series:
            middle_band = prices.rolling(
                window=20,
                min_periods=20,
            ).mean()

            rolling_std = prices.rolling(
                window=20,
                min_periods=20,
            ).std()

            upper_band = (
                middle_band
                + (2 * rolling_std)
            )

            lower_band = (
                middle_band
                - (2 * rolling_std)
            )

            width = (
                upper_band - lower_band
            ) / middle_band

            return width

        df["bollinger_width"] = (
            df.groupby("symbol")[price_column]
            .transform(calculate_width)
        )

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

        for column in self.REQUIRED_COLUMNS:
            if df[column].isna().any():
                raise ValueError(
                    f"Column '{column}' contains null values."
                )