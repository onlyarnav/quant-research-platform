"""
Target generation for supervised machine learning.

WARNING
-------
This module intentionally creates FORWARD-LOOKING labels using future prices.

Targets are generated using:

    future_return_n =
        (price[t + n] / price[t]) - 1

These targets contain information from the future and MUST NEVER be
used as model features.

Before live inference or production prediction:

    - Drop all target columns
    - Generate only feature columns
    - Never compute targets on live market data

This module is strictly for model training, validation,
and backtesting workflows.
"""

from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


class TargetGenerator:
    """
    Generate supervised learning targets.

    Input DataFrame requirements:
        symbol
        date
        close

    Optional:
        adj_close

    Output:
        symbol
        date
        future_return_1d
        future_return_5d
        future_return_10d
    """

    REQUIRED_COLUMNS: tuple[str, ...] = (
        "symbol",
        "date",
        "close",
    )

    TARGET_HORIZONS: tuple[int, ...] = (
        1,
        5,
        10,
    )

    TARGET_COLUMNS: tuple[str, ...] = (
        "future_return_1d",
        "future_return_5d",
        "future_return_10d",
    )

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate forward return targets.

        Parameters
        ----------
        df : pd.DataFrame
            Processed market data.

        Returns
        -------
        pd.DataFrame
            DataFrame containing symbol, date,
            and forward return targets.

        Notes
        -----
        NaN values at the end of each symbol series are preserved.
        The downstream ML pipeline is responsible for removing
        rows that cannot be used for training.
        """

        self._validate_input(df)

        logger.info(
            "Generating targets for %s rows across %s symbols.",
            len(df),
            df["symbol"].nunique(),
        )

        result = df.copy()

        result = result.sort_values(
            ["symbol", "date"]
        ).reset_index(drop=True)

        price_column = self._get_price_column(result)

        logger.info(
            "Using '%s' as source price column for targets.",
            price_column,
        )

        result = self._generate_forward_returns(
            result,
            price_column,
        )

        logger.info(
            "Successfully generated %s target columns.",
            len(self.TARGET_COLUMNS),
        )

        return result[
            [
                "symbol",
                "date",
                *self.TARGET_COLUMNS,
            ]
        ]

    def _generate_forward_returns(
        self,
        df: pd.DataFrame,
        price_column: str,
    ) -> pd.DataFrame:
        """
        Generate forward return targets.

        Formula:
            future_return_n =
                close(t+n) / close(t) - 1

        Uses shift(-n) because targets intentionally
        look into the future.
        """

        for horizon in self.TARGET_HORIZONS:
            column_name = f"future_return_{horizon}d"

            future_price = (
                df.groupby("symbol")[price_column]
                .transform(
                    lambda x, h=horizon: x.shift(-h)
                )
            )

            df[column_name] = (
                future_price / df[price_column]
            ) - 1.0

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