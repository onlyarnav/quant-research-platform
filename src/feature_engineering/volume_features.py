"""
Volume feature generation.

Generates volume-based technical indicators used by downstream
machine learning models and quantitative research pipelines.

Features:
    - Volume Moving Average (20)
    - Volume Ratio
    - On Balance Volume (OBV)

Rules:
    - Computed independently per symbol
    - No lookahead bias
    - Returns only symbol, date, and generated feature columns
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class VolumeFeatureGenerator:
    """
    Generate volume-based technical indicators.

    Input DataFrame requirements:
        symbol
        date
        close
        volume

    Output:
        symbol
        date
        volume_ma_20
        volume_ratio
        obv
    """

    REQUIRED_COLUMNS: tuple[str, ...] = (
        "symbol",
        "date",
        "close",
        "volume",
    )

    FEATURE_COLUMNS: tuple[str, ...] = (
        "volume_ma_20",
        "volume_ratio",
        "obv",
    )

    VOLUME_WINDOW: int = 20

    def generate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Generate all volume features.

        Parameters
        ----------
        df : pd.DataFrame
            Processed market data.

        Returns
        -------
        pd.DataFrame
            DataFrame containing symbol, date, and volume features.
        """

        self._validate_input(df)

        logger.info(
            "Generating volume features for %s rows across %s symbols.",
            len(df),
            df["symbol"].nunique(),
        )

        result = df.copy()

        result = result.sort_values(
            ["symbol", "date"]
        ).reset_index(drop=True)

        result = self._generate_volume_ma(result)
        result = self._generate_volume_ratio(result)
        result = self._generate_obv(result)

        logger.info(
            "Successfully generated %s volume features.",
            len(self.FEATURE_COLUMNS),
        )

        return result[
            [
                "symbol",
                "date",
                *self.FEATURE_COLUMNS,
            ]
        ]

    def _generate_volume_ma(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate Volume Moving Average (20).

        Formula:
            volume_ma_20 = SMA(20) of volume
        """

        df["volume_ma_20"] = (
            df.groupby("symbol")["volume"]
            .transform(
                lambda x, w=self.VOLUME_WINDOW: x.rolling(
                    window=w,
                    min_periods=w,
                ).mean()
            )
        )

        return df

    def _generate_volume_ratio(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate Volume Ratio.

        Formula:
            volume_ratio = volume / volume_ma_20
        """

        volume_ma = df["volume_ma_20"].replace(
            0,
            np.nan,
        )

        df["volume_ratio"] = (
            df["volume"] / volume_ma
        )

        return df

    def _generate_obv(
        self,
        df: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Generate On Balance Volume (OBV).

        Rules:
            close > prev_close  -> +volume
            close < prev_close  -> -volume
            close == prev_close -> 0

        OBV:
            cumulative sum of signed volume
        """

        def calculate_obv(
            group: pd.DataFrame,
        ) -> pd.Series:
            close_diff = group["close"].diff()

            signed_volume = np.where(
                close_diff > 0,
                group["volume"],
                np.where(
                    close_diff < 0,
                    -group["volume"],
                    0,
                ),
            )

            return pd.Series(
                signed_volume,
                index=group.index,
            ).cumsum()

        df["obv"] = (
            df.groupby(
                "symbol",
                group_keys=False,
            )
            .apply(calculate_obv)
        )

        return df

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

        if df["volume"].isna().any():
            raise ValueError(
                "Column 'volume' contains null values."
            )