"""
Feature schema definitions and validation helpers for the platform.

Defines canonical identifier, target, and technical feature column tuples,
ensuring strict adherence to Data_Schema.md across feature generation,
dataset building, and model training.

Constants:
    IDENTIFIER_COLUMNS: Primary asset identifiers ('symbol', 'date').
    TARGET_COLUMNS: Supported forward-return prediction horizons.
    FEATURE_COLUMNS: The 26 canonical technical indicator columns in exact schema order.
"""

from __future__ import annotations

from typing import Iterable

import pandas as pd

IDENTIFIER_COLUMNS: tuple[str, ...] = ("symbol", "date")

TARGET_COLUMNS: tuple[str, ...] = (
    "future_return_1d",
    "future_return_5d",
    "future_return_10d",
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
    "rsi_14",
    "macd",
    "macd_signal",
    "roc_5",
    "roc_10",
    "momentum_5",
    "momentum_10",
    "volatility_10",
    "volatility_20",
    "volatility_50",
    "atr_14",
    "bollinger_width",
    "zscore_20",
    "distance_from_sma",
    "distance_from_ema",
    "volume_ma_20",
    "volume_ratio",
    "obv",
)


def get_feature_columns(columns: Iterable[str]) -> list[str]:
    """
    Extract feature column names from an iterable of column names.

    Filters out identifier columns ('symbol', 'date') and target columns
    ('future_return_1d', etc.), preserving the original input order.

    Args:
        columns: Iterable of column names to inspect.

    Returns:
        List of column names that represent features.
    """
    excluded = set(IDENTIFIER_COLUMNS) | set(TARGET_COLUMNS)
    return [col for col in columns if col not in excluded]


def validate_feature_frame(
    df: pd.DataFrame,
    require_targets: bool = True,
) -> None:
    """
    Validate that a DataFrame conforms to the canonical feature dataset schema.

    Checks that the DataFrame is not empty and contains all required identifiers,
    the 26 canonical technical features, and optionally the forward-return targets.

    Args:
        df: DataFrame to validate.
        require_targets: Whether target columns must also be present.

    Raises:
        ValueError: If df is empty, or if any required columns are missing.
    """
    if df.empty:
        raise ValueError("Feature frame cannot be empty.")

    required = set(IDENTIFIER_COLUMNS) | set(FEATURE_COLUMNS)
    if require_targets:
        required |= set(TARGET_COLUMNS)

    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"Feature frame missing required columns: {missing}")
