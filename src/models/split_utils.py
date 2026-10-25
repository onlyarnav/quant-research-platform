"""
Date split calculation and timestamp alignment utilities for financial time-series.

Provides deterministic position calculations for chronological train/validation/test
splits with optional embargo purging to prevent label leakage between contiguous splits.
Also aligns timestamps so comparisons never fail across tz-aware and naive dates.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True, slots=True)
class SplitPositions:
    """
    Cutoff positions for chronological dataset splitting.

    Attributes:
        train_idx: Index of the last training date before embargo.
        val_idx: Index of the last validation date before embargo.
        embargo_dates: Number of dates purged at split boundaries.
    """

    train_idx: int
    val_idx: int
    embargo_dates: int


def compute_split_positions(
    n_dates: int,
    train_ratio: float,
    val_ratio: float,
    embargo_dates: int = 0,
) -> SplitPositions:
    """
    Compute chronological split indices with boundary clamping and embargo validation.

    Clamping happens strictly before index validation to prevent degenerate
    splits where the validation or test sets would be empty.

    Args:
        n_dates: Total number of unique chronological dates.
        train_ratio: Proportion of dates allocated to training (e.g. 0.70).
        val_ratio: Proportion of dates allocated to validation (e.g. 0.15).
        embargo_dates: Number of dates to purge prior to validation and test boundaries
            to eliminate forward-looking label overlap.

    Returns:
        SplitPositions instance containing clamped cutoff indices.

    Raises:
        ValueError: If dates < 3, embargo < 0, or embargo is too large.
    """
    if n_dates < 3:
        raise ValueError("Insufficient unique dates for train/val/test split.")

    if embargo_dates < 0:
        raise ValueError(f"embargo_dates must be non-negative, got {embargo_dates}.")

    train_idx = int(n_dates * train_ratio)
    val_idx = int(n_dates * (train_ratio + val_ratio))

    # Clamp before validate
    train_idx = min(train_idx, n_dates - 2)
    val_idx = min(val_idx, n_dates - 1)

    if train_idx >= val_idx:
        raise ValueError(
            f"Invalid split: train_cutoff_idx={train_idx} >= val_cutoff_idx={val_idx}. "
            "Adjust TRAIN_SPLIT_RATIO/VAL_SPLIT_RATIO or provide more data."
        )

    if (train_idx - embargo_dates < 0) or (val_idx - embargo_dates <= train_idx):
        raise ValueError(
            f"Embargo of {embargo_dates} dates is too large for the available history."
        )

    return SplitPositions(
        train_idx=train_idx,
        val_idx=val_idx,
        embargo_dates=embargo_dates,
    )


def align_timestamp(ts: pd.Timestamp, like: pd.Series) -> pd.Timestamp:
    """
    Align the timezone of a scalar Timestamp to match the timezone of a target Series.

    Ensures date comparisons between scalar thresholds and DataFrame date columns
    never raise TypeError: Cannot compare tz-naive and tz-aware timestamps.

    Args:
        ts: Timestamp to align.
        like: Datetime Series whose timezone defines the target timezone.

    Returns:
        Aligned pd.Timestamp.
    """
    if not isinstance(ts, pd.Timestamp):
        ts = pd.Timestamp(ts)

    series_dt = like.dt if hasattr(like, "dt") else None
    series_tz = (
        series_dt.tz if series_dt is not None else getattr(like.dtype, "tz", None)
    )

    if series_tz is not None and ts.tz is None:
        return ts.tz_localize(series_tz)
    elif series_tz is None and ts.tz is not None:
        return ts.tz_convert(None)
    elif series_tz is not None and ts.tz is not None and ts.tz != series_tz:
        return ts.tz_convert(series_tz)

    return ts
