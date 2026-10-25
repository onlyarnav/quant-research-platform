"""
Tests for split_utils: compute_split_positions and align_timestamp.
"""

from __future__ import annotations

import pandas as pd
import pytest

from src.models.split_utils import align_timestamp, compute_split_positions

# =====================================================================
# compute_split_positions()
# =====================================================================


def test_compute_split_positions_golden_20_dates() -> None:
    """Assert 20 dates with 0.7/0.15 gives train_idx=14, val_idx=17."""
    # 20 * 0.7 = 14, 20 * (0.7 + 0.15) = 17
    pos = compute_split_positions(20, 0.7, 0.15, embargo_dates=0)
    assert pos.train_idx == 14
    assert pos.val_idx == 17
    assert pos.embargo_dates == 0


def test_compute_split_positions_raises_on_insufficient_unique_dates() -> None:
    """Assert fewer than 3 dates raises ValueError with Insufficient unique dates."""
    with pytest.raises(ValueError, match="Insufficient unique dates"):
        compute_split_positions(2, 0.7, 0.15)


def test_compute_split_positions_raises_on_negative_embargo() -> None:
    """Assert negative embargo raises ValueError."""
    with pytest.raises(ValueError, match="embargo_dates must be non-negative"):
        compute_split_positions(20, 0.7, 0.15, embargo_dates=-1)


def test_compute_split_positions_raises_on_colliding_ratios() -> None:
    """Assert ratios where train_idx >= val_idx raise Invalid split."""
    # 10 dates with 0.5 train and 0.0 val -> train_idx=5, val_idx=5
    with pytest.raises(ValueError, match="Invalid split"):
        compute_split_positions(10, 0.5, 0.0)


def test_compute_split_positions_clamp_then_validate_regression() -> None:
    """
    Assert ratios that cause train_idx >= val_idx (e.g. 10 dates with 0.80/0.05)
    raise Invalid split after clamping instead of creating an empty or invalid split.
    """
    with pytest.raises(ValueError, match="Invalid split"):
        compute_split_positions(10, 0.80, 0.05)


def test_compute_split_positions_embargo_too_large() -> None:
    """Assert embargo dates that exceed available split intervals raise error."""
    # 20 dates: train_idx=14, val_idx=17.
    # If embargo_dates=15 -> train_idx - 15 = -1 < 0 -> too large
    with pytest.raises(ValueError, match="too large"):
        compute_split_positions(20, 0.7, 0.15, embargo_dates=15)

    # If embargo_dates=3 -> val_idx - 3 = 14 <= train_idx (14) -> too large
    with pytest.raises(ValueError, match="too large"):
        compute_split_positions(20, 0.7, 0.15, embargo_dates=3)


def test_compute_split_positions_embargo_zero_equals_default() -> None:
    """Assert embargo_dates=0 produces identical split indices to default."""
    p_def = compute_split_positions(30, 0.7, 0.15)
    p_zero = compute_split_positions(30, 0.7, 0.15, embargo_dates=0)
    assert p_def.train_idx == p_zero.train_idx
    assert p_def.val_idx == p_zero.val_idx


# =====================================================================
# align_timestamp()
# =====================================================================


@pytest.mark.parametrize(
    "series_tz, ts_tz, expected_tz",
    [
        ("Asia/Kolkata", None, "Asia/Kolkata"),  # aware series + naive ts -> aware ts
        (None, "Asia/Kolkata", None),  # naive series + aware ts -> naive ts
        ("Asia/Kolkata", "Asia/Kolkata", "Asia/Kolkata"),  # matching aware -> unchanged
        (None, None, None),  # matching naive -> unchanged
    ],
)
def test_align_timestamp(
    series_tz: str | None, ts_tz: str | None, expected_tz: str | None
) -> None:
    """Assert align_timestamp aligns timezones correctly across aware and naive combinations."""
    dt_series = pd.Series(
        pd.date_range("2023-01-01", periods=5, freq="D", tz=series_tz)
    )
    base_ts = pd.Timestamp("2023-01-03", tz=ts_tz)

    aligned = align_timestamp(base_ts, dt_series)

    if expected_tz is None:
        assert aligned.tz is None
    else:
        assert str(aligned.tz) == expected_tz

    # Assert comparison with Series never raises TypeError
    comp = dt_series > aligned
    assert isinstance(comp, pd.Series)
