"""Unit tests for DriftDetector and DriftReport."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.models.drift_detector import DriftDetector, DriftReport

# ==============================================================================
# PSI & KS Calculation
# ==============================================================================


def test_psi_identical_distributions() -> None:
    """Assert identical distributions yield PSI of exactly 0.0."""
    data = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]
    psi = DriftDetector.population_stability_index(data, data)
    assert psi == pytest.approx(0.0)


def test_psi_golden_calculation() -> None:
    """Assert PSI matches hand-computed two-bin shift."""
    # Ref: 100 values <= 0.5, 100 values > 0.5 -> p_ref = [0.5, 0.5]
    # Cur: 10 values <= 0.5, 190 values > 0.5 -> p_cur = [0.05, 0.95]
    # PSI = (0.05 - 0.5) * ln(0.05 / 0.5) + (0.95 - 0.5) * ln(0.95 / 0.5)
    #     = -0.45 * ln(0.1) + 0.45 * ln(1.9)
    #     = -0.45 * (-2.302585) + 0.45 * 0.641854
    #     = 1.036163 + 0.288834 = 1.324998
    ref = [0.0] * 100 + [1.0] * 100
    cur = [0.0] * 10 + [1.0] * 190
    psi = DriftDetector.population_stability_index(ref, cur, n_bins=2)
    assert psi == pytest.approx(1.324998, rel=1e-3)


def test_psi_constant_value_reference() -> None:
    """Assert reference with single unique value falls back to two bins safely."""
    ref = [5.0] * 50
    cur = [5.0] * 25 + [10.0] * 25
    psi = DriftDetector.population_stability_index(ref, cur)
    assert psi > 0.0


def test_ks_statistic() -> None:
    """Assert KS statistic detects identity and complete separation."""
    s1 = [1.0, 2.0, 3.0, 4.0, 5.0]
    assert DriftDetector.ks_statistic(s1, s1) == pytest.approx(0.0)

    s2 = [10.0, 20.0, 30.0, 40.0, 50.0]
    assert DriftDetector.ks_statistic(s1, s2) == pytest.approx(1.0)


# ==============================================================================
# DriftDetector.detect()
# ==============================================================================


def test_detect_validation() -> None:
    """Assert detect raises ValueError on empty DataFrames or missing columns."""
    detector = DriftDetector(min_rows=10)
    df_valid = pd.DataFrame({"f1": range(20), "f2": range(20)})

    with pytest.raises(ValueError, match="reference cannot be empty."):
        detector.detect(pd.DataFrame(), df_valid, ["f1"])

    with pytest.raises(ValueError, match="current cannot be empty."):
        detector.detect(df_valid, pd.DataFrame(), ["f1"])

    with pytest.raises(
        ValueError, match="reference missing required columns: \\['missing'\\]"
    ):
        detector.detect(df_valid, df_valid, ["f1", "missing"])

    df_with_missing = pd.DataFrame({"f1": range(20), "missing": range(20)})
    with pytest.raises(
        ValueError, match="current missing required columns: \\['missing'\\]"
    ):
        detector.detect(df_with_missing, df_valid, ["f1", "missing"])


def test_detect_below_min_rows() -> None:
    """Assert detect skips calculation when row count is below min_rows."""
    detector = DriftDetector(min_rows=50)
    ref = pd.DataFrame({"f1": range(30)})
    cur = pd.DataFrame({"f1": range(30)})

    report = detector.detect(ref, cur, ["f1"])
    assert not report.drift_detected
    assert report.max_psi == 0.0
    assert report.feature_psi == {}


def test_detect_flags_drift(caplog) -> None:
    """Assert detect flags drifted features exceeding psi_threshold."""
    detector = DriftDetector(psi_threshold=0.25, min_rows=20)
    rng = np.random.default_rng(42)

    ref = pd.DataFrame(
        {
            "stable": rng.normal(0, 1, 200),
            "drifted": rng.normal(0, 1, 200),
        }
    )
    cur = pd.DataFrame(
        {
            "stable": rng.normal(0, 1, 200),
            "drifted": rng.normal(5, 1, 200),  # Mean shift
        }
    )

    report = detector.detect(ref, cur, ["stable", "drifted"])
    assert isinstance(report, DriftReport)
    assert report.drift_detected
    assert "drifted" in report.drifted_features
    assert "stable" not in report.drifted_features
    assert report.max_psi > 0.25


def test_detect_no_mutation() -> None:
    """Assert detect does not mutate input DataFrames."""
    detector = DriftDetector(min_rows=10)
    ref = pd.DataFrame({"f1": [1.0, 2.0, 3.0] * 10})
    cur = pd.DataFrame({"f1": [1.0, 2.0, 3.0] * 10})
    ref_copy = ref.copy()
    cur_copy = cur.copy()

    detector.detect(ref, cur, ["f1"])
    pd.testing.assert_frame_equal(ref, ref_copy)
    pd.testing.assert_frame_equal(cur, cur_copy)


# ==============================================================================
# DriftDetector.performance_degraded()
# ==============================================================================


def test_performance_degraded() -> None:
    """Assert performance degradation logic for higher-is-better and lower-is-better."""
    # Higher is better (e.g. IC, accuracy)
    assert not DriftDetector.performance_degraded(
        0.10, 0.09, tolerance=0.02, higher_is_better=True
    )
    assert DriftDetector.performance_degraded(
        0.10, 0.07, tolerance=0.02, higher_is_better=True
    )

    # Lower is better (e.g. MSE, RMSE)
    assert not DriftDetector.performance_degraded(
        0.05, 0.06, tolerance=0.02, higher_is_better=False
    )
    assert DriftDetector.performance_degraded(
        0.05, 0.08, tolerance=0.02, higher_is_better=False
    )

    # NaN handling
    assert DriftDetector.performance_degraded(0.10, float("nan"))
    assert not DriftDetector.performance_degraded(float("nan"), 0.10)
