"""Feature distribution drift and model performance degradation detection.

Provides Population Stability Index (PSI) and Kolmogorov-Smirnov (KS)
tests to detect data drift between reference training sets and current
production or inference sets. Also evaluates metric degradation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

from config.settings import settings
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class DriftReport:
    """Contains statistical drift analysis results across feature sets.

    Attributes:
        feature_psi: Dictionary mapping feature name to its calculated PSI score.
        drifted_features: List of feature names where PSI exceeds threshold.
        max_psi: Maximum observed PSI value across evaluated features.
        drift_detected: True if at least one feature exceeded the PSI threshold.
        n_reference: Number of rows in the reference dataset.
        n_current: Number of rows in the current dataset.
    """

    feature_psi: dict[str, float]
    drifted_features: list[str]
    max_psi: float
    drift_detected: bool
    n_reference: int
    n_current: int


class DriftDetector:
    """Detects feature distribution drift and model performance degradation."""

    def __init__(
        self,
        psi_threshold: float | None = None,
        n_bins: int = 10,
        min_rows: int | None = None,
    ) -> None:
        """Initialize DriftDetector.

        Args:
            psi_threshold: PSI threshold above which drift is flagged.
                Defaults to settings.DRIFT_PSI_THRESHOLD.
            n_bins: Number of quantile bins for PSI computation. Default 10.
            min_rows: Minimum row count required to compute drift.
                Defaults to settings.DRIFT_MIN_ROWS.
        """
        self.psi_threshold = (
            psi_threshold if psi_threshold is not None else settings.DRIFT_PSI_THRESHOLD
        )
        self.n_bins = n_bins
        self.min_rows = min_rows if min_rows is not None else settings.DRIFT_MIN_ROWS

    @staticmethod
    def population_stability_index(
        reference: Sequence[float] | np.ndarray | pd.Series,
        current: Sequence[float] | np.ndarray | pd.Series,
        n_bins: int = 10,
    ) -> float:
        """Calculate the Population Stability Index (PSI) between two samples.

        Args:
            reference: Baseline reference distribution sample.
            current: Current distribution sample to test against reference.
            n_bins: Number of quantile bins.

        Returns:
            PSI value >= 0.0.
        """
        ref_arr = np.asarray(reference, dtype=float)
        cur_arr = np.asarray(current, dtype=float)

        ref_clean = ref_arr[~np.isnan(ref_arr)]
        cur_clean = cur_arr[~np.isnan(cur_arr)]

        if len(ref_clean) == 0 and len(cur_clean) == 0:
            return 0.0
        if len(ref_clean) == 0 or len(cur_clean) == 0:
            return float("inf")
        if np.array_equal(ref_clean, cur_clean):
            return 0.0

        quantiles = np.linspace(0, 1, n_bins + 1)
        edges = np.unique(np.quantile(ref_clean, quantiles))

        if len(edges) == 1:
            v = edges[0]
            ref_counts = np.array([np.sum(ref_clean <= v), np.sum(ref_clean > v)])
            cur_counts = np.array([np.sum(cur_clean <= v), np.sum(cur_clean > v)])
        else:
            edges = edges.astype(float)
            edges[0] = -np.inf
            edges[-1] = np.inf
            ref_counts, _ = np.histogram(ref_clean, bins=edges)
            cur_counts, _ = np.histogram(cur_clean, bins=edges)

        p_ref = np.clip(ref_counts / len(ref_clean), 1e-6, 1.0)
        p_cur = np.clip(cur_counts / len(cur_clean), 1e-6, 1.0)

        psi = float(np.sum((p_cur - p_ref) * np.log(p_cur / p_ref)))
        return max(0.0, psi)

    @staticmethod
    def ks_statistic(
        reference: Sequence[float] | np.ndarray | pd.Series,
        current: Sequence[float] | np.ndarray | pd.Series,
    ) -> float:
        """Calculate the Kolmogorov-Smirnov test statistic.

        Args:
            reference: Baseline reference sample.
            current: Current sample to test.

        Returns:
            KS statistic between 0.0 and 1.0.
        """
        ref_arr = np.asarray(reference, dtype=float)
        cur_arr = np.asarray(current, dtype=float)

        ref_clean = ref_arr[~np.isnan(ref_arr)]
        cur_clean = cur_arr[~np.isnan(cur_arr)]

        if len(ref_clean) == 0 or len(cur_clean) == 0:
            return 0.0

        res = ks_2samp(ref_clean, cur_clean)
        return float(res.statistic)

    def detect(
        self,
        reference: pd.DataFrame,
        current: pd.DataFrame,
        feature_columns: Sequence[str],
    ) -> DriftReport:
        """Evaluate drift across multiple features between reference and current frames.

        Args:
            reference: Reference historical dataset (e.g. training set).
            current: Current dataset (e.g. inference set or recent window).
            feature_columns: Names of feature columns to evaluate.

        Returns:
            DriftReport containing PSI scores and drift status.

        Raises:
            ValueError: If either frame is empty or missing specified feature columns.
        """
        if reference.empty:
            raise ValueError("reference cannot be empty.")
        if current.empty:
            raise ValueError("current cannot be empty.")

        missing_ref = set(feature_columns) - set(reference.columns)
        if missing_ref:
            raise ValueError(
                f"reference missing required columns: {sorted(missing_ref)}"
            )

        missing_cur = set(feature_columns) - set(current.columns)
        if missing_cur:
            raise ValueError(f"current missing required columns: {sorted(missing_cur)}")

        n_ref = len(reference)
        n_cur = len(current)

        if n_ref < self.min_rows or n_cur < self.min_rows:
            logger.warning(
                "Data has fewer rows (ref=%s, cur=%s) than min_rows=%s; "
                "skipping drift calculation.",
                n_ref,
                n_cur,
                self.min_rows,
            )
            return DriftReport(
                feature_psi={},
                drifted_features=[],
                max_psi=0.0,
                drift_detected=False,
                n_reference=n_ref,
                n_current=n_cur,
            )

        feature_psi: dict[str, float] = {}
        drifted: list[str] = []

        for col in feature_columns:
            psi = self.population_stability_index(
                reference[col], current[col], n_bins=self.n_bins
            )
            feature_psi[col] = psi
            if psi > self.psi_threshold:
                drifted.append(col)

        max_psi = max(feature_psi.values()) if feature_psi else 0.0
        drift_detected = bool(drifted)

        if drift_detected:
            logger.warning(
                "Feature drift detected in %s features: %s (max PSI=%.4f).",
                len(drifted),
                drifted,
                max_psi,
            )
        else:
            logger.info(
                "No feature drift detected across %s features (max PSI=%.4f).",
                len(feature_columns),
                max_psi,
            )

        return DriftReport(
            feature_psi=feature_psi,
            drifted_features=drifted,
            max_psi=max_psi,
            drift_detected=drift_detected,
            n_reference=n_ref,
            n_current=n_cur,
        )

    @staticmethod
    def performance_degraded(
        baseline: float,
        current: float,
        tolerance: float = 0.0,
        higher_is_better: bool = True,
    ) -> bool:
        """Determine if current model performance is degraded beyond tolerance.

        Args:
            baseline: Baseline performance metric.
            current: Current performance metric.
            tolerance: Margin of acceptable variation.
            higher_is_better: True if larger metric values indicate better performance.

        Returns:
            True if current performance is worse than baseline beyond tolerance.
        """
        if math.isnan(current):
            return True
        if math.isnan(baseline):
            return False

        if higher_is_better:
            return current < (baseline - tolerance)
        return current > (baseline + tolerance)
