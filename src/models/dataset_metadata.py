"""
Metadata definitions, dataset summaries, and cryptographic versioning for ML datasets.

Provides immutable data structures describing dataset generation parameters,
split boundaries, and deterministic content hashing (SHA-256) for lineage tracking.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from typing import Any, Sequence

import pandas as pd


@dataclass(frozen=True, slots=True)
class SplitSummary:
    """
    Summary metrics and date boundaries for a dataset split.

    Attributes:
        name: Split identifier ('train', 'val', or 'test').
        n_rows: Number of observations in the split.
        n_symbols: Number of distinct asset symbols present.
        start_date: Earliest observation date in ISO-8601 format, or None if empty.
        end_date: Latest observation date in ISO-8601 format, or None if empty.
    """

    name: str
    n_rows: int
    n_symbols: int
    start_date: str | None
    end_date: str | None


@dataclass(frozen=True, slots=True)
class DatasetMetadata:
    """
    Complete provenance and configuration metadata for an ML dataset.

    Attributes:
        horizon: Prediction horizon in trading days (e.g. 1, 5, 10).
        target_column: Exact target column name.
        feature_columns: Ordered tuple of feature column names used.
        feature_set_version: Version tag of the feature pipeline (e.g. 'v1').
        embargo_dates: Number of purge dates applied at split cutoffs.
        lookback_dates: Number of rolling lookback dates retained, or None if all.
        splits: Tuple of SplitSummary instances for train, val, and test.
        dataset_version: 12-character hex SHA-256 digest identifying the dataset.
    """

    horizon: int
    target_column: str
    feature_columns: tuple[str, ...]
    feature_set_version: str
    embargo_dates: int
    lookback_dates: int | None
    splits: tuple[SplitSummary, ...]
    dataset_version: str

    def to_dict(self) -> dict[str, Any]:
        """
        Convert metadata to a JSON-serializable dictionary.

        Returns:
            Dictionary with native Python types suitable for json.dumps.
        """
        data = asdict(self)
        data["feature_columns"] = list(self.feature_columns)
        data["splits"] = [asdict(s) for s in self.splits]
        return data

    def split(self, name: str) -> SplitSummary:
        """
        Retrieve a SplitSummary by split name.

        Args:
            name: Split name ('train', 'val', or 'test').

        Returns:
            Matching SplitSummary.

        Raises:
            KeyError: If no split with the given name exists.
        """
        for s in self.splits:
            if s.name == name:
                return s
        raise KeyError(f"Split '{name}' not found in metadata.")


def compute_data_checksum(parts: Sequence[pd.DataFrame | pd.Series]) -> str:
    """
    Compute a deterministic SHA-256 checksum over one or more DataFrame/Series parts.

    Uses pandas object hashing across rows without index dependency, summed modulo 2^64.

    Args:
        parts: Sequence of DataFrames or Series to hash.

    Returns:
        Hexadecimal SHA-256 digest string.
    """
    hasher = hashlib.sha256()
    for part in parts:
        if part is None or len(part) == 0:
            part_sum = 0
        else:
            hashed = pd.util.hash_pandas_object(part, index=False)
            part_sum = int(hashed.sum()) % (2**64)
        hasher.update(str(part_sum).encode("utf-8"))
    return hasher.hexdigest()


def compute_dataset_version(
    horizon: int,
    feature_columns: Sequence[str],
    splits: Sequence[SplitSummary],
    data_checksum: str,
) -> str:
    """
    Compute a 12-character canonical hex digest representing a dataset version.

    Args:
        horizon: Prediction horizon in days.
        feature_columns: Sequence of feature column names.
        splits: Sequence of SplitSummary objects.
        data_checksum: SHA-256 data checksum from compute_data_checksum.

    Returns:
        First 12 hex characters of the SHA-256 digest.
    """
    payload = {
        "horizon": horizon,
        "feature_columns": list(feature_columns),
        "splits": [
            {
                "name": s.name,
                "n_rows": s.n_rows,
                "start": s.start_date,
                "end": s.end_date,
            }
            for s in splits
        ],
        "data_checksum": data_checksum,
    }
    canonical_json = json.dumps(payload, sort_keys=True)
    digest = hashlib.sha256(canonical_json.encode("utf-8")).hexdigest()
    return digest[:12]


def build_split_summary(name: str, split_df: pd.DataFrame) -> SplitSummary:
    """
    Build a SplitSummary instance from a split DataFrame.

    Args:
        name: Split name ('train', 'val', 'test').
        split_df: Sliced DataFrame for this split.

    Returns:
        Populated SplitSummary dataclass.
    """
    if split_df.empty:
        return SplitSummary(
            name=name,
            n_rows=0,
            n_symbols=0,
            start_date=None,
            end_date=None,
        )
    return SplitSummary(
        name=name,
        n_rows=len(split_df),
        n_symbols=int(split_df["symbol"].nunique()),
        start_date=pd.Timestamp(split_df["date"].min()).isoformat(),
        end_date=pd.Timestamp(split_df["date"].max()).isoformat(),
    )
