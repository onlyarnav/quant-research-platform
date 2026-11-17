"""
Data models and serialization helpers for backtest storage.
"""

from __future__ import annotations

import json
import math
import os
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _sanitize_for_json(obj: Any) -> Any:
    """Recursively convert float NaN/inf to None and Timestamps to ISO strings."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, (np.floating, np.integer)):
        val = obj.item()
        if isinstance(val, float) and (math.isnan(val) or math.isinf(val)):
            return None
        return val
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, datetime):
        return obj.isoformat()
    if isinstance(obj, dict):
        return {str(k): _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def _atomic_write_json(file_path: Path, payload: Any) -> None:
    """Write JSON data to a file atomically via a temporary file."""
    sanitized = _sanitize_for_json(payload)
    parent = file_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        "w", dir=parent, delete=False, encoding="utf-8"
    ) as tf:
        json.dump(sanitized, tf, indent=2, default=str)
        temp_name = tf.name
    os.replace(temp_name, file_path)


def _atomic_write_parquet(file_path: Path, df: pd.DataFrame) -> None:
    """Write DataFrame to parquet atomically via a temporary file."""
    parent = file_path.parent
    parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("wb", dir=parent, delete=False) as tf:
        temp_name = tf.name
    df.to_parquet(temp_name, engine="pyarrow", index=False)
    os.replace(temp_name, file_path)


@dataclass(slots=True)
class RunRecord:
    """Metadata summary record for a stored backtest simulation."""

    run_id: str
    strategy_name: str
    created_at: str
    config: dict[str, Any]
    n_trades: int
    final_equity: float
    start_date: str | None
    end_date: str | None
    extras: dict[str, Any]
