"""
Validation functions for platform configuration settings.
"""

from __future__ import annotations

import re
from typing import Any

TIME_RE = re.compile(r"^([01]\d|2[0-3]):([0-5]\d)$")


def validate_update_mode(v: str) -> str:
    allowed = {"full", "incremental"}
    if v not in allowed:
        raise ValueError(f"DATA_UPDATE_MODE must be one of {allowed}, got '{v}'")
    return v


def validate_ratio_in_zero_one(v: float, name: str) -> float:
    if not (0.0 < v < 1.0):
        raise ValueError(f"{name} must be in (0, 1), got {v}")
    return v


def validate_positive_int(v: int, name: str, min_val: int = 1) -> int:
    if v < min_val:
        raise ValueError(f"{name} must be >= {min_val}, got {v}")
    return v


def validate_positive_float(v: float, name: str) -> float:
    if v <= 0:
        raise ValueError(f"{name} must be > 0, got {v}")
    return v


def validate_non_negative_float(v: float, name: str) -> float:
    if v < 0:
        raise ValueError(f"{name} must be >= 0, got {v}")
    return v


def validate_allocation_method(v: str) -> str:
    allowed = {
        "fixed_notional",
        "equal_weight",
        "signal_weighted",
        "volatility_weighted",
        "risk_parity",
        "mean_variance",
    }
    if v not in allowed:
        raise ValueError(
            f"BACKTEST_ALLOCATION_METHOD must be one of {allowed}, got '{v}'"
        )
    return v


def validate_rebalance_frequency(v: str) -> str:
    allowed = {"on_signal", "daily", "weekly", "monthly"}
    if v not in allowed:
        raise ValueError(
            f"BACKTEST_REBALANCE_FREQUENCY must be one of {allowed}, got '{v}'"
        )
    return v


def validate_stop_and_take_profit(v: Any) -> float | None:
    if v is None or v == "":
        return None
    v_float = float(v)
    if not (0.0 < v_float <= 1.0):
        raise ValueError(f"Stop loss/take profit must be in (0, 1], got {v_float}")
    return v_float


def validate_optional_str(v: Any) -> str | None:
    if v == "" or v is None:
        return None
    return str(v)


def validate_app_env(v: str) -> str:
    allowed = {"dev", "staging", "prod"}
    if v not in allowed:
        raise ValueError(f"APP_ENV must be one of {allowed}, got '{v}'")
    return v


def validate_log_format(v: str) -> str:
    allowed = {"text", "json"}
    if v not in allowed:
        raise ValueError(f"LOG_FORMAT must be one of {allowed}, got '{v}'")
    return v


def validate_pipeline_run_time(v: str) -> str:
    if not TIME_RE.match(v.strip()):
        raise ValueError(f"Invalid time format '{v}'. Expected 24h format 'HH:MM'")
    return v.strip()


def validate_workers(v: int) -> int:
    if not (1 <= v <= 32):
        raise ValueError(f"API_WORKERS must be between 1 and 32, got {v}")
    return v


def validate_cors_origins(v: str) -> list[str]:
    return [orig.strip() for orig in v.split(",") if orig.strip()]
