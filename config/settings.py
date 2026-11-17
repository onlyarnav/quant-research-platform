"""
Master configuration loader for the Quant AI Research Platform.

Loads all settings from environment variables via .env file.
All engineers should import settings from here — never use os.getenv() directly.

Usage:
    from config.settings import settings
    print(settings.DATABASE_URL)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from config.settings_validators import (
    validate_allocation_method,
    validate_app_env,
    validate_log_format,
    validate_non_negative_float,
    validate_optional_str,
    validate_pipeline_run_time,
    validate_positive_float,
    validate_positive_int,
    validate_ratio_in_zero_one,
    validate_rebalance_frequency,
    validate_stop_and_take_profit,
    validate_update_mode,
    validate_workers,
)

_PROJECT_ROOT = Path(__file__).parent.parent


class Settings(BaseSettings):
    """Central settings object for the platform."""

    model_config = SettingsConfigDict(
        env_file=_PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
        extra="ignore",
    )

    # ── Database ──────────────────────────────────────────────
    DATABASE_URL: str = Field(
        default="postgresql://postgres:postgres@localhost:5432/quant_research"
    )

    # ── External API Keys (optional integrations) ─────────────
    ALPHA_VANTAGE_KEY: str = Field(default="")
    FRED_API_KEY: str = Field(default="")
    NEWS_API_KEY: str = Field(default="")

    # ── Data Settings ─────────────────────────────────────────
    DATA_START_DATE: str = Field(default="2015-01-01")
    DATA_UPDATE_MODE: str = Field(default="incremental")
    DATA_REFRESH_INTERVAL_HOURS: int = Field(default=24)

    # ── Data Lake Paths ───────────────────────────────────────
    DATA_RAW_PATH: Path = _PROJECT_ROOT / "data" / "raw"
    DATA_PROCESSED_PATH: Path = _PROJECT_ROOT / "data" / "processed"
    DATA_FEATURES_PATH: Path = _PROJECT_ROOT / "data" / "features"
    DATA_SIGNALS_PATH: Path = _PROJECT_ROOT / "data" / "signals"
    DATA_PREDICTIONS_PATH: Path = _PROJECT_ROOT / "data" / "predictions"

    # ── Market Universe ───────────────────────────────────────
    MARKET: str = Field(default="NSE")
    STOCK_UNIVERSE: str = Field(default="NIFTY500")

    # ── ML Settings ───────────────────────────────────────────
    MODEL_LOOKBACK_WINDOW: int = Field(default=252)
    MODEL_PREDICTION_HORIZON: int = Field(default=1)
    MODEL_RETRAIN_INTERVAL_DAYS: int = Field(default=7)
    TRAIN_SPLIT_RATIO: float = Field(default=0.7)
    VAL_SPLIT_RATIO: float = Field(default=0.15)
    MODEL_ARTIFACT_PATH: Path = _PROJECT_ROOT / "models"
    MLFLOW_EXPERIMENT_NAME: str = Field(default="quant-ml-research")
    FEATURE_SET_VERSION: str = Field(default="v1")
    WALK_FORWARD_SPLITS: int = Field(default=5)
    DRIFT_PSI_THRESHOLD: float = Field(default=0.2)
    DRIFT_MIN_ROWS: int = Field(default=200)
    RETRAIN_PROMOTION_TOLERANCE: float = Field(default=0.0)
    SIGNAL_THRESHOLD: float = Field(default=0.01)
    SIGNAL_TOP_N: int = Field(default=10)
    SIGNAL_BOTTOM_N: int = Field(default=0)

    # ── Backtesting ───────────────────────────────────────────
    TRANSACTION_COST: float = Field(default=0.001)
    SLIPPAGE: float = Field(default=0.0005)
    INITIAL_CAPITAL: float = Field(default=1_000_000.0)

    # ── Portfolio ─────────────────────────────────────────────
    PORTFOLIO_RISK_FREE_RATE: float = Field(default=0.07)
    MAX_POSITION_SIZE: float = Field(default=0.05)

    # ── Pipeline Automation ───────────────────────────────────
    PIPELINE_RUN_TIME: str = Field(default="18:30")
    PIPELINE_TIMEZONE: str = Field(default="Asia/Kolkata")

    # ── MLflow ────────────────────────────────────────────────
    MLFLOW_TRACKING_URI: str = Field(default="sqlite:///mlflow.db")

    # ── Quantitative Strategy, Portfolio & Pipeline Settings ───
    BACKTEST_RESULTS_PATH: Path = _PROJECT_ROOT / "data" / "backtests"
    BACKTEST_EXECUTION_LAG: int = Field(default=0)
    BACKTEST_CASH_RESERVE: float = Field(default=0.0)
    BACKTEST_ALLOCATION_METHOD: str = Field(default="fixed_notional")
    BACKTEST_REBALANCE_FREQUENCY: str = Field(default="on_signal")
    BACKTEST_REBALANCE_TOLERANCE: float = Field(default=0.0025)
    BACKTEST_STOP_LOSS: float | None = Field(default=None)
    BACKTEST_TAKE_PROFIT: float | None = Field(default=None)
    BACKTEST_LOOKBACK_WINDOW: int = Field(default=60)
    BENCHMARK_SYMBOL: str = Field(default="NIFTY50")
    TRADING_DAYS_PER_YEAR: int = Field(default=252)
    ROLLING_WINDOW: int = Field(default=63)
    VAR_CONFIDENCE: float = Field(default=0.95)
    PIPELINE_RUNS_PATH: Path = _PROJECT_ROOT / "data" / "pipeline_runs"
    API_HOST: str = Field(default="127.0.0.1")
    API_PORT: int = Field(default=8000)
    DASHBOARD_API_URL: str | None = Field(default=None)
    DASHBOARD_API_KEY: str | None = Field(default=None)
    DASHBOARD_CACHE_TTL_SECONDS: int = Field(default=60)

    # ── Production Environment & Hardening Settings ───────────
    APP_ENV: str = Field(default="dev")
    LOG_LEVEL: str = Field(default="INFO")
    LOG_FORMAT: str = Field(default="text")
    API_ROOT_PATH: str = Field(default="")
    API_WORKERS: int = Field(default=1)
    API_CORS_ORIGINS: str = Field(default="")
    API_DOCS_ENABLED: bool = Field(default=True)
    DASHBOARD_READ_ONLY: bool = Field(default=False)
    DASHBOARD_PASSWORD: str | None = Field(default=None)
    BACKTEST_MAX_CONCURRENT_JOBS: int = Field(default=1)

    # ── Validators ────────────────────────────────────────────
    @field_validator("DATA_UPDATE_MODE")
    @classmethod
    def val_update_mode(cls, v: str) -> str:
        return validate_update_mode(v)

    @field_validator("TRAIN_SPLIT_RATIO", "VAL_SPLIT_RATIO")
    @classmethod
    def val_splits(cls, v: float, info: Any) -> float:
        return validate_ratio_in_zero_one(v, info.field_name)

    @model_validator(mode="after")
    def validate_split_ratios_sum(self) -> Settings:
        if self.TRAIN_SPLIT_RATIO + self.VAL_SPLIT_RATIO >= 1.0:
            raise ValueError(
                f"TRAIN_SPLIT_RATIO + VAL_SPLIT_RATIO must be < 1.0, "
                f"got {self.TRAIN_SPLIT_RATIO + self.VAL_SPLIT_RATIO}"
            )
        return self

    @field_validator("WALK_FORWARD_SPLITS", "DRIFT_MIN_ROWS", "SIGNAL_TOP_N")
    @classmethod
    def val_pos_ints(cls, v: int, info: Any) -> int:
        return validate_positive_int(v, info.field_name)

    @field_validator("DRIFT_PSI_THRESHOLD")
    @classmethod
    def val_pos_floats(cls, v: float, info: Any) -> float:
        return validate_positive_float(v, info.field_name)

    @field_validator(
        "RETRAIN_PROMOTION_TOLERANCE",
        "SIGNAL_BOTTOM_N",
        "SIGNAL_THRESHOLD",
        "BACKTEST_EXECUTION_LAG",
    )
    @classmethod
    def val_non_neg(cls, v: float, info: Any) -> float:
        return validate_non_negative_float(v, info.field_name)

    @field_validator("BACKTEST_CASH_RESERVE")
    @classmethod
    def val_cash_reserve(cls, v: float) -> float:
        if not (0.0 <= v < 1.0):
            raise ValueError(f"BACKTEST_CASH_RESERVE must be in [0, 1), got {v}")
        return v

    @field_validator("BACKTEST_ALLOCATION_METHOD")
    @classmethod
    def val_alloc(cls, v: str) -> str:
        return validate_allocation_method(v)

    @field_validator("BACKTEST_REBALANCE_FREQUENCY")
    @classmethod
    def val_rebal(cls, v: str) -> str:
        return validate_rebalance_frequency(v)

    @field_validator("BACKTEST_STOP_LOSS", "BACKTEST_TAKE_PROFIT", mode="before")
    @classmethod
    def val_stop_tp(cls, v: Any) -> float | None:
        return validate_stop_and_take_profit(v)

    @field_validator("ROLLING_WINDOW")
    @classmethod
    def val_roll_win(cls, v: int) -> int:
        return validate_positive_int(v, "ROLLING_WINDOW", min_val=2)

    @field_validator("VAR_CONFIDENCE")
    @classmethod
    def val_var_conf(cls, v: float) -> float:
        return validate_ratio_in_zero_one(v, "VAR_CONFIDENCE")

    @field_validator("TRADING_DAYS_PER_YEAR")
    @classmethod
    def val_trade_days(cls, v: int) -> int:
        return validate_positive_int(v, "TRADING_DAYS_PER_YEAR", min_val=1)

    @field_validator("API_PORT")
    @classmethod
    def val_port(cls, v: int) -> int:
        if not (1 <= v <= 65535):
            raise ValueError(f"API_PORT must be in 1-65535, got {v}")
        return v

    @field_validator(
        "DASHBOARD_API_URL",
        "DASHBOARD_API_KEY",
        "DASHBOARD_PASSWORD",
        mode="before",
    )
    @classmethod
    def val_opt_str(cls, v: Any) -> str | None:
        return validate_optional_str(v)

    @field_validator("APP_ENV")
    @classmethod
    def val_env(cls, v: str) -> str:
        return validate_app_env(v)

    @field_validator("LOG_FORMAT")
    @classmethod
    def val_log_fmt(cls, v: str) -> str:
        return validate_log_format(v)

    @field_validator("PIPELINE_RUN_TIME")
    @classmethod
    def val_pipe_time(cls, v: str) -> str:
        return validate_pipeline_run_time(v)

    @field_validator("API_WORKERS")
    @classmethod
    def val_workers(cls, v: int) -> int:
        return validate_workers(v)

    @model_validator(mode="after")
    def validate_prod_security(self) -> Settings:
        if self.APP_ENV == "prod":
            if not self.DASHBOARD_READ_ONLY and not self.DASHBOARD_API_KEY:
                raise ValueError(
                    "In prod, DASHBOARD_API_KEY must be configured when DASHBOARD_READ_ONLY=False."
                )
        return self


settings = Settings()
