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

from pydantic import Field, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

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
    MLFLOW_TRACKING_URI: str = Field(default="http://localhost:5000")

    # ── Validators ────────────────────────────────────────────
    @field_validator("DATA_UPDATE_MODE")
    @classmethod
    def validate_update_mode(cls, v: str) -> str:
        allowed = {"full", "incremental"}
        if v not in allowed:
            raise ValueError(f"DATA_UPDATE_MODE must be one of {allowed}, got '{v}'")
        return v

    @field_validator("TRAIN_SPLIT_RATIO")
    @classmethod
    def validate_train_split_ratio(cls, v: float) -> float:
        if not (0.0 < v < 1.0):
            raise ValueError(f"TRAIN_SPLIT_RATIO must be in (0, 1), got {v}")
        return v

    @field_validator("VAL_SPLIT_RATIO")
    @classmethod
    def validate_val_split_ratio(cls, v: float) -> float:
        if not (0.0 < v < 1.0):
            raise ValueError(f"VAL_SPLIT_RATIO must be in (0, 1), got {v}")
        return v

    @model_validator(mode="after")
    def validate_split_ratios_sum(self) -> Settings:
        if self.TRAIN_SPLIT_RATIO + self.VAL_SPLIT_RATIO >= 1.0:
            raise ValueError(
                f"TRAIN_SPLIT_RATIO + VAL_SPLIT_RATIO must be < 1.0, "
                f"got {self.TRAIN_SPLIT_RATIO + self.VAL_SPLIT_RATIO}"
            )
        return self

    @field_validator("WALK_FORWARD_SPLITS")
    @classmethod
    def validate_walk_forward_splits(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"WALK_FORWARD_SPLITS must be >= 1, got {v}")
        return v

    @field_validator("DRIFT_PSI_THRESHOLD")
    @classmethod
    def validate_drift_psi_threshold(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"DRIFT_PSI_THRESHOLD must be > 0, got {v}")
        return v

    @field_validator("DRIFT_MIN_ROWS")
    @classmethod
    def validate_drift_min_rows(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"DRIFT_MIN_ROWS must be >= 1, got {v}")
        return v

    @field_validator("RETRAIN_PROMOTION_TOLERANCE")
    @classmethod
    def validate_retrain_promotion_tolerance(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"RETRAIN_PROMOTION_TOLERANCE must be >= 0, got {v}")
        return v

    @field_validator("SIGNAL_TOP_N")
    @classmethod
    def validate_signal_top_n(cls, v: int) -> int:
        if v < 1:
            raise ValueError(f"SIGNAL_TOP_N must be >= 1, got {v}")
        return v

    @field_validator("SIGNAL_BOTTOM_N")
    @classmethod
    def validate_signal_bottom_n(cls, v: int) -> int:
        if v < 0:
            raise ValueError(f"SIGNAL_BOTTOM_N must be >= 0, got {v}")
        return v

    @field_validator("SIGNAL_THRESHOLD")
    @classmethod
    def validate_signal_threshold(cls, v: float) -> float:
        if v < 0:
            raise ValueError(f"SIGNAL_THRESHOLD must be >= 0, got {v}")
        return v


settings = Settings()
