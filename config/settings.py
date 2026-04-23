"""
Master configuration loader for the Quant AI Research Platform.

Loads all settings from environment variables via .env file.
All engineers should import settings from here — never use os.getenv() directly.

Usage:
    from config.settings import settings
    print(settings.DATABASE_URL)
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Settings:
    """Central settings object for the platform."""

    # ── Database ──────────────────────────────────────────────
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL",
        "postgresql://postgres:postgres@localhost:5432/quantdb"
    )

    # ── External API Keys ─────────────────────────────────────
    ALPHA_VANTAGE_KEY: str = os.getenv("ALPHA_VANTAGE_KEY", "")
    FRED_API_KEY: str = os.getenv("FRED_API_KEY", "")
    NEWS_API_KEY: str = os.getenv("NEWS_API_KEY", "")

    # ── Data Settings ─────────────────────────────────────────
    DATA_START_DATE: str = os.getenv("DATA_START_DATE", "2015-01-01")
    DATA_UPDATE_MODE: str = os.getenv("DATA_UPDATE_MODE", "incremental")
    DATA_REFRESH_INTERVAL_HOURS: int = int(
        os.getenv("DATA_REFRESH_INTERVAL_HOURS", "24")
    )

    # ── Market Universe ───────────────────────────────────────
    MARKET: str = os.getenv("MARKET", "NSE")
    STOCK_UNIVERSE: str = os.getenv("STOCK_UNIVERSE", "NIFTY500")

    # ── ML Settings ───────────────────────────────────────────
    MODEL_LOOKBACK_WINDOW: int = int(os.getenv("MODEL_LOOKBACK_WINDOW", "252"))
    MODEL_PREDICTION_HORIZON: int = int(os.getenv("MODEL_PREDICTION_HORIZON", "1"))
    MODEL_RETRAIN_INTERVAL_DAYS: int = int(
        os.getenv("MODEL_RETRAIN_INTERVAL_DAYS", "7")
    )

    # ── Backtesting ───────────────────────────────────────────
    TRANSACTION_COST: float = float(os.getenv("TRANSACTION_COST", "0.001"))
    SLIPPAGE: float = float(os.getenv("SLIPPAGE", "0.0005"))

    # ── Portfolio ─────────────────────────────────────────────
    PORTFOLIO_RISK_FREE_RATE: float = float(
        os.getenv("PORTFOLIO_RISK_FREE_RATE", "0.02")
    )
    MAX_POSITION_SIZE: float = float(os.getenv("MAX_POSITION_SIZE", "0.05"))

    # ── Pipeline Automation ───────────────────────────────────
    PIPELINE_RUN_TIME: str = os.getenv("PIPELINE_RUN_TIME", "18:30")
    PIPELINE_TIMEZONE: str = os.getenv("PIPELINE_TIMEZONE", "Asia/Kolkata")

    # ── MLflow ────────────────────────────────────────────────
    MLFLOW_TRACKING_URI: str = os.getenv(
        "MLFLOW_TRACKING_URI", "http://localhost:5000"
    )


# Single shared instance — import this everywhere
settings = Settings()