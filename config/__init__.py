"""
Configuration package for the Quant AI Research Platform.

This module provides a clean public interface for configuration.

Engineers should import settings and constants from this package instead of
directly importing from internal config modules.

Examples:
    from config import settings
    from config import TRADING_DAYS_PER_YEAR, SIGNAL_BUY
"""

from config.constants import (
    # Calendar constants
    TRADING_DAYS_PER_YEAR,
    TRADING_DAYS_PER_MONTH,
    TRADING_DAYS_PER_WEEK,

    # Market session constants
    NSE_MARKET_OPEN,
    NSE_MARKET_CLOSE,
    NSE_TIMEZONE,

    # Exchanges / venues
    EXCHANGE_NSE,
    EXCHANGE_BSE,
    EXCHANGE_BINANCE,
    EXCHANGE_FX,
    EXCHANGE_FRED,
    SUPPORTED_EXCHANGES,

    # Asset classes
    ASSET_CLASS_EQUITY,
    ASSET_CLASS_INDEX,
    ASSET_CLASS_ETF,
    ASSET_CLASS_FOREX,
    ASSET_CLASS_COMMODITY,
    ASSET_CLASS_CRYPTO,
    ASSET_CLASS_MACRO,
    SUPPORTED_ASSET_CLASSES,

    # Signal encoding
    SIGNAL_BUY,
    SIGNAL_HOLD,
    SIGNAL_SELL,

    # Data sources
    SOURCE_YFINANCE,
    SOURCE_ALPHA_VANTAGE,
    SOURCE_FRED,
    SOURCE_NEWSAPI,
    SUPPORTED_DATA_SOURCES,

    # Timeframes
    TIMEFRAME_HOURLY,
    TIMEFRAME_DAILY,
    TIMEFRAME_WEEKLY,
    TIMEFRAME_MONTHLY,
    SUPPORTED_TIMEFRAMES,

    # Database table names
    TABLE_ASSETS,
    TABLE_PRICES,
    TABLE_FEATURES,
    TABLE_SIGNALS,
    TABLE_TRADES,
    TABLE_PORTFOLIO_METRICS,

    # Model constants
    MIN_TRAIN_SAMPLES,
    INITIAL_CAPITAL_INR
)

from config.settings import Settings, settings

__all__ = [
    # Settings
    "Settings",
    "settings",

    # Calendar constants
    "TRADING_DAYS_PER_YEAR",
    "TRADING_DAYS_PER_MONTH",
    "TRADING_DAYS_PER_WEEK",

    # Market session constants
    "NSE_MARKET_OPEN",
    "NSE_MARKET_CLOSE",
    "NSE_TIMEZONE",

    # Exchanges / venues
    "EXCHANGE_NSE",
    "EXCHANGE_BSE",
    "EXCHANGE_BINANCE",
    "EXCHANGE_FX",
    "EXCHANGE_FRED",
    "SUPPORTED_EXCHANGES",

    # Asset classes
    "ASSET_CLASS_EQUITY",
    "ASSET_CLASS_INDEX",
    "ASSET_CLASS_ETF",
    "ASSET_CLASS_FOREX",
    "ASSET_CLASS_COMMODITY",
    "ASSET_CLASS_CRYPTO",
    "ASSET_CLASS_MACRO",
    "SUPPORTED_ASSET_CLASSES",

    # Signal encoding
    "SIGNAL_BUY",
    "SIGNAL_HOLD",
    "SIGNAL_SELL",

    # Data sources
    "SOURCE_YFINANCE",
    "SOURCE_ALPHA_VANTAGE",
    "SOURCE_FRED",
    "SOURCE_NEWSAPI",
    "SUPPORTED_DATA_SOURCES",

    # Timeframes
    "TIMEFRAME_HOURLY",
    "TIMEFRAME_DAILY",
    "TIMEFRAME_WEEKLY",
    "TIMEFRAME_MONTHLY",
    "SUPPORTED_TIMEFRAMES",

    # Database table names
    "TABLE_ASSETS",
    "TABLE_PRICES",
    "TABLE_FEATURES",
    "TABLE_SIGNALS",
    "TABLE_TRADES",
    "TABLE_PORTFOLIO_METRICS",

    # Model constants
    "MIN_TRAIN_SAMPLES",

    # Portfolio constants
    "INITIAL_CAPITAL_INR"
]