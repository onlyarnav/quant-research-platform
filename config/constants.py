"""
Project-wide constants for the Quant AI Research Platform.

These are fixed domain values — not environment-specific config.
Never put secrets or environment-specific values here.

Usage:
    from config.constants import TRADING_DAYS_PER_YEAR, SIGNAL_BUY
"""

# ── Calendar Constants ────────────────────────────────────────
TRADING_DAYS_PER_YEAR: int = 252
TRADING_DAYS_PER_MONTH: int = 21
TRADING_DAYS_PER_WEEK: int = 5

# ── Market Constants ──────────────────────────────────────────
NSE_MARKET_OPEN: str = "09:15"
NSE_MARKET_CLOSE: str = "15:30"
NSE_TIMEZONE: str = "Asia/Kolkata"

# ── Asset Classes ─────────────────────────────────────────────
ASSET_CLASS_EQUITY: str = "equity"
ASSET_CLASS_INDEX: str = "index"
ASSET_CLASS_ETF: str = "etf"
ASSET_CLASS_FOREX: str = "forex"
ASSET_CLASS_COMMODITY: str = "commodity"
ASSET_CLASS_CRYPTO: str = "crypto"
ASSET_CLASS_MACRO: str = "macro"

SUPPORTED_ASSET_CLASSES: list[str] = [
    ASSET_CLASS_EQUITY,
    ASSET_CLASS_INDEX,
    ASSET_CLASS_ETF,
    ASSET_CLASS_FOREX,
    ASSET_CLASS_COMMODITY,
    ASSET_CLASS_CRYPTO,
    ASSET_CLASS_MACRO,
]

# ── Signal Encoding ───────────────────────────────────────────
SIGNAL_BUY: int = 1
SIGNAL_HOLD: int = 0
SIGNAL_SELL: int = -1

# ── Data Sources ──────────────────────────────────────────────
SOURCE_YFINANCE: str = "yfinance"
SOURCE_ALPHA_VANTAGE: str = "alpha_vantage"
SOURCE_FRED: str = "fred"
SOURCE_BINANCE: str = "binance"
SOURCE_COINGECKO: str = "coingecko"

# ── File Naming ───────────────────────────────────────────────
TIMEFRAME_DAILY: str = "daily"
TIMEFRAME_HOURLY: str = "hourly"
TIMEFRAME_WEEKLY: str = "weekly"

# ── Data Lake Paths ───────────────────────────────────────────
DATA_RAW_PATH: str = "data/raw"
DATA_PROCESSED_PATH: str = "data/processed"
DATA_FEATURES_PATH: str = "data/features"
DATA_SIGNALS_PATH: str = "data/signals"

# ── Database Table Names ──────────────────────────────────────
TABLE_ASSETS: str = "assets"
TABLE_PRICES: str = "prices"
TABLE_FEATURES: str = "features"
TABLE_SIGNALS: str = "signals"
TABLE_TRADES: str = "trades"
TABLE_PORTFOLIO_METRICS: str = "portfolio_metrics"

# ── Model Constants ───────────────────────────────────────────
INITIAL_CAPITAL_INR: float = 1_000_000.0
MIN_TRAIN_SAMPLES: int = 252