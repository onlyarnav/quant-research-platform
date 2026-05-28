"""
Market universe definitions for the Quant AI Research Platform.

Equity universe is loaded dynamically from the NIFTY 500 CSV downloaded
from the NSE website.

Index universe is hardcoded as it is a small fixed list.

Usage:
    from src.data_pipeline.universe import get_full_universe
    assets = get_full_universe()
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from config.constants import (
    ASSET_CLASS_EQUITY,
    ASSET_CLASS_INDEX,
    EXCHANGE_NSE,
    TIMEFRAME_DAILY,
)
from src.data_pipeline.ingestion import AssetConfig

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).parent.parent.parent
_NIFTY500_CSV_PATH = _PROJECT_ROOT / "data" / "universe" / "nifty500.csv"


# ── NSE Index Universe (hardcoded fixed list) ─────────────────
_NSE_INDEX_SYMBOLS: list[dict[str, str]] = [
    {"symbol": "^NSEI", "currency": "INR"},       # NIFTY 50
    {"symbol": "^NSEBANK", "currency": "INR"},    # NIFTY BANK
    {"symbol": "^CNXAUTO", "currency": "INR"},    # NIFTY AUTO
    {"symbol": "^CNXIT", "currency": "INR"},      # NIFTY IT
    {"symbol": "^CNXPHARMA", "currency": "INR"},  # NIFTY PHARMA
    {"symbol": "^CNXFMCG", "currency": "INR"},    # NIFTY FMCG
    {"symbol": "^CNXMETAL", "currency": "INR"},   # NIFTY METAL
    {"symbol": "^CNXREALTY", "currency": "INR"},  # NIFTY REALTY
]


def get_nse_equity_universe(
    timeframe: str = TIMEFRAME_DAILY,
) -> list[AssetConfig]:
    """
    Load NIFTY 500 equity universe from CSV.

    Args:
        timeframe: Data timeframe. Defaults to daily.

    Returns:
        List of AssetConfig objects for all NIFTY 500 constituents.

    Raises:
        FileNotFoundError: If nifty500.csv is not found.
        ValueError: If CSV is missing required columns.
    """
    if not _NIFTY500_CSV_PATH.exists():
        raise FileNotFoundError(
            f"NIFTY 500 universe file not found at {_NIFTY500_CSV_PATH}. "
            "Download it from NSE website and place it at data/universe/nifty500.csv"
        )

    df = pd.read_csv(_NIFTY500_CSV_PATH)

    required_columns = {"Symbol", "Industry"}
    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"NIFTY 500 CSV missing required columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    df["Symbol"] = df["Symbol"].str.strip()
    df = df[df["Symbol"].notna() & (df["Symbol"] != "")]

    assets = [
        AssetConfig(
            symbol=f"{row['Symbol']}.NS",
            asset_class=ASSET_CLASS_EQUITY,
            exchange=EXCHANGE_NSE,
            currency="INR",
            timeframe=timeframe,
        )
        for _, row in df.iterrows()
    ]

    logger.info("Loaded %s equities from NIFTY 500 universe.", len(assets))

    return assets


def get_nse_index_universe(
    timeframe: str = TIMEFRAME_DAILY,
) -> list[AssetConfig]:
    """
    Return AssetConfig list for NSE index universe.

    Args:
        timeframe: Data timeframe. Defaults to daily.

    Returns:
        List of AssetConfig objects for NSE indices.
    """
    return [
        AssetConfig(
            symbol=asset["symbol"],
            asset_class=ASSET_CLASS_INDEX,
            exchange=EXCHANGE_NSE,
            currency=asset["currency"],
            timeframe=timeframe,
        )
        for asset in _NSE_INDEX_SYMBOLS
    ]


def get_full_universe(
    timeframe: str = TIMEFRAME_DAILY,
) -> list[AssetConfig]:
    """
    Return combined NIFTY 500 equity and index universe.

    Args:
        timeframe: Data timeframe. Defaults to daily.

    Returns:
        Combined list of all AssetConfig objects.
    """
    equities = get_nse_equity_universe(timeframe)
    indices = get_nse_index_universe(timeframe)
    total = equities + indices

    logger.info(
        "Full universe loaded. equities=%s indices=%s total=%s",
        len(equities),
        len(indices),
        len(total),
    )

    return total