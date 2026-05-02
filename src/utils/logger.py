"""
Centralized logging configuration for the Quant AI Research Platform.

All modules must use this instead of configuring logging independently.

Usage:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
"""

import logging
import sys
from pathlib import Path


_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_initialized = False


def _get_log_level() -> int:
    """Read log level from environment, default INFO."""
    import os
    level = os.getenv("LOG_LEVEL", "INFO").upper()
    return getattr(logging, level, logging.INFO)


def setup_logging(log_file: Path | None = None) -> None:
    """
    Configure root logger for the entire platform.

    Must be called once at application entry point (scripts, CLI, etc).
    Subsequent calls are no-ops.

    Args:
        log_file: Optional path to write logs to disk alongside stdout.
    """
    global _initialized
    if _initialized:
        return

    level = _get_log_level()
    handlers: list[logging.Handler] = []

    # stdout handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
    handlers.append(console_handler)

    # optional file handler
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(logging.Formatter(_LOG_FORMAT, _DATE_FORMAT))
        handlers.append(file_handler)

    logging.basicConfig(
        level=level,
        handlers=handlers,
        force=True,
    )

    # silence noisy third-party loggers
    logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy.pool").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    logging.getLogger("yfinance").setLevel(logging.WARNING)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    _initialized = True


def get_logger(name: str) -> logging.Logger:
    """
    Return a named logger.

    Args:
        name: Typically __name__ from the calling module.

    Returns:
        Configured Logger instance.
    """
    return logging.getLogger(name)