"""
Centralized logging configuration for the Quant AI Research Platform.

All modules must use this instead of configuring logging independently.

Usage:
    from src.utils.logger import get_logger
    logger = get_logger(__name__)
"""

from __future__ import annotations

import contextvars
import json
import logging
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from config.settings import settings

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
_initialized = False

# Context variable for tracing request IDs across async API calls
request_id_ctx: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "request_id", default=None
)

_URL_PASSWORD_RE = re.compile(r"://([^:]+):([^@]+)@")
_TOKEN_SECRET_RE = re.compile(
    r"(?i)(api[_-]?key|password|token|secret)\s*[:=]\s*['\"]?([A-Za-z0-9_-]{8,})['\"]?"
)


def mask_secrets(text: str) -> str:
    """Mask credentials, passwords in URLs, and tokens in log strings."""
    if not text:
        return text
    # Mask URL passwords: ://user:password@ -> ://user:***@
    masked = _URL_PASSWORD_RE.sub(r"://\1:***@", text)
    # Mask API keys and secrets:
    masked = _TOKEN_SECRET_RE.sub(r"\1=***", masked)
    return masked


class JsonFormatter(logging.Formatter):
    """Formats log records as structured JSON lines for CloudWatch ingestion."""

    def format(self, record: logging.LogRecord) -> str:
        msg = mask_secrets(record.getMessage())
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": msg,
        }
        req_id = request_id_ctx.get()
        if req_id is not None:
            payload["request_id"] = req_id

        if record.exc_info:
            exc_text = self.formatException(record.exc_info)
            payload["exception"] = mask_secrets(exc_text)

        return json.dumps(payload)


class SecretMaskingTextFormatter(logging.Formatter):
    """Standard text formatter that masks secrets in messages."""

    def format(self, record: logging.LogRecord) -> str:
        formatted = super().format(record)
        return mask_secrets(formatted)


def _get_log_level() -> int:
    """Read log level from settings, default INFO."""
    level = getattr(settings, "LOG_LEVEL", "INFO").upper()
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
    log_format = getattr(settings, "LOG_FORMAT", "text").lower()
    handlers: list[logging.Handler] = []

    formatter: logging.Formatter
    if log_format == "json":
        formatter = JsonFormatter()
    else:
        formatter = SecretMaskingTextFormatter(_LOG_FORMAT, _DATE_FORMAT)

    # stdout handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    handlers.append(console_handler)

    # optional file handler
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
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
