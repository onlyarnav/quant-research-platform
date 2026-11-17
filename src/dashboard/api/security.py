"""
API authentication and security dependencies.
"""

from __future__ import annotations

import secrets

from fastapi import Header, HTTPException, status

from config.settings import settings


def verify_api_key(x_api_key: str | None = Header(None, alias="X-API-Key")) -> None:
    """
    Validate incoming request X-API-Key header against settings.DASHBOARD_API_KEY.

    If DASHBOARD_API_KEY is unset or empty, requests are permitted (local use).
    """
    expected_key = settings.DASHBOARD_API_KEY
    if not expected_key:
        return

    if not x_api_key or not secrets.compare_digest(x_api_key, expected_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing API key.",
        )
