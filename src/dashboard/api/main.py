"""
FastAPI application factory and middleware configuration.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Callable
from typing import Any

from fastapi import FastAPI, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from sqlalchemy.exc import SQLAlchemyError

from config.settings import settings
from src.dashboard.api.routes_backtests import router as backtests_router
from src.dashboard.api.routes_runs import router as runs_router
from src.dashboard.api.routes_signals import router as signals_router
from src.dashboard.api.routes_system import router as system_router
from src.dashboard.config import DashboardConfig
from src.dashboard.services import DashboardService
from src.utils.logger import get_logger, request_id_ctx

logger = get_logger(__name__)


def create_app(
    service: DashboardService | None = None,
    config: DashboardConfig | None = None,
) -> FastAPI:
    """
    Construct and configure the FastAPI application.

    Args:
        service: Injected DashboardService instance. Defaults to default DashboardService().
        config: Injected DashboardConfig instance. Defaults to DashboardConfig().

    Returns:
        Configured FastAPI app.
    """
    docs_enabled = getattr(settings, "API_DOCS_ENABLED", True)
    root_path = getattr(settings, "API_ROOT_PATH", "")

    app = FastAPI(
        title="Quant Research Platform API",
        version="1.0.0",
        description="REST API for quantitative backtesting, portfolio analytics, and production pipelines.",
        root_path=root_path,
        docs_url="/docs" if docs_enabled else None,
        redoc_url="/redoc" if docs_enabled else None,
        openapi_url="/openapi.json" if docs_enabled else None,
    )

    app.state.config = config or DashboardConfig()
    app.state.service = service or DashboardService()

    # CORS configuration
    cors_origins = [
        o.strip()
        for o in getattr(settings, "API_CORS_ORIGINS", "").split(",")
        if o.strip()
    ]
    if cors_origins:
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_origins,
            allow_credentials=True,
            allow_methods=["*"],
            allow_headers=["*"],
        )

    # Security headers and Request-ID / Access Log middleware
    @app.middleware("http")
    async def request_lifecycle_middleware(
        request: Request, call_next: Callable[[Request], Any]
    ) -> Response:
        req_id = request.headers.get("X-Request-ID") or uuid.uuid4().hex
        token = request_id_ctx.set(req_id)
        start_time = time.perf_counter()

        try:
            response: Response = await call_next(request)
        finally:
            duration_ms = (time.perf_counter() - start_time) * 1000.0
            request_id_ctx.reset(token)

        # Set security and trace headers
        response.headers["X-Request-ID"] = req_id
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["X-Frame-Options"] = "DENY"

        logger.info(
            "%s %s -> %s (%.2fms) [req_id=%s]",
            request.method,
            request.url.path,
            response.status_code,
            duration_ms,
            req_id,
        )
        return response

    @app.exception_handler(FileNotFoundError)
    async def file_not_found_handler(
        request: Request, exc: FileNotFoundError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_404_NOT_FOUND,
            content={"detail": str(exc)},
        )

    @app.exception_handler(ValueError)
    async def value_error_handler(request: Request, exc: ValueError) -> JSONResponse:
        return JSONResponse(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            content={"detail": str(exc)},
        )

    @app.exception_handler(SQLAlchemyError)
    async def database_exception_handler(
        request: Request, exc: SQLAlchemyError
    ) -> JSONResponse:
        logger.error(
            "Database error on %s %s: %s",
            request.method,
            request.url.path,
            exc,
        )
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": "Database service is temporarily unavailable."},
        )

    @app.exception_handler(Exception)
    async def generic_exception_handler(
        request: Request, exc: Exception
    ) -> JSONResponse:
        logger.exception(
            "Unhandled API exception on %s %s: %s",
            request.method,
            request.url.path,
            exc,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={"detail": "An internal server error occurred."},
        )

    app.include_router(system_router)
    app.include_router(runs_router)
    app.include_router(signals_router)
    app.include_router(backtests_router)

    return app


app = create_app()
