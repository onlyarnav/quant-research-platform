"""
FastAPI application factory and middleware configuration.
"""

from __future__ import annotations

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.dashboard.api.routes_backtests import router as backtests_router
from src.dashboard.api.routes_runs import router as runs_router
from src.dashboard.api.routes_signals import router as signals_router
from src.dashboard.api.routes_system import router as system_router
from src.dashboard.config import DashboardConfig
from src.dashboard.services import DashboardService
from src.utils.logger import get_logger

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
    app = FastAPI(
        title="Quant Research Platform API",
        version="1.0.0",
        description="REST API for quantitative backtesting, portfolio analytics, and production pipelines.",
    )

    app.state.config = config or DashboardConfig()
    app.state.service = service or DashboardService()

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

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
