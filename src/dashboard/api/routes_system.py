"""
System health, models, and pipeline run API endpoints.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from sqlalchemy import text

from config.settings import settings
from src.dashboard.api.security import verify_api_key
from src.dashboard.schemas import HealthResponse, ModelInfo, PipelineRunInfo, sanitize
from src.database.sessions import get_session

router = APIRouter(tags=["System"])

DISCLAIMER_TEXT = (
    "Research and educational use only. Not investment advice. "
    "Not a SEBI-registered investment adviser or research analyst. "
    "Backtests are simulations with simplifying assumptions; past performance does not indicate future results. "
    "Market data is obtained from Yahoo Finance via yfinance and may be delayed, incomplete or inaccurate."
)


@router.get("/health", response_model=HealthResponse)
def get_health(request: Request) -> Any:
    """Cheap liveness probe returning system operational status."""
    service = request.app.state.service
    res = service.health()
    return sanitize(
        {
            "status": res.get("overall", "ok"),
            "checks": res.get("checks", []),
            "generated_at": res.get("generated_at", ""),
        }
    )


@router.get("/ready")
def get_readiness(request: Request) -> dict[str, Any]:
    """Readiness probe checking database connectivity and storage accessibility."""
    # 1. Check storage directories
    for path in [
        settings.DATA_RAW_PATH,
        settings.DATA_PROCESSED_PATH,
        settings.DATA_FEATURES_PATH,
        settings.DATA_SIGNALS_PATH,
    ]:
        if not path.exists():
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Storage path {path} not accessible.",
            )

    # 2. Check Database connectivity
    try:
        with get_session() as session:
            session.execute(text("SELECT 1"))
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Database unreachable: {exc}",
        )

    return {"status": "ready"}


@router.get("/disclaimer")
def get_disclaimer() -> dict[str, str]:
    """Return legal regulatory risk disclaimer."""
    return {"disclaimer": DISCLAIMER_TEXT}


@router.get(
    "/models", dependencies=[Depends(verify_api_key)], response_model=list[ModelInfo]
)
def list_models(request: Request) -> Any:
    """List trained machine learning model artifacts."""
    service = request.app.state.service
    models = service.list_models()
    return sanitize(models)


@router.get(
    "/pipeline/runs",
    dependencies=[Depends(verify_api_key)],
    response_model=list[PipelineRunInfo],
)
def list_pipeline_runs(request: Request, limit: int = Query(20, ge=1, le=100)) -> Any:
    """List recent automated pipeline orchestration runs."""
    service = request.app.state.service
    runs = service.pipeline_runs(limit=limit)
    return sanitize(runs)
