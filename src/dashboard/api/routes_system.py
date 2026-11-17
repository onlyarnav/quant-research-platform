"""
System health, models, and pipeline run API endpoints.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, Query, Request

from src.dashboard.api.security import verify_api_key
from src.dashboard.schemas import HealthResponse, ModelInfo, PipelineRunInfo, sanitize

router = APIRouter(tags=["System"])


@router.get("/health", response_model=HealthResponse)
def get_health(request: Request) -> Any:
    """Return operational health of database, pipelines, and models."""
    service = request.app.state.service
    res = service.health()
    return sanitize(
        {
            "status": res.get("overall", "ok"),
            "checks": res.get("checks", []),
            "generated_at": res.get("generated_at", ""),
        }
    )


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
