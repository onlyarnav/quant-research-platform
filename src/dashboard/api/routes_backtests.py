"""
Backtest job submission and status polling endpoints.
"""

from __future__ import annotations

import re
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status

from config.settings import settings
from src.dashboard.api.security import verify_api_key
from src.dashboard.schemas import BacktestJobStatus, BacktestRequest, sanitize

SAFE_ID_RE = re.compile(r"^[A-Za-z0-9_.-]+$")

router = APIRouter(
    prefix="/backtests", tags=["Backtests"], dependencies=[Depends(verify_api_key)]
)


@router.post("", status_code=status.HTTP_202_ACCEPTED)
def submit_backtest(req: BacktestRequest, request: Request) -> dict[str, str]:
    """Submit a backtest task to execute in the background."""
    if getattr(settings, "DASHBOARD_READ_ONLY", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Platform is running in read-only mode.",
        )

    service = request.app.state.service
    try:
        job_id = service.submit_backtest(req.model_dump())
        return {"job_id": job_id}
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        )
    except RuntimeError as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS, detail=str(exc)
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to schedule backtest.",
        )


@router.get("/{job_id}", response_model=BacktestJobStatus)
def get_job_status(job_id: str, request: Request) -> Any:
    """Poll status of a background backtest execution."""
    if not SAFE_ID_RE.match(job_id):
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Invalid job_id format: '{job_id}'",
        )
    service = request.app.state.service
    try:
        res = service.job_status(job_id)
        return sanitize(res)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Job {job_id} not found."
        )
