"""
Backtest job submission and status polling endpoints.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Request, status

from src.dashboard.api.security import verify_api_key
from src.dashboard.schemas import BacktestJobStatus, BacktestRequest, sanitize

router = APIRouter(
    prefix="/backtests", tags=["Backtests"], dependencies=[Depends(verify_api_key)]
)


@router.post("", status_code=status.HTTP_202_ACCEPTED)
def submit_backtest(req: BacktestRequest, request: Request) -> dict[str, str]:
    """Submit a backtest task to execute in the background."""
    service = request.app.state.service
    try:
        job_id = service.submit_backtest(req.model_dump())
        return {"job_id": job_id}
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY, detail=str(exc)
        )
    except Exception:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to schedule backtest.",
        )


@router.get("/{job_id}", response_model=BacktestJobStatus)
def get_job_status(job_id: str, request: Request) -> Any:
    """Poll status of a background backtest execution."""
    service = request.app.state.service
    try:
        res = service.job_status(job_id)
        return sanitize(res)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Job {job_id} not found."
        )
