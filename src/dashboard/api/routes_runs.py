"""
Backtest run analytics and time-series endpoints.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from src.dashboard.api.security import verify_api_key
from src.dashboard.config import MAX_PAGE_SIZE
from src.dashboard.schemas import (
    MetricsResponse,
    Page,
    RunSummary,
    SeriesResponse,
    sanitize,
)

router = APIRouter(
    prefix="/runs", tags=["Runs"], dependencies=[Depends(verify_api_key)]
)


def _to_series(df) -> dict[str, Any]:
    """Helper to pack DataFrame into SeriesResponse dict."""
    if df.empty:
        return {"columns": list(df.columns), "rows": []}
    return {
        "columns": list(df.columns),
        "rows": df.to_dict(orient="records"),
    }


@router.get("", response_model=list[RunSummary])
def list_runs(request: Request, limit: int = Query(100, ge=1, le=1000)) -> Any:
    """List summary records of completed backtests."""
    service = request.app.state.service
    return sanitize(service.list_runs(limit=limit))


@router.get("/{run_id}", response_model=RunSummary)
def get_run(run_id: str, request: Request) -> Any:
    """Get metadata and configuration for a backtest run."""
    service = request.app.state.service
    try:
        return sanitize(service.get_run(run_id))
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Run {run_id} not found."
        )


@router.get("/{run_id}/metrics", response_model=MetricsResponse)
def get_metrics(run_id: str, request: Request) -> Any:
    """Get performance, trade stats, and benchmark metrics."""
    service = request.app.state.service
    try:
        return sanitize(service.get_metrics(run_id))
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Run {run_id} not found."
        )


@router.get("/{run_id}/equity", response_model=SeriesResponse)
def get_equity(run_id: str, request: Request) -> Any:
    """Get portfolio equity curve and cumulative returns."""
    service = request.app.state.service
    try:
        df = service.get_equity(run_id)
        return sanitize(_to_series(df))
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Run {run_id} not found."
        )


@router.get("/{run_id}/drawdown", response_model=SeriesResponse)
def get_drawdown(run_id: str, request: Request) -> Any:
    """Get portfolio underwater drawdown series."""
    service = request.app.state.service
    try:
        df = service.get_drawdown(run_id)
        return sanitize(_to_series(df))
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Run {run_id} not found."
        )


@router.get("/{run_id}/rolling", response_model=SeriesResponse)
def get_rolling(run_id: str, request: Request, window: int = Query(63)) -> Any:
    """Get rolling Sharpe ratio and volatility curves."""
    if window < 2:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Rolling window must be >= 2.",
        )
    service = request.app.state.service
    try:
        df = service.get_rolling(run_id, window=window)
        return sanitize(_to_series(df))
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Run {run_id} not found."
        )


@router.get("/{run_id}/monthly-returns", response_model=SeriesResponse)
def get_monthly_returns(run_id: str, request: Request) -> Any:
    """Get compounded monthly returns matrix."""
    service = request.app.state.service
    try:
        df = service.get_monthly_returns(run_id)
        return sanitize(_to_series(df))
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Run {run_id} not found."
        )


@router.get("/{run_id}/trades", response_model=Page[dict[str, Any]])
def get_trades(
    run_id: str,
    request: Request,
    symbol: str | None = Query(None),
    start: str | None = Query(None),
    end: str | None = Query(None),
    limit: int = Query(100, ge=1),
    offset: int = Query(0, ge=0),
) -> Any:
    """Get paginated trade execution records."""
    if limit > MAX_PAGE_SIZE:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f"Requested limit {limit} exceeds maximum allowable page size {MAX_PAGE_SIZE}.",
        )
    service = request.app.state.service
    try:
        items, total = service.get_trades(
            run_id, symbol=symbol, start=start, end=end, limit=limit, offset=offset
        )
        return sanitize(
            {"items": items, "total": total, "limit": limit, "offset": offset}
        )
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Run {run_id} not found."
        )


@router.get("/{run_id}/trade-distribution", response_model=SeriesResponse)
def get_trade_distribution(run_id: str, request: Request) -> Any:
    """Get trade return percentage histogram bins."""
    service = request.app.state.service
    try:
        df = service.get_trade_distribution(run_id)
        return sanitize(_to_series(df))
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Run {run_id} not found."
        )


@router.get("/{run_id}/allocation", response_model=SeriesResponse)
def get_allocation(
    run_id: str, request: Request, on_date: str | None = Query(None)
) -> Any:
    """Get portfolio weights allocation."""
    service = request.app.state.service
    try:
        df = service.get_allocation(run_id, on_date=on_date)
        return sanitize(_to_series(df))
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Run {run_id} not found."
        )


@router.get("/{run_id}/benchmark")
def get_benchmark(run_id: str, request: Request) -> Any:
    """Get benchmark comparison statistics."""
    service = request.app.state.service
    try:
        res = service.get_benchmark(run_id)
        return sanitize(res)
    except FileNotFoundError:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail=f"Run {run_id} not found."
        )
