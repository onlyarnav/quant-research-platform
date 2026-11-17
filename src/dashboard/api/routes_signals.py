"""
Trading signals query endpoints.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status

from src.dashboard.api.security import verify_api_key
from src.dashboard.schemas import SeriesResponse, SignalCounts, sanitize

router = APIRouter(
    prefix="/signals", tags=["Signals"], dependencies=[Depends(verify_api_key)]
)


@router.get("/latest", response_model=SeriesResponse)
def get_latest_signals(
    request: Request,
    asset_class: str = Query("equity"),
    timeframe: str = Query("daily"),
    as_of: str | None = Query(None),
) -> Any:
    """Retrieve the most recent trade signal per symbol."""
    service = request.app.state.service
    try:
        df = service.latest_signals(
            asset_class=asset_class, timeframe=timeframe, as_of=as_of
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Signal data unavailable: {exc}",
        )

    if df.empty:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No signals found for specified parameters.",
        )

    return sanitize(
        {
            "columns": list(df.columns),
            "rows": df.to_dict(orient="records"),
        }
    )


@router.get("/counts", response_model=SignalCounts)
def get_signal_counts(
    request: Request,
    asset_class: str = Query("equity"),
    timeframe: str = Query("daily"),
    as_of: str | None = Query(None),
) -> Any:
    """Retrieve aggregate counts of buy, neutral, and sell signals."""
    service = request.app.state.service
    try:
        latest_date, counts = service.signal_counts(
            asset_class=asset_class, timeframe=timeframe, as_of=as_of
        )
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f"Signal counts unavailable: {exc}",
        )

    if latest_date is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="No signal files found.",
        )

    return sanitize({"date": latest_date, "counts": counts})
