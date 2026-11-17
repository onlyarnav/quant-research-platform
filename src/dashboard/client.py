"""
Dashboard client abstraction layer.

Provides a unified interface (DashboardClient Protocol) with two interchangeable
implementations: LocalClient (in-process calls) and ApiClient (HTTP calls via httpx).
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol

import pandas as pd

from src.dashboard.api_client import ApiClient, DashboardClientError
from src.dashboard.config import DashboardConfig
from src.dashboard.services import DashboardService
from src.utils.logger import get_logger

logger = get_logger(__name__)

__all__ = [
    "DashboardClient",
    "DashboardClientError",
    "LocalClient",
    "ApiClient",
    "get_client",
]


class DashboardClient(Protocol):
    """Protocol defining common interface for local and remote dashboard operations."""

    def list_runs(self, limit: int = 100) -> list[dict[str, Any]]: ...
    def get_run(self, run_id: str) -> dict[str, Any]: ...
    def get_metrics(self, run_id: str) -> dict[str, Any]: ...
    def get_equity(self, run_id: str) -> pd.DataFrame: ...
    def get_drawdown(self, run_id: str) -> pd.DataFrame: ...
    def get_rolling(self, run_id: str, window: int = 63) -> pd.DataFrame: ...
    def get_monthly_returns(self, run_id: str) -> pd.DataFrame: ...

    def get_trades(
        self,
        run_id: str,
        symbol: str | None = None,
        start: str | None = None,
        end: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]: ...

    def get_trade_distribution(self, run_id: str) -> pd.DataFrame: ...

    def get_allocation(
        self, run_id: str, on_date: str | None = None
    ) -> pd.DataFrame: ...

    def get_benchmark(self, run_id: str) -> dict[str, Any] | None: ...

    def latest_signals(
        self,
        asset_class: str = "equity",
        timeframe: str = "daily",
        as_of: str | None = None,
    ) -> pd.DataFrame: ...

    def signal_counts(
        self,
        asset_class: str = "equity",
        timeframe: str = "daily",
        as_of: str | None = None,
    ) -> tuple[str | None, dict[str, int]]: ...
    def list_models(self) -> list[dict[str, Any]]: ...
    def pipeline_runs(self, limit: int = 20) -> list[dict[str, Any]]: ...
    def health(self) -> dict[str, Any]: ...
    def submit_backtest(self, req: Mapping[str, Any]) -> str: ...
    def job_status(self, job_id: str) -> dict[str, Any]: ...


class LocalClient:
    """In-process implementation directly invoking DashboardService."""

    def __init__(self, service: DashboardService | None = None) -> None:
        self.service = service or DashboardService()

    def list_runs(self, limit: int = 100) -> list[dict[str, Any]]:
        return self.service.list_runs(limit=limit)

    def get_run(self, run_id: str) -> dict[str, Any]:
        return self.service.get_run(run_id)

    def get_metrics(self, run_id: str) -> dict[str, Any]:
        return self.service.get_metrics(run_id)

    def get_equity(self, run_id: str) -> pd.DataFrame:
        return self.service.get_equity(run_id)

    def get_drawdown(self, run_id: str) -> pd.DataFrame:
        return self.service.get_drawdown(run_id)

    def get_rolling(self, run_id: str, window: int = 63) -> pd.DataFrame:
        return self.service.get_rolling(run_id, window=window)

    def get_monthly_returns(self, run_id: str) -> pd.DataFrame:
        return self.service.get_monthly_returns(run_id)

    def get_trades(
        self,
        run_id: str,
        symbol: str | None = None,
        start: str | None = None,
        end: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        return self.service.get_trades(
            run_id, symbol=symbol, start=start, end=end, limit=limit, offset=offset
        )

    def get_trade_distribution(self, run_id: str) -> pd.DataFrame:
        return self.service.get_trade_distribution(run_id)

    def get_allocation(self, run_id: str, on_date: str | None = None) -> pd.DataFrame:
        return self.service.get_allocation(run_id, on_date=on_date)

    def get_benchmark(self, run_id: str) -> dict[str, Any] | None:
        return self.service.get_benchmark(run_id)

    def latest_signals(
        self,
        asset_class: str = "equity",
        timeframe: str = "daily",
        as_of: str | None = None,
    ) -> pd.DataFrame:
        return self.service.latest_signals(
            asset_class=asset_class, timeframe=timeframe, as_of=as_of
        )

    def signal_counts(
        self,
        asset_class: str = "equity",
        timeframe: str = "daily",
        as_of: str | None = None,
    ) -> tuple[str | None, dict[str, int]]:
        return self.service.signal_counts(
            asset_class=asset_class, timeframe=timeframe, as_of=as_of
        )

    def list_models(self) -> list[dict[str, Any]]:
        return self.service.list_models()

    def pipeline_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        return self.service.pipeline_runs(limit=limit)

    def health(self) -> dict[str, Any]:
        return self.service.health()

    def submit_backtest(self, req: Mapping[str, Any]) -> str:
        return self.service.submit_backtest(req)

    def job_status(self, job_id: str) -> dict[str, Any]:
        return self.service.job_status(job_id)


def get_client(config: DashboardConfig | None = None) -> DashboardClient:
    """Factory creating appropriate client based on settings."""
    cfg = config or DashboardConfig()
    if cfg.api_url:
        try:
            client = ApiClient(base_url=cfg.api_url, api_key=cfg.api_key, timeout=1.5)
            client.health()
            return client
        except Exception:
            logger.warning(
                "API at %s unreachable, falling back to LocalClient", cfg.api_url
            )
    return LocalClient(DashboardService(cache_ttl_seconds=cfg.cache_ttl_seconds))
