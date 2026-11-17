"""
Dashboard client abstraction layer.

Provides a unified interface (DashboardClient Protocol) with two interchangeable
implementations: LocalClient (in-process calls) and ApiClient (HTTP calls via httpx).
"""

from __future__ import annotations

from typing import Any, Mapping, Protocol

import httpx
import pandas as pd

from src.dashboard.config import DashboardConfig
from src.dashboard.services import DashboardService
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DashboardClientError(Exception):
    """Raised by ApiClient when an API call fails with non-2xx status."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(f"API Error [{status_code}]: {detail}")
        self.status_code = status_code
        self.detail = detail


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


class ApiClient:
    """HTTP implementation communicating with FastAPI backend via httpx."""

    def __init__(
        self,
        http_client: httpx.Client | None = None,
        base_url: str = "",
        api_key: str | None = None,
        timeout: float = 10.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        headers = {"X-API-Key": api_key} if api_key else {}
        self.client = http_client or httpx.Client(
            base_url=self.base_url, headers=headers, timeout=timeout
        )

    def _request(self, method: str, path: str, **kwargs: Any) -> Any:
        url = (
            f"{self.base_url}{path}"
            if self.base_url and not path.startswith("http")
            else path
        )
        try:
            resp = self.client.request(method, url, **kwargs)
        except DashboardClientError:
            raise
        except Exception as exc:
            raise DashboardClientError(
                503, f"Failed to connect to backend API: {exc}"
            ) from exc
        if resp.status_code >= 400:
            try:
                detail = resp.json().get("detail", resp.text)
            except Exception:
                detail = resp.text
            raise DashboardClientError(resp.status_code, str(detail))
        return resp.json()

    def list_runs(self, limit: int = 100) -> list[dict[str, Any]]:
        return self._request("GET", "/runs", params={"limit": limit})

    def get_run(self, run_id: str) -> dict[str, Any]:
        return self._request("GET", f"/runs/{run_id}")

    def get_metrics(self, run_id: str) -> dict[str, Any]:
        return self._request("GET", f"/runs/{run_id}/metrics")

    def get_equity(self, run_id: str) -> pd.DataFrame:
        data = self._request("GET", f"/runs/{run_id}/equity")
        return pd.DataFrame(data.get("rows", []))

    def get_drawdown(self, run_id: str) -> pd.DataFrame:
        data = self._request("GET", f"/runs/{run_id}/drawdown")
        return pd.DataFrame(data.get("rows", []))

    def get_rolling(self, run_id: str, window: int = 63) -> pd.DataFrame:
        data = self._request(
            "GET", f"/runs/{run_id}/rolling", params={"window": window}
        )
        return pd.DataFrame(data.get("rows", []))

    def get_monthly_returns(self, run_id: str) -> pd.DataFrame:
        data = self._request("GET", f"/runs/{run_id}/monthly-returns")
        return pd.DataFrame(data.get("rows", []))

    def get_trades(
        self,
        run_id: str,
        symbol: str | None = None,
        start: str | None = None,
        end: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        raw = {
            "symbol": symbol,
            "start": start,
            "end": end,
            "limit": limit,
            "offset": offset,
        }
        params = {k: v for k, v in raw.items() if v is not None}
        data = self._request("GET", f"/runs/{run_id}/trades", params=params)
        return data.get("items", []), data.get("total", 0)

    def get_trade_distribution(self, run_id: str) -> pd.DataFrame:
        data = self._request("GET", f"/runs/{run_id}/trade-distribution")
        return pd.DataFrame(data.get("rows", []))

    def get_allocation(self, run_id: str, on_date: str | None = None) -> pd.DataFrame:
        params = {"on_date": on_date} if on_date else {}
        data = self._request("GET", f"/runs/{run_id}/allocation", params=params)
        return pd.DataFrame(data.get("rows", []))

    def get_benchmark(self, run_id: str) -> dict[str, Any] | None:
        return self._request("GET", f"/runs/{run_id}/benchmark")

    def latest_signals(
        self,
        asset_class: str = "equity",
        timeframe: str = "daily",
        as_of: str | None = None,
    ) -> pd.DataFrame:
        params = {"asset_class": asset_class, "timeframe": timeframe, "as_of": as_of}
        params = {k: v for k, v in params.items() if v is not None}
        data = self._request("GET", "/signals/latest", params=params)
        return pd.DataFrame(data.get("rows", []))

    def signal_counts(
        self,
        asset_class: str = "equity",
        timeframe: str = "daily",
        as_of: str | None = None,
    ) -> tuple[str | None, dict[str, int]]:
        params = {"asset_class": asset_class, "timeframe": timeframe, "as_of": as_of}
        params = {k: v for k, v in params.items() if v is not None}
        data = self._request("GET", "/signals/counts", params=params)
        return data.get("date"), data.get("counts", {})

    def list_models(self) -> list[dict[str, Any]]:
        return self._request("GET", "/models")

    def pipeline_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        return self._request("GET", "/pipeline/runs", params={"limit": limit})

    def health(self) -> dict[str, Any]:
        return self._request("GET", "/health")

    def submit_backtest(self, req: Mapping[str, Any]) -> str:
        data = self._request("POST", "/backtests", json=dict(req))
        return data["job_id"]

    def job_status(self, job_id: str) -> dict[str, Any]:
        return self._request("GET", f"/backtests/{job_id}")


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
