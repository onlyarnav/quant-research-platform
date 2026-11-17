"""
HTTP client implementation for remote FastAPI dashboard communication.
"""

from __future__ import annotations

from typing import Any, Mapping

import httpx
import pandas as pd

from src.utils.logger import get_logger

logger = get_logger(__name__)


class DashboardClientError(Exception):
    """Raised by ApiClient when an API call fails with non-2xx status."""

    def __init__(self, status_code: int, detail: str) -> None:
        super().__init__(f"API Error [{status_code}]: {detail}")
        self.status_code = status_code
        self.detail = detail


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
