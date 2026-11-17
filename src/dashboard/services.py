"""
Dashboard business logic service layer.

Provides unified, cached read-only and backtest-submission operations across
runs, analytics, signals, health, and pipeline history.
"""

from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from config.settings import settings
from src.analytics.backtest_pipeline import BacktestPipeline
from src.analytics.report_builder import StrategyReport
from src.analytics.rolling_metrics import rolling_sharpe, rolling_volatility
from src.backtesting.backtest_store import BacktestStore
from src.backtesting.signal_loader import SignalLoader
from src.dashboard.job_manager import BacktestJobManager
from src.infra.health import HealthChecker
from src.infra.run_history import RunHistory
from src.utils.logger import get_logger

logger = get_logger(__name__)


class DashboardService:
    """Service layer backing the Dashboard API and local UI clients."""

    def __init__(
        self,
        store: BacktestStore | None = None,
        pipeline: BacktestPipeline | None = None,
        signal_loader: SignalLoader | None = None,
        run_history: RunHistory | None = None,
        health_checker: HealthChecker | None = None,
        cache_ttl_seconds: int = 60,
    ) -> None:
        """Initialize DashboardService with injected infrastructure."""
        self.store = store or BacktestStore()
        self.pipeline = pipeline or BacktestPipeline(store=self.store)
        self.signal_loader = signal_loader or SignalLoader()
        self.run_history = run_history or RunHistory()
        self.health_checker = health_checker or HealthChecker()
        self.cache_ttl = cache_ttl_seconds
        self.job_manager = BacktestJobManager()

        self._report_cache: dict[str, tuple[float, StrategyReport]] = {}
        self._lock = threading.Lock()

    def _get_report(self, run_id: str) -> StrategyReport:
        """Fetch StrategyReport for run_id, using memory cache or disk rebuild."""
        now = time.time()
        with self._lock:
            if run_id in self._report_cache:
                cached_time, rep = self._report_cache[run_id]
                if now - cached_time < self.cache_ttl:
                    return rep

        run_dir = self.store.root / run_id
        if not run_dir.exists():
            raise FileNotFoundError(f"Run {run_id} not found.")

        rep_file = run_dir / "report.json"
        if rep_file.exists():
            try:
                with open(rep_file, "r", encoding="utf-8") as f:
                    d = json.load(f)
                keys = ("equity", "drawdown", "rolling_sharpe", "rolling_volatility",
                        "monthly_returns", "trade_distribution", "pnl_by_symbol", "exposure")
                rep = StrategyReport(
                    metrics=d.get("metrics", {}),
                    trade_stats=d.get("trade_stats", {}),
                    benchmark=d.get("benchmark"),
                    benchmark_label=d.get("benchmark_label"),
                    **{k: pd.DataFrame(d.get(k, [])) for k in keys},
                )
                with self._lock:
                    self._report_cache[run_id] = (now, rep)
                return rep
            except Exception as exc:
                logger.warning("Failed to load report.json for %s: %s", run_id, exc)

        rep = self.pipeline.rebuild_report(run_id)
        with self._lock:
            self._report_cache[run_id] = (now, rep)
        return rep

    def list_runs(self, limit: int = 100) -> list[dict[str, Any]]:
        """List summary records of completed backtests."""
        return [
            {
                "run_id": r.run_id,
                "strategy_name": r.strategy_name,
                "created_at": r.created_at,
                "n_trades": r.n_trades,
                "final_equity": r.final_equity,
                "start_date": r.start_date,
                "end_date": r.end_date,
                "config": r.config,
            }
            for r in self.store.list_runs()[:limit]
        ]

    def get_run(self, run_id: str) -> dict[str, Any]:
        """Get summary and config for a specific run."""
        record, _ = self.store.load_run(run_id)
        return {
            "run_id": record.run_id,
            "strategy_name": record.strategy_name,
            "created_at": record.created_at,
            "n_trades": record.n_trades,
            "final_equity": record.final_equity,
            "start_date": record.start_date,
            "end_date": record.end_date,
            "config": record.config,
        }

    def get_metrics(self, run_id: str) -> dict[str, Any]:
        """Get performance, trade stats, and benchmark comparison for a run."""
        rep = self._get_report(run_id)
        return {
            "metrics": rep.metrics,
            "trade_stats": rep.trade_stats,
            "benchmark": rep.benchmark,
        }

    def get_equity(self, run_id: str) -> pd.DataFrame:
        """Get equity series DataFrame."""
        return self._get_report(run_id).equity

    def get_drawdown(self, run_id: str) -> pd.DataFrame:
        """Get drawdown series DataFrame."""
        return self._get_report(run_id).drawdown

    def get_rolling(self, run_id: str, window: int = 63) -> pd.DataFrame:
        """Get rolling Sharpe and volatility DataFrame."""
        if window < 2:
            raise ValueError("Rolling window must be at least 2.")
        _, output = self.store.load_run(run_id)
        hist = output.portfolio_history_df
        s_df, v_df = rolling_sharpe(hist, window=window), rolling_volatility(hist, window=window)
        if s_df.empty or v_df.empty:
            return pd.DataFrame(columns=["date", "rolling_sharpe", "rolling_volatility"])
        return pd.merge(s_df, v_df, on="date")

    def get_monthly_returns(self, run_id: str) -> pd.DataFrame:
        """Get monthly returns table."""
        return self._get_report(run_id).monthly_returns

    def get_trades(
        self,
        run_id: str,
        symbol: str | None = None,
        start: str | None = None,
        end: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """Get filtered and paginated trade records."""
        _, output = self.store.load_run(run_id)
        df = output.trades_df
        if df.empty:
            return [], 0
        df = df.copy()
        if "entry_date" in df.columns:
            df = df.sort_values("entry_date", ascending=False)
        if symbol:
            df = df[df["symbol"] == symbol]
        if start:
            df = df[pd.to_datetime(df["entry_date"]) >= pd.to_datetime(start)]
        if end:
            df = df[pd.to_datetime(df["exit_date"]) <= pd.to_datetime(end)]
        return df.iloc[offset : offset + limit].to_dict(orient="records"), len(df)

    def get_trade_distribution(self, run_id: str) -> pd.DataFrame:
        """Get trade return distribution histogram bins."""
        return self._get_report(run_id).trade_distribution

    def get_allocation(self, run_id: str, on_date: str | None = None) -> pd.DataFrame:
        """Get holdings weights (latest or as of a specific date)."""
        _, output = self.store.load_run(run_id)
        pos_df = output.positions_history_df
        if pos_df.empty:
            return pd.DataFrame(columns=["symbol", "weight", "market_value"])
        if on_date is not None:
            pos_df = pos_df[pd.to_datetime(pos_df["date"]) == pd.to_datetime(on_date)]
        else:
            pos_df = pos_df[pos_df["date"] == pos_df["date"].max()]
        tot_val = pos_df["market_value"].sum() if not pos_df.empty else 0.0
        weights = (pos_df["market_value"] / tot_val).fillna(0.0) if tot_val > 0 else 0.0
        return pd.DataFrame({
            "symbol": pos_df["symbol"], "market_value": pos_df["market_value"], "weight": weights
        }).reset_index(drop=True)

    def get_benchmark(self, run_id: str) -> dict[str, Any] | None:
        """Get benchmark comparison metrics."""
        return self._get_report(run_id).benchmark

    def latest_signals(
        self,
        asset_class: str = "equity",
        timeframe: str = "daily",
        as_of: str | None = None,
    ) -> pd.DataFrame:
        """Get the latest signal per symbol."""
        df = self.signal_loader.load_signals(
            asset_class=asset_class, timeframe=timeframe
        )
        cols = ["date", "symbol", "signal", "confidence", "rank"]
        if df.empty:
            return pd.DataFrame(columns=cols)

        df["date"] = pd.to_datetime(df["date"])
        if as_of:
            df = df[df["date"] <= pd.to_datetime(as_of)]
        if df.empty:
            return pd.DataFrame(columns=cols)

        return df.sort_values("date").groupby("symbol").last().reset_index()

    def signal_counts(
        self,
        asset_class: str = "equity",
        timeframe: str = "daily",
        as_of: str | None = None,
    ) -> tuple[str | None, dict[str, int]]:
        """Get counts of buy (+1), neutral (0), and sell (-1) signals."""
        signals = self.latest_signals(
            asset_class=asset_class, timeframe=timeframe, as_of=as_of
        )
        if signals.empty:
            return None, {"buy": 0, "neutral": 0, "sell": 0}

        d0 = signals["date"].max()
        latest_date = d0.isoformat() if hasattr(d0, "isoformat") else str(d0)
        counts = {
            "buy": int((signals["signal"] > 0).sum()),
            "neutral": int((signals["signal"] == 0).sum()),
            "sell": int((signals["signal"] < 0).sum()),
        }
        return latest_date, counts

    def list_models(self) -> list[dict[str, Any]]:
        """List trained models available in registry."""
        reg_dir = Path(getattr(settings, "MODEL_ARTIFACT_PATH", Path("models")))
        if not reg_dir.exists():
            return []
        models: list[dict[str, Any]] = []
        seen: set[str] = set()
        for p in sorted(reg_dir.iterdir()):
            latest_f = p / "LATEST"
            if p.is_dir() and latest_f.exists():
                ver = latest_f.read_text(encoding="utf-8").strip()
                meta_f = p / ver / "metadata.json"
                if meta_f.exists():
                    try:
                        d = json.loads(meta_f.read_text(encoding="utf-8"))
                        models.append({
                            "name": d.get("model_name", p.name),
                            "version": d.get("version", ver),
                            "created_at": d.get("created_at", time.ctime(p.stat().st_ctime)),
                            "metrics": d.get("test_metrics") or d.get("val_metrics") or {},
                        })
                        seen.add(p.name)
                    except Exception as exc:
                        logger.warning("Error reading metadata for %s: %s", p.name, exc)
        for f in reg_dir.glob("*.pkl"):
            if f.stem not in seen:
                models.append({
                    "name": f.stem,
                    "version": "1.0",
                    "created_at": time.ctime(f.stat().st_ctime),
                    "metrics": {},
                })
        return models

    def pipeline_runs(self, limit: int = 20) -> list[dict[str, Any]]:
        """List recent pipeline execution reports."""
        return [r.to_dict() for r in self.run_history.list(limit=limit)]

    def health(self) -> dict[str, Any]:
        """Run system health checks."""
        return self.health_checker.run_all().to_dict()

    def submit_backtest(self, req: Mapping[str, Any]) -> str:
        """Submit a background backtest job."""
        return self.job_manager.submit(req, self.pipeline)

    def job_status(self, job_id: str) -> dict[str, Any]:
        """Fetch status of background job."""
        return self.job_manager.get_status(job_id)
