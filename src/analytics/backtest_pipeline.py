"""
Cross-layer backtest pipeline orchestrator.

Coordinates data loading, simulation execution (legacy or capital-constrained weighted),
benchmark generation, performance reporting, and persistence.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import pandas as pd

from src.analytics.benchmark_builder import BenchmarkBuilder
from src.analytics.metrics_persistence import persist_metrics, persist_trades
from src.analytics.report_builder import ReportBuilder, StrategyReport
from src.backtesting.backtest_config import BacktestConfig
from src.backtesting.backtest_store import BacktestStore
from src.backtesting.portfolio_simulator import PortfolioSimulator
from src.backtesting.price_loader import PriceLoader
from src.backtesting.signal_loader import SignalLoader
from src.backtesting.simulator_models import SimulationOutput
from src.backtesting.weighted_simulator import WeightedPortfolioSimulator
from src.database.sessions import get_session
from src.portfolio.allocator_factory import create_allocator
from src.portfolio.constraints import PortfolioConstraints
from src.utils.logger import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class BacktestRunResult:
    """Container for complete backtest run artifacts."""

    run_id: str
    config: BacktestConfig
    output: SimulationOutput
    report: StrategyReport
    run_dir: Path | None


class BacktestPipeline:
    """High-level runner connecting signals, prices, simulation, and analytics."""

    def __init__(
        self,
        price_loader: PriceLoader | None = None,
        signal_loader: SignalLoader | None = None,
        store: BacktestStore | None = None,
        report_builder: ReportBuilder | None = None,
        benchmark_builder: BenchmarkBuilder | None = None,
        sector_map: Mapping[str, str] | None = None,
    ) -> None:
        """Initialize BacktestPipeline with pluggable loaders and builders."""
        self.price_loader = price_loader or PriceLoader()
        self.signal_loader = signal_loader or SignalLoader()
        self.store = store or BacktestStore()
        self.report_builder = report_builder or ReportBuilder()
        self.benchmark_builder = benchmark_builder or BenchmarkBuilder()
        self.sector_map = dict(sector_map or {})

    def _convert_legacy_output(
        self, legacy_res, config: BacktestConfig
    ) -> SimulationOutput:
        """Convert a legacy PortfolioSimulationResult into a SimulationOutput."""
        hist = legacy_res.portfolio_history_df
        trades = legacy_res.trades_df

        trade_details = trades.copy() if not trades.empty else pd.DataFrame()
        if not trade_details.empty and "exit_reason" not in trade_details.columns:
            trade_details["exit_reason"] = "signal"

        daily_details = pd.DataFrame(
            {
                "date": hist["date"],
                "portfolio_value": hist["portfolio_value"],
                "cash": hist["cash"],
                "invested_capital": hist["invested_capital"],
                "long_exposure": hist["invested_capital"],
                "short_exposure": 0.0,
                "net_exposure": hist["invested_capital"],
                "gross_exposure": hist["invested_capital"],
                "n_positions": 0,
                "turnover": 0.0,
                "rebalanced": False,
            }
        )

        positions_hist = pd.DataFrame(
            columns=[
                "date",
                "symbol",
                "direction",
                "quantity",
                "entry_price",
                "current_price",
                "market_value",
                "unrealized_pnl",
            ]
        )

        summary = {
            "total_trades": float(len(legacy_res.trades)),
            "total_pnl": float(trades["pnl"].sum()) if not trades.empty else 0.0,
            "final_equity": (
                float(hist["portfolio_value"].iloc[-1])
                if not hist.empty
                else config.initial_capital
            ),
        }

        return SimulationOutput(
            trades=legacy_res.trades,
            trades_df=trades,
            trade_details_df=trade_details,
            portfolio_history_df=hist,
            daily_details_df=daily_details,
            positions_history_df=positions_hist,
            summary=summary,
        )

    def run(
        self,
        config: BacktestConfig,
        symbols: Sequence[str] | None = None,
        asset_class: str = "equity",
        timeframe: str = "daily",
        start: pd.Timestamp | str | None = None,
        end: pd.Timestamp | str | None = None,
        persist: bool = True,
        persist_db: bool = False,
        signals_df: pd.DataFrame | None = None,
        prices_df: pd.DataFrame | None = None,
    ) -> BacktestRunResult:
        """
        Execute full end-to-end backtest pipeline.

        Args:
            config: Backtest configuration.
            symbols: Optional symbols to load. Defaults to intersection of available symbols.
            asset_class: Asset class folder.
            timeframe: Data timeframe.
            start: Start date.
            end: End date.
            persist: Whether to persist files to disk.
            persist_db: Whether to persist metrics and trades to PostgreSQL.
            signals_df: Injected signals DataFrame.
            prices_df: Injected prices DataFrame.

        Returns:
            BacktestRunResult with output, report, and paths.
        """
        resolved_config = config.resolved()
        resolved_config.validate()

        # 1. Load data if not injected
        if signals_df is None:
            signals_df = self.signal_loader.load_signals(
                symbols=symbols,
                asset_class=asset_class,
                timeframe=timeframe,
                start=start,
                end=end,
            )

        if prices_df is None:
            load_syms = symbols or (
                sorted(signals_df["symbol"].unique()) if not signals_df.empty else None
            )
            prices_df = self.price_loader.load_prices(
                symbols=load_syms,
                asset_class=asset_class,
                timeframe=timeframe,
                start=start,
                end=end,
            )

        # Validate symbol overlap
        sig_syms = set(signals_df["symbol"].unique()) if not signals_df.empty else set()
        px_syms = set(prices_df["symbol"].unique()) if not prices_df.empty else set()
        common = sig_syms.intersection(px_syms)
        if not common:
            raise ValueError("No common symbols between signals and prices.")

        # Filter to common symbols
        sig_filtered = signals_df[signals_df["symbol"].isin(common)].reset_index(
            drop=True
        )
        px_filtered = prices_df[prices_df["symbol"].isin(common)].reset_index(drop=True)

        # 2. Execute simulation
        if resolved_config.is_legacy_compatible():
            sim = PortfolioSimulator(
                initial_capital=resolved_config.initial_capital,
                max_position_size=resolved_config.max_position_size,
                transaction_cost=resolved_config.transaction_cost,
                slippage=resolved_config.slippage,
            )
            legacy_res = sim.run(signals_df=sig_filtered, prices_df=px_filtered)
            output = self._convert_legacy_output(legacy_res, resolved_config)
        else:
            constraints = PortfolioConstraints(
                max_position_size=resolved_config.max_position_size,
                cash_reserve=resolved_config.cash_reserve,
                sector_map=self.sector_map,
            )
            allocator = create_allocator(
                method=resolved_config.allocation_method,
                constraints=constraints,
                lookback_window=resolved_config.lookback_window,
            )
            weighted_sim = WeightedPortfolioSimulator(
                config=resolved_config, allocator=allocator
            )
            output = weighted_sim.run(signals_df=sig_filtered, prices_df=px_filtered)

        # 3. Build benchmark & report
        bench_hist, bench_lbl = self.benchmark_builder.build(
            prices_df=px_filtered,
            initial_capital=resolved_config.initial_capital,
            dates=output.portfolio_history_df["date"],
            prefer_index=True,
        )

        report = self.report_builder.build(
            portfolio_history_df=output.portfolio_history_df,
            trades_df=output.trades_df,
            benchmark_history_df=bench_hist,
            benchmark_label=bench_lbl,
            daily_details_df=output.daily_details_df,
        )

        # 4. Persistence
        run_id = f"run_{uuid.uuid4().hex[:12]}"
        run_dir = None

        if persist:
            run_dir = self.store.save_run(run_id, resolved_config, output)
            self.store.save_json(run_id, "metrics.json", report.metrics)
            self.store.save_json(run_id, "report.json", report.to_dict())

        if persist_db:
            try:
                with get_session() as session:
                    persist_metrics(
                        session, resolved_config.strategy_name, report.metrics
                    )
                    if not output.trades_df.empty:
                        persist_trades(session, output.trades_df)
            except Exception as exc:
                logger.error("Failed to persist backtest results to database: %s", exc)

        return BacktestRunResult(
            run_id=run_id,
            config=resolved_config,
            output=output,
            report=report,
            run_dir=run_dir,
        )

    def rebuild_report(self, run_id: str) -> StrategyReport:
        """Recompute StrategyReport from persisted disk artifacts."""
        record, output = self.store.load_run(run_id)
        config = (
            BacktestConfig.from_dict(record.config)
            if isinstance(record.config, dict)
            else record.config
        )
        init_cap = (
            float(config.initial_capital)
            if hasattr(config, "initial_capital") and config.initial_capital is not None
            else 100_000.0
        )

        bench_hist, bench_lbl = None, None
        try:
            bench_hist, bench_lbl = self.benchmark_builder.build(
                initial_capital=init_cap,
                dates=output.portfolio_history_df["date"],
                prefer_index=True,
            )
        except Exception as exc:
            logger.info("Benchmark unavailable for report rebuild: %s", exc)

        return self.report_builder.build(
            portfolio_history_df=output.portfolio_history_df,
            trades_df=output.trades_df,
            benchmark_history_df=bench_hist,
            benchmark_label=bench_lbl,
            daily_details_df=output.daily_details_df,
        )
