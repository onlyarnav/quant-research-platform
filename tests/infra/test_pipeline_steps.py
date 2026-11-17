"""
Unit tests for pipeline step adapters in dry-run mode.
"""

from __future__ import annotations

from src.infra.pipeline_steps import (
    AnalyticsStep,
    BacktestStep,
    FeatureStep,
    IngestionStep,
    PipelineContext,
    PredictionStep,
    SignalStep,
)


def test_pipeline_steps_dry_run() -> None:
    ctx = PipelineContext(
        run_id="dry_run_test",
        asset_class="equity",
        timeframe="daily",
        symbols=["AAPL"],
        dry_run=True,
    )

    steps = [
        IngestionStep(),
        FeatureStep(),
        PredictionStep(),
        SignalStep(),
        BacktestStep(),
        AnalyticsStep(),
    ]

    for step in steps:
        res = step.run(ctx)
        assert isinstance(res, dict)
        assert res.get("dry_run") is True
