"""
Unit tests for scheduler utilities.
"""

from __future__ import annotations

import pytest

from src.infra.pipeline_models import PipelineRunReport
from src.infra.scheduler import build_scheduler, parse_run_time, run_once


def test_parse_run_time() -> None:
    h, m = parse_run_time("08:30")
    assert h == 8
    assert m == 30

    h, m = parse_run_time("23:59")
    assert h == 23
    assert m == 59

    with pytest.raises(ValueError, match="Invalid PIPELINE_RUN_TIME"):
        parse_run_time("25:00")

    with pytest.raises(ValueError, match="Invalid PIPELINE_RUN_TIME"):
        parse_run_time("invalid")


def test_build_scheduler() -> None:
    sched = build_scheduler(run_time="17:45", timezone="Asia/Kolkata")
    job = sched.get_job("daily_pipeline")
    assert job is not None
    # Verify trigger fields
    trigger = job.trigger
    hour_field = next(f for f in trigger.fields if f.name == "hour")
    min_field = next(f for f in trigger.fields if f.name == "minute")
    assert str(hour_field) == "17"
    assert str(min_field) == "45"


def test_run_once_with_mock() -> None:
    class MockOrchestrator:
        def run(self, context):
            return PipelineRunReport(
                run_id=context.run_id,
                started_at="2024-01-01T00:00:00",
                finished_at="2024-01-01T00:01:00",
                steps=[],
                config={},
            )

    rep = run_once(orchestrator=MockOrchestrator())
    assert rep.run_id.endswith("_pipeline")
