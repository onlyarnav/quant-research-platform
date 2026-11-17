"""
Unit tests for pipeline data models and run history persistence.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from src.infra.pipeline_models import PipelineRunReport, StepResult, StepStatus
from src.infra.run_history import RunHistory


def test_step_result_and_report_properties() -> None:
    step1 = StepResult(
        name="step1",
        status=StepStatus.SUCCESS,
        started_at="2024-01-01T00:00:00",
        finished_at="2024-01-01T00:01:00",
        duration_seconds=60.0,
    )
    step2 = StepResult(
        name="step2",
        status=StepStatus.FAILED,
        started_at="2024-01-01T00:01:00",
        finished_at="2024-01-01T00:01:10",
        duration_seconds=10.0,
        message="Failure reason",
    )

    rep = PipelineRunReport(
        run_id="run_test",
        started_at="2024-01-01T00:00:00",
        finished_at="2024-01-01T00:01:10",
        steps=[step1, step2],
        config={"asset_class": "equity"},
    )

    assert not rep.all_succeeded
    assert rep.total_duration_seconds == pytest.approx(70.0)

    # Serialization roundtrip
    d = rep.to_dict()
    restored = PipelineRunReport.from_dict(d)
    assert restored.run_id == rep.run_id
    assert len(restored.steps) == 2
    assert restored.steps[0].status == StepStatus.SUCCESS
    assert restored.steps[1].status == StepStatus.FAILED


def test_run_history_save_and_list() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        history = RunHistory(root=Path(tmp_dir))
        assert history.list() == []
        assert history.latest() is None

        rep = PipelineRunReport(
            run_id="20240101T000000Z_pipeline",
            started_at="2024-01-01T00:00:00",
            finished_at="2024-01-01T00:01:00",
            steps=[],
            config={},
        )
        saved_path = history.save(rep)
        assert saved_path.exists()

        runs = history.list()
        assert len(runs) == 1
        assert runs[0].run_id == rep.run_id

        latest = history.latest()
        assert latest is not None
        assert latest.run_id == rep.run_id
