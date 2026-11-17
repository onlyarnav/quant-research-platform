"""
Unit tests for PipelineOrchestrator failure propagation.
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

from src.infra.pipeline_models import StepStatus
from src.infra.pipeline_orchestrator import PipelineOrchestrator
from src.infra.pipeline_steps import PipelineContext
from src.infra.run_history import RunHistory


class GoodStep:
    name = "step_good"

    def run(self, context: PipelineContext) -> dict[str, Any]:
        return {"result": "ok"}


class BadStep:
    name = "step_bad"

    def run(self, context: PipelineContext) -> dict[str, Any]:
        raise RuntimeError("Step failed!")


class DownstreamStep:
    name = "step_downstream"

    def run(self, context: PipelineContext) -> dict[str, Any]:
        return {"result": "should not run"}


def test_failure_propagation_skips_downstream() -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        history = RunHistory(root=Path(tmp_dir))
        orchestrator = PipelineOrchestrator(
            steps=[GoodStep(), BadStep(), DownstreamStep()],
            history=history,
        )

        ctx = PipelineContext(run_id="test_fail_prop")
        rep = orchestrator.run(ctx)

        assert not rep.all_succeeded
        assert len(rep.steps) == 3

        assert rep.steps[0].status == StepStatus.SUCCESS
        assert rep.steps[1].status == StepStatus.FAILED
        assert rep.steps[2].status == StepStatus.SKIPPED
        assert "upstream step 'step_bad' failed" in rep.steps[2].message

        # Verify saved to disk
        saved = history.latest()
        assert saved is not None
        assert saved.run_id == "test_fail_prop"
