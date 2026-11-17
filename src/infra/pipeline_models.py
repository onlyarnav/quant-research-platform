"""
Pipeline execution data models and report schemas.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class StepStatus(str, Enum):
    """Execution status for an orchestrator pipeline step."""

    SUCCESS = "success"
    FAILED = "failed"
    SKIPPED = "skipped"


@dataclass(slots=True)
class StepResult:
    """Outcome and execution metadata for a single pipeline step."""

    name: str
    status: StepStatus
    started_at: str
    finished_at: str
    duration_seconds: float
    message: str = ""
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Convert StepResult to a dictionary."""
        return {
            "name": self.name,
            "status": self.status.value,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "duration_seconds": self.duration_seconds,
            "message": self.message,
            "details": self.details,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> StepResult:
        """Construct StepResult from a dictionary."""
        return cls(
            name=data["name"],
            status=StepStatus(data["status"]),
            started_at=data["started_at"],
            finished_at=data["finished_at"],
            duration_seconds=float(data["duration_seconds"]),
            message=data.get("message", ""),
            details=data.get("details", {}),
        )


@dataclass(slots=True)
class PipelineRunReport:
    """Consolidated report across all pipeline execution steps."""

    run_id: str
    started_at: str
    finished_at: str
    steps: list[StepResult]
    config: dict[str, Any]

    @property
    def all_succeeded(self) -> bool:
        """True if every step succeeded."""
        return bool(self.steps) and all(
            s.status == StepStatus.SUCCESS for s in self.steps
        )

    @property
    def total_duration_seconds(self) -> float:
        """Sum of duration of all steps."""
        return sum(s.duration_seconds for s in self.steps)

    def to_dict(self) -> dict[str, Any]:
        """Convert PipelineRunReport to a dictionary."""
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "finished_at": self.finished_at,
            "steps": [s.to_dict() for s in self.steps],
            "config": self.config,
            "all_succeeded": self.all_succeeded,
            "total_duration_seconds": self.total_duration_seconds,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> PipelineRunReport:
        """Construct PipelineRunReport from a dictionary."""
        steps = [StepResult.from_dict(s) for s in data.get("steps", [])]
        return cls(
            run_id=data["run_id"],
            started_at=data["started_at"],
            finished_at=data["finished_at"],
            steps=steps,
            config=data.get("config", {}),
        )
