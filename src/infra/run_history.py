"""
Pipeline run history persistence and retrieval.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

from config.settings import settings
from src.infra.pipeline_models import PipelineRunReport
from src.utils.logger import get_logger

logger = get_logger(__name__)


class RunHistory:
    """Manages disk persistence and listing of pipeline run execution reports."""

    def __init__(self, root: Path | str | None = None) -> None:
        """Initialize RunHistory with storage root directory."""
        self.root = Path(settings.PIPELINE_RUNS_PATH) if root is None else Path(root)

    def save(self, report: PipelineRunReport) -> Path:
        """
        Atomically persist a pipeline run report to disk.

        Args:
            report: The pipeline run report to save.

        Returns:
            Path to the saved JSON file.
        """
        self.root.mkdir(parents=True, exist_ok=True)
        file_path = self.root / f"{report.run_id}.json"

        with tempfile.NamedTemporaryFile(
            "w", dir=self.root, delete=False, encoding="utf-8"
        ) as tf:
            json.dump(report.to_dict(), tf, indent=2)
            temp_name = tf.name

        os.replace(temp_name, file_path)
        logger.info("Saved pipeline run report to %s", file_path)
        return file_path

    def list(self, limit: int = 20) -> list[PipelineRunReport]:
        """
        List past pipeline runs sorted newest first.

        Args:
            limit: Maximum number of reports to return.

        Returns:
            List of PipelineRunReport instances.
        """
        if not self.root.exists():
            return []

        reports: list[PipelineRunReport] = []
        for file in sorted(self.root.glob("*.json"), reverse=True):
            try:
                with open(file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                reports.append(PipelineRunReport.from_dict(data))
                if len(reports) >= limit:
                    break
            except Exception as exc:
                logger.warning("Corrupt pipeline report in %s: %s", file, exc)

        reports.sort(key=lambda r: r.started_at, reverse=True)
        return reports[:limit]

    def latest(self) -> PipelineRunReport | None:
        """Return the most recent pipeline run report, or None if none exist."""
        reports = self.list(limit=1)
        return reports[0] if reports else None
